import json
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import LongformerTokenizerFast
from tqdm import tqdm

from model import RoleClassifier, build_global_attention_mask, BASE_MODEL_NAME, ROLES
from dataset import RoleDataset

BATCH_SIZE = 8
ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
EPOCHS = 5
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

TRAIN_PATH = "train.jsonl"
VAL_PATH = "val.jsonl"
TEST_PATH = "test.jsonl"

ADAPTER_OUT_DIR = "lora_adapter"
HEAD_OUT_PATH = "classifier_head.pt"


def compute_class_weights(train_jsonl_path):
    roles = [json.loads(line)["role"] for line in open(train_jsonl_path, encoding="utf-8")]
    counts = Counter(roles)
    total = sum(counts.values())
    weights = [total / (len(ROLES) * counts[role]) for role in ROLES]
    return torch.tensor(weights, dtype=torch.float32)


def move_batch_to_device(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "label": batch["label"].to(device),
    }


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, accum_steps=1):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = len(loader)

    if is_train:
        optimizer.zero_grad()

    progress = tqdm(loader, total=n_batches, desc="train" if is_train else "eval")

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for step, raw_batch in enumerate(progress):
            batch = move_batch_to_device(raw_batch, device)
            global_mask = build_global_attention_mask(batch["input_ids"])

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    global_attention_mask=global_mask,
                )
                loss = criterion(logits, batch["label"])

            if is_train:
                scaler.scale(loss / accum_steps).backward()

                is_accum_boundary = (step + 1) % accum_steps == 0
                is_last_batch = (step + 1) == n_batches
                if is_accum_boundary or is_last_batch:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(batch["label"].detach().cpu().numpy().tolist())
    avg_loss = total_loss / n_batches
    return avg_loss, all_preds, all_labels


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type != "cuda":
        print("cuda недоступна")
    tokenizer = LongformerTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    train_ds = RoleDataset(TRAIN_PATH, tokenizer)
    val_ds = RoleDataset(VAL_PATH, tokenizer)
    test_ds = RoleDataset(TEST_PATH, tokenizer)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = RoleClassifier(lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT).to(device)
    model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE)

    class_weights = compute_class_weights(TRAIN_PATH).to(device)
    print(f"Веса классов {ROLES}: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler, ACCUM_STEPS)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, device, criterion)

        print(f"\n Эпоха {epoch}/{EPOCHS}")
        print(f"train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
        print(classification_report(val_labels, val_preds, target_names=ROLES, zero_division=0))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.encoder.save_pretrained(ADAPTER_OUT_DIR)
            torch.save(model.classifier.state_dict(), HEAD_OUT_PATH)

    model.encoder.load_adapter(ADAPTER_OUT_DIR, adapter_name="default")
    model.classifier.load_state_dict(torch.load(HEAD_OUT_PATH, map_location=device))

    print("\n финальная оценка на test ")
    test_loss, test_preds, test_labels = run_epoch(model, test_loader, device, criterion)
    print(f"test loss: {test_loss:.4f}")
    print(classification_report(test_labels, test_preds, target_names=ROLES, zero_division=0))
    print("Confusion matrix", "):")
    print(confusion_matrix(test_labels, test_preds))


if __name__ == "__main__":
    train()