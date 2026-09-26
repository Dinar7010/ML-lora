import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL_NAME = "kazzand/ru-longformer-base-4096"
MAX_LENGTH = 4096
ROLES = ["истец", "ответчик", "третье лицо"]  # порядок = индексы классов 0,1,2


class RoleClassifier(nn.Module):
    def __init__(
        self,
        base_model_name= BASE_MODEL_NAME,
        num_labels= len(ROLES),
        lora_r = 8,
        lora_alpha = 16,
        lora_dropout = 0.1,
        head_dropout = 0.1,
    ):
        super().__init__()

        base = LongformerModel.from_pretrained(base_model_name)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(base, lora_config)
        hidden_size = self.encoder.config.hidden_size
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        cls_repr = self.head_dropout(cls_repr)
        logits = self.classifier(cls_repr)
        return logits

    def print_trainable_parameters(self):
        self.encoder.print_trainable_parameters()
        head_params = sum(p.numel() for p in self.classifier.parameters())
        print(f"Параметры classifier head (все обучаемые): {head_params}")


def build_global_attention_mask(input_ids):
    mask = torch.zeros_like(input_ids)
    mask[:, 0] = 1
    return mask


def tokenize_example(tokenizer, inn, excerpt, max_length = MAX_LENGTH):
    encoded = tokenizer(
        f"ИНН: {inn}",
        excerpt,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded


if __name__ == "__main__":
    tokenizer = LongformerTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    model = RoleClassifier()
    model.print_trainable_parameters()

    encoded = tokenize_example(tokenizer, "3444048169", "по иску 3444048169 к 3444125462 ...", max_length=128)
    global_mask = build_global_attention_mask(encoded["input_ids"])

    logits = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        global_attention_mask=global_mask,
    )
    print("Форма логитов:", logits.shape)
    print("Логиты:", logits)