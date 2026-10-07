# model.py
import torch
import torch.nn as nn
from transformers import LongformerModel
from peft import get_peft_model, LoraConfig, TaskType  # <-- Добавляем PEFT

BASE_MODEL_NAME = "kazzand/ru-longformer-base-4096"
MAX_LENGTH = 4096
ROLES = ["истец", "ответчик", "третье лицо"]


def build_global_attention_mask(input_ids: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    global_mask = torch.zeros_like(input_ids, dtype=torch.int32)
    global_mask[:, 0] = 1
    global_mask = global_mask | target_mask.int()
    return global_mask


class RoleClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # 1. Загружаем базовую модель
        self.encoder = LongformerModel.from_pretrained(BASE_MODEL_NAME)

        # 2. Настраиваем LoRA ТОЛЬКО на query и value (как ты и планировал)
        # В Longformer модули внимания обычно называются 'query' и 'value'
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value","query_global","value_global"]
        )

        # 3. Оборачиваем энкодер. PEFT автоматически заморозит базовые веса
        # и сделает requires_grad=True только для весов LoRA.
        self.encoder = get_peft_model(self.encoder, peft_config)

        # (Опционально) Раскомментируй, чтобы увидеть красивую статистику в консоли:
        # self.encoder.print_trainable_parameters()

        # 4. Голова классификатора (всегда требует градиента)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, target_mask):
        global_attention_mask = build_global_attention_mask(input_ids, target_mask)

        # Важно: при использовании PEFT мы вызываем self.encoder как обычно,
        # он сам знает, как применить LoRA-модули внутри.
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )

        hidden_states = out.last_hidden_state
        mask_expanded = target_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        return logits