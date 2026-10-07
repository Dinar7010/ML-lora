import json
import torch
from torch.utils.data import Dataset
from model import ROLES

class RoleDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["excerpt"]
        inn = ex["inn"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True
        )
        offsets = encoding.pop("offset_mapping")[0]
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]

        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        inn_start = text.find(inn)
        if inn_start != -1:
            inn_end = inn_start + len(inn)
            for i, (start, end) in enumerate(offsets):
                if start < inn_end and end > inn_start:
                    target_mask[i] = True

        if not target_mask.any():
            raise ValueError(f"инн {inn} не попал в токены")
        role_str = ex["role"]
        label = ROLES.index(role_str)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "label": label
        }