import json
import torch
from torch.utils.data import Dataset
from model import tokenize_example, ROLES, MAX_LENGTH

class RoleDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length = MAX_LENGTH):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {role: idx for idx, role in enumerate(ROLES)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoded = tokenize_example(self.tokenizer, ex["inn"], ex["excerpt"], self.max_length)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label2id[ex["role"]], dtype=torch.long),
        }