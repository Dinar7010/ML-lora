import json
import random
from collections import Counter

ROLES = ["истец", "ответчик", "третье лицо", "иное"]
HEAD_CHARS = 3000
TAIL_CHARS = 6000

def build_excerpt(text,head=HEAD_CHARS,tail=TAIL_CHARS):
    if len(text)<=head+tail:
        return text
    head = text[:head]
    tail = text[-tail:]
    return f"{head}\n\n...\n\n{tail}"

def load_json(path):
    doc=[]
    with open(path,"r",encoding="utf-8") as f:
        for line_n,line in enumerate(f,start=1):
            line=line.strip()
            if not line:
                continue
            try:
                doc.append(json.loads(line))
            except json.decoder.JSONDecodeError as e:
                print(f"Ошибка JSON в строке {line_n}: {e}")
    return doc

def build_examples(doc):
    examples=[]
    skipped_no_inn_in_text = 0
    skipped_bad_role = 0

    for doc_index,doc in enumerate(doc,start=1):
        resolution=doc.get("resolution","")
        participants=doc.get("participants",[])
        excerpt=build_excerpt(resolution)

        for p in participants:
            inn=p.get("inn","").strip()
            role=p.get("role","").strip()
            if not inn or role not in ROLES:
                skipped_bad_role+=1
                continue
            if inn not in excerpt:
                skipped_no_inn_in_text+=1
                continue
            examples.append({
                "doc_id":doc_index,
                "inn":inn,
                "role":role,
                "excerpt":excerpt,
            })
    return examples


def split_by_document(examples, val_ratio=0.15, test_ratio=0.15, seed=42):
    doc_ids = sorted({ex["doc_id"] for ex in examples})
    random.Random(seed).shuffle(doc_ids)

    n = len(doc_ids)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_ids = set(doc_ids[:n_test])
    val_ids = set(doc_ids[n_test:n_test + n_val])
    train_ids = set(doc_ids[n_test + n_val:])

    train = [ex for ex in examples if ex["doc_id"] in train_ids]
    val = [ex for ex in examples if ex["doc_id"] in val_ids]
    test = [ex for ex in examples if ex["doc_id"] in test_ids]

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        roles = [ex["role"] for ex in subset]
        print(name, Counter(roles))
    return train, val, test


def save_jsonl(examples: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ANNOTATIONS_PATH = "annotations.json"

    documents = load_json(ANNOTATIONS_PATH)
    print(f"Загружено документов: {len(documents)}")

    examples = build_examples(documents)

    train, val, test = split_by_document(examples)
    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")

    save_jsonl(train, "train.jsonl")
    save_jsonl(val, "val.jsonl")
    save_jsonl(test, "test.jsonl")