import json
import csv
import os
from collections import Counter, defaultdict

RAW_DIR = "data/raw/ibm_rank_30k"

def inspect_json(file_path):
    print(f"--- Inspecting {file_path} ---")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total entries: {len(data)}")
    if len(data) > 0:
        sample_key = list(data.keys())[0] if isinstance(data, dict) else 0
        sample = data[sample_key] if isinstance(data, dict) else data[0]
        print(f"Sample structure: {sample}")
    return data

def inspect_csv(file_path):
    print(f"\n--- Inspecting {file_path} ---")
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Total CSV rows: {len(rows)}")
    if len(rows) > 0:
        print(f"Columns: {list(rows[0].keys())}")
        print(f"Sample row: {rows[0]}")
    
    # Topic breakdown
    topics = Counter(r.get("topic", r.get("Topic", "")) for r in rows)
    print(f"Total Unique Topics: {len(topics)}")

    # Score stats
    scores = []
    for r in rows:
        val = r.get("score", r.get("score_weighted", r.get("label", None)))
        if val is not None and val != "":
            try:
                scores.append(float(val))
            except ValueError:
                pass
    if scores:
        print(f"Score Count: {len(scores)}")
        print(f"Min Score: {min(scores):.4f}, Max Score: {max(scores):.4f}")
        print(f"Mean Score: {sum(scores)/len(scores):.4f}")
    return rows, topics

def main():
    train_data = inspect_json(os.path.join(RAW_DIR, "train.json"))
    val_data = inspect_json(os.path.join(RAW_DIR, "val.json"))
    test_data = inspect_json(os.path.join(RAW_DIR, "test.json"))

    if os.path.exists(os.path.join(RAW_DIR, "arg_30k.csv")):
        rows, topics = inspect_csv(os.path.join(RAW_DIR, "arg_30k.csv"))

    # Analyze topic splits in json files
    train_topics = set(item.get("topic") for item in (train_data.values() if isinstance(train_data, dict) else train_data))
    val_topics = set(item.get("topic") for item in (val_data.values() if isinstance(val_data, dict) else val_data))
    test_topics = set(item.get("topic") for item in (test_data.values() if isinstance(test_data, dict) else test_data))

    print("\n--- Topic Split Integrity Check ---")
    print(f"Train topics count: {len(train_topics)}")
    print(f"Val topics count: {len(val_topics)}")
    print(f"Test topics count: {len(test_topics)}")
    print(f"Train & Val Overlap: {len(train_topics.intersection(val_topics))}")
    print(f"Train & Test Overlap: {len(train_topics.intersection(test_topics))}")
    print(f"Val & Test Overlap: {len(val_topics.intersection(test_topics))}")

if __name__ == "__main__":
    main()
