import json
import csv
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

RAW_DIR = "data/raw/ibm_rank_30k"
PROCESSED_DIR = "data/processed"

class AQADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.topics = self.df['topic'].astype(str).tolist()
        self.arguments = self.df['argument'].astype(str).tolist()
        self.scores = self.df['quality_score'].astype(float).values

        self.encodings = tokenizer(
            self.topics,
            self.arguments,
            truncation=True,
            max_length=max_len,
            padding='max_length',
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.scores[idx], dtype=torch.float),
            'topic': self.topics[idx]
        }

def process_split(json_path, split_name):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = []
    items = data.values() if isinstance(data, dict) else data
    for item in items:
        arg_text = item.get("argument", "").strip()
        topic_text = item.get("topic", "").strip()
        label_val = float(item.get("label", 0.0))
        item_id = item.get("id_string", f"{split_name}_{len(records)}")
        
        records.append({
            "id": item_id,
            "topic": topic_text,
            "argument": arg_text,
            "quality_score": label_val,
            "split": split_name
        })
    
    df = pd.DataFrame(records)
    out_path = os.path.join(PROCESSED_DIR, f"{split_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {split_name} split: {len(df)} samples to {out_path}")
    return df

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df = process_split(os.path.join(RAW_DIR, "train.json"), "train")
    val_df = process_split(os.path.join(RAW_DIR, "val.json"), "val")
    test_df = process_split(os.path.join(RAW_DIR, "test.json"), "test")

    print("\n--- Summary of Processed Splits ---")
    print(f"Train: {len(train_df)} rows, {train_df['topic'].nunique()} topics")
    print(f"Val:   {len(val_df)} rows, {val_df['topic'].nunique()} topics")
    print(f"Test:  {len(test_df)} rows, {test_df['topic'].nunique()} topics")

if __name__ == "__main__":
    main()
