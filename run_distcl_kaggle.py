import sys, os; sys.path.insert(0, os.getcwd())

import sys
import os
import types
import runpy
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.getcwd())

dataset_module = types.ModuleType("src.dataset")

class AQADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        text = f"{row['topic']} [SEP] {row['argument']}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(
                float(row["quality_score"]),
                dtype=torch.float
            )
        }

dataset_module.AQADataset = AQADataset
sys.modules["src.dataset"] = dataset_module

sys.argv = [
    "src.train_distcl",
    "--kernel_type", "gaussian",
    "--sigma", "0.15",
    "--tau", "0.07",
    "--seeds", "42", "123", "456",
    "--epochs", "5",
    "--batch_size", "32",
    "--lr", "2e-5"
]

runpy.run_module("src.train_distcl", run_name="__main__")
