import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error

# HuggingFace Transformers
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

class AQADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        topic = str(row['topic'])
        argument = str(row['argument'])
        score = float(row['quality_score'])

        # Input format: [CLS] Topic [SEP] Argument [SEP]
        encoding = self.tokenizer(
            topic,
            argument,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(score, dtype=torch.float)
        }

class RoBERTaAQAModel(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super(RoBERTaAQAModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :] # [CLS] embedding
        logits = self.regressor(cls_rep).squeeze(-1)
        return logits, cls_rep

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits, _ = model(input_ids, attention_mask)
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    pearson_r, _ = pearsonr(labels, preds)
    spearman_r, _ = spearmanr(labels, preds)
    kendall_t, _ = kendalltau(labels, preds)
    mae = mean_absolute_error(labels, preds)

    return {
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "kendall_tau": float(kendall_t),
        "mae": float(mae)
    }

def train_single_seed(seed, model_name, epochs, batch_size, lr):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n================ Training Seed {seed} on {device} ================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_loader = DataLoader(AQADataset(train_df, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AQADataset(val_df, tokenizer), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(AQADataset(test_df, tokenizer), batch_size=batch_size, shuffle=False)

    model = RoBERTaAQAModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    best_val_pearson = -1.0
    best_test_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loader):.4f} | Val Pearson: {val_metrics['pearson']:.4f}, Spearman: {val_metrics['spearman']:.4f}")

        if val_metrics['pearson'] > best_val_pearson:
            best_val_pearson = val_metrics['pearson']
            best_test_metrics = evaluate(model, test_loader, device)
            print(f"--> Best Val Model Updated! Test Pearson: {best_test_metrics['pearson']:.4f}, Spearman: {best_test_metrics['spearman']:.4f}")

    return best_test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    all_seed_results = []
    for seed in args.seeds:
        res = train_single_seed(seed, args.model_name, args.epochs, args.batch_size, args.lr)
        all_seed_results.append(res)

    pearson_scores = [r["pearson"] for r in all_seed_results]
    spearman_scores = [r["spearman"] for r in all_seed_results]
    mae_scores = [r["mae"] for r in all_seed_results]

    summary = {
        "seeds": args.seeds,
        "pearson_mean": float(np.mean(pearson_scores)),
        "pearson_std": float(np.std(pearson_scores)),
        "spearman_mean": float(np.mean(spearman_scores)),
        "spearman_std": float(np.std(spearman_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "all_seed_results": all_seed_results
    }

    print("\n================ FINAL SUMMARY (RoBERTa MSE) ================")
    print(f"Pearson:  {summary['pearson_mean']:.4f} ± {summary['pearson_std']:.4f}")
    print(f"Spearman: {summary['spearman_mean']:.4f} ± {summary['spearman_std']:.4f}")
    print(f"MAE:      {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/roberta_mse_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
