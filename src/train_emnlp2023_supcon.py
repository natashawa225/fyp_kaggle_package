import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.dataset import AQADataset
from src.train_roberta_mse import RoBERTaAQAModel, set_seed, evaluate
from src.losses import QuartileSupConLoss

def train_single_seed_emnlp(seed, model_name, epochs, batch_size, lr, lambda_cl, tau):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n================ Training EMNLP 2023 Quartile SupCon Seed {seed} | τ={tau} | λ={lambda_cl} ================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_loader = DataLoader(AQADataset(train_df, tokenizer), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(AQADataset(val_df, tokenizer), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(AQADataset(test_df, tokenizer), batch_size=batch_size, shuffle=False, pin_memory=True)

    model = RoBERTaAQAModel(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    mse_criterion = nn.MSELoss()
    supcon_criterion = QuartileSupConLoss(temperature=tau)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    best_val_pearson = -1.0
    best_test_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_cl = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits, cls_rep = model(input_ids, attention_mask)
                loss_mse = mse_criterion(logits, labels)
                loss_cl = supcon_criterion(cls_rep, labels)
                loss = loss_mse + lambda_cl * loss_cl

            scaler.scale(loss).backward()
            try:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            except ValueError:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_cl += loss_cl.item()

        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loader):.4f} (MSE: {total_mse/len(train_loader):.4f}, CL: {total_cl/len(train_loader):.4f}) | Val Pearson: {val_metrics['pearson']:.4f}")

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
    parser.add_argument("--lambda_cl", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()

    all_seed_results = []
    for seed in args.seeds:
        res = train_single_seed_emnlp(seed, args.model_name, args.epochs, args.batch_size, args.lr, args.lambda_cl, args.tau)
        all_seed_results.append(res)

    pearson_scores = [r["pearson"] for r in all_seed_results]
    spearman_scores = [r["spearman"] for r in all_seed_results]
    mae_scores = [r["mae"] for r in all_seed_results]

    summary = {
        "tau": args.tau,
        "lambda_cl": args.lambda_cl,
        "seeds": args.seeds,
        "pearson_mean": float(np.mean(pearson_scores)),
        "pearson_std": float(np.std(pearson_scores)),
        "spearman_mean": float(np.mean(spearman_scores)),
        "spearman_std": float(np.std(spearman_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "all_seed_results": all_seed_results
    }

    print(f"\n================ FINAL SUMMARY (EMNLP 2023 Quartile SupCon) ================")
    print(f"Pearson:  {summary['pearson_mean']:.4f} ± {summary['pearson_std']:.4f}")
    print(f"Spearman: {summary['spearman_mean']:.4f} ± {summary['spearman_std']:.4f}")
    print(f"MAE:      {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")

    os.makedirs("results", exist_ok=True)
    out_file = "results/emnlp2023_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
