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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.dataset import AQADataset
from src.model import RoBERTaAQAModel, set_seed
from src.losses import QuartileSupConLoss
from src.metrics import evaluate, print_evaluation_summary

def train_single_seed_quartile_cl(seed, model_name, epochs, batch_size, lr, alpha, beta, tau, use_proj, checkpoint_dir, deterministic=False):
    set_seed(seed, deterministic=deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n================ Training Quartile CL (Wang et al. 2023) | Seed {seed} | α={alpha}, β={beta}, τ={tau} ================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_loader = DataLoader(AQADataset(train_df, tokenizer), batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(AQADataset(val_df, tokenizer), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(AQADataset(test_df, tokenizer), batch_size=batch_size, shuffle=False, pin_memory=True)

    model = RoBERTaAQAModel(model_name, use_proj=use_proj).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    mse_criterion = nn.MSELoss()
    quartile_cl_criterion = QuartileSupConLoss(temperature=tau, alpha=alpha)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"model_seed{seed}.pt")

    best_val_pearson = -1.0
    best_val_metrics = None

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
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, cl_rep = model(input_ids, attention_mask)
                loss_mse = mse_criterion(logits, labels)
                loss_cl = quartile_cl_criterion(cl_rep, labels)
                loss = beta * loss_cl + (1.0 - beta) * loss_mse

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
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loader):.4f} (MSE: {total_mse/len(train_loader):.4f}, CL: {total_cl/len(train_loader):.4f}) | Val Pearson: {val_metrics['global_pearson']:.4f}")

        if val_metrics['global_pearson'] > best_val_pearson:
            best_val_pearson = val_metrics['global_pearson']
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Saved Best Checkpoint (Val Pearson: {best_val_pearson:.4f}) to {ckpt_path}")

    # Load best checkpoint for test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print_evaluation_summary(test_metrics, title=f"TEST METRICS (Quartile CL Seed {seed})")

    return {
        "seed": seed,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1011])
    parser.add_argument("--output", type=str, default="results/quartile_cl.json")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints/quartile_cl")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    all_seed_results = []
    for seed in args.seeds:
        res = train_single_seed_quartile_cl(
            seed, args.model_name, args.epochs, args.batch_size, args.lr,
            args.alpha, args.beta, args.tau, args.use_proj,
            args.checkpoint_dir, args.deterministic
        )
        all_seed_results.append(res)

    def extract_metric(res_list, metric_key):
        return [r["test_metrics"][metric_key] for r in res_list]

    summary = {
        "model_variant": "Quartile_CL",
        "alpha": args.alpha,
        "beta": args.beta,
        "tau": args.tau,
        "use_proj": args.use_proj,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seeds": args.seeds,
        "global_pearson_mean": float(np.mean(extract_metric(all_seed_results, "global_pearson"))),
        "global_pearson_std": float(np.std(extract_metric(all_seed_results, "global_pearson"))),
        "global_spearman_mean": float(np.mean(extract_metric(all_seed_results, "global_spearman"))),
        "global_spearman_std": float(np.std(extract_metric(all_seed_results, "global_spearman"))),
        "per_topic_macro_pearson_mean": float(np.mean(extract_metric(all_seed_results, "per_topic_macro_pearson"))),
        "per_topic_macro_pearson_std": float(np.std(extract_metric(all_seed_results, "per_topic_macro_pearson"))),
        "per_topic_macro_spearman_mean": float(np.mean(extract_metric(all_seed_results, "per_topic_macro_spearman"))),
        "per_topic_macro_spearman_std": float(np.std(extract_metric(all_seed_results, "per_topic_macro_spearman"))),
        "per_topic_weighted_pearson_mean": float(np.mean(extract_metric(all_seed_results, "per_topic_weighted_pearson"))),
        "per_topic_weighted_pearson_std": float(np.std(extract_metric(all_seed_results, "per_topic_weighted_pearson"))),
        "per_topic_weighted_spearman_mean": float(np.mean(extract_metric(all_seed_results, "per_topic_weighted_spearman"))),
        "per_topic_weighted_spearman_std": float(np.std(extract_metric(all_seed_results, "per_topic_weighted_spearman"))),
        "mae_mean": float(np.mean(extract_metric(all_seed_results, "mae"))),
        "mae_std": float(np.std(extract_metric(all_seed_results, "mae"))),
        "rmse_mean": float(np.mean(extract_metric(all_seed_results, "rmse"))),
        "rmse_std": float(np.std(extract_metric(all_seed_results, "rmse"))),
        "all_seed_results": all_seed_results
    }

    print("\n================ FINAL SUMMARY (Quartile CL) ================")
    print(f"Global Pearson:              {summary['global_pearson_mean']:.4f} ± {summary['global_pearson_std']:.4f}")
    print(f"Global Spearman:             {summary['global_spearman_mean']:.4f} ± {summary['global_spearman_std']:.4f}")
    print(f"Per-Topic Macro Pearson:     {summary['per_topic_macro_pearson_mean']:.4f} ± {summary['per_topic_macro_pearson_std']:.4f}")
    print(f"Per-Topic Macro Spearman:    {summary['per_topic_macro_spearman_mean']:.4f} ± {summary['per_topic_macro_spearman_std']:.4f}")
    print(f"Per-Topic Weighted Pearson:  {summary['per_topic_weighted_pearson_mean']:.4f} ± {summary['per_topic_weighted_pearson_std']:.4f}")
    print(f"Per-Topic Weighted Spearman: {summary['per_topic_weighted_spearman_mean']:.4f} ± {summary['per_topic_weighted_spearman_std']:.4f}")
    print(f"MAE:                         {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"RMSE:                        {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
