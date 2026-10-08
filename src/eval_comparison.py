import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset import AQADataset
from src.model import RoBERTaAQAModel, set_seed
from src.metrics import evaluate

def get_bins():
    return {
        "Bin A (<0.05)": (0.0, 0.05),
        "Bin B (0.05-0.15)": (0.05, 0.15),
        "Bin C (0.15-0.25)": (0.15, 0.25),
        "Bin D (0.25-0.50)": (0.25, 0.50),
        "Bin E (>=0.50)": (0.50, float("inf"))
    }

def construct_and_sample_pairs(test_df, seed=42, max_pairs_per_bin=2000):
    """
    Constructs all within-topic argument pairs, bins them by true score difference |y_i - y_j|,
    and samples up to max_pairs_per_bin pairs per bin using a fixed seed.
    """
    rng = np.random.RandomState(seed)
    bins = get_bins()
    binned_pairs = {b_name: [] for b_name in bins}

    # Group by topic
    for topic_name, group in test_df.groupby('topic'):
        indices = group.index.values
        scores = group['quality_score'].values
        n = len(indices)

        for i in range(n):
            for j in range(i + 1, n):
                idx1, idx2 = indices[i], indices[j]
                y1, y2 = scores[i], scores[j]
                diff = abs(y1 - y2)

                # Determine bin
                for b_name, (low, high) in bins.items():
                    if low <= diff < high:
                        binned_pairs[b_name].append((idx1, idx2, y1, y2, diff))
                        break

    sampled_binned_pairs = {}
    for b_name, pair_list in binned_pairs.items():
        if len(pair_list) > max_pairs_per_bin:
            sample_idx = rng.choice(len(pair_list), size=max_pairs_per_bin, replace=False)
            sampled_binned_pairs[b_name] = [pair_list[k] for k in sample_idx]
        else:
            sampled_binned_pairs[b_name] = pair_list

    return sampled_binned_pairs

def predict_test_scores(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, _ = model(input_ids, attention_mask)
            all_preds.extend(logits.detach().cpu().numpy().flatten().tolist())
    return np.array(all_preds, dtype=float)

def evaluate_pairwise_comparison(pred_scores, sampled_binned_pairs):
    """
    Evaluates pairwise comparison accuracy per bin and overall.
    """
    results = {}
    total_correct = 0
    total_valid_pairs = 0

    for b_name, pairs in sampled_binned_pairs.items():
        correct = 0
        valid = 0
        for idx1, idx2, y1, y2, diff in pairs:
            if y1 == y2:
                continue
            valid += 1

            true_winner = 1 if y1 > y2 else 2
            pred1, pred2 = pred_scores[idx1], pred_scores[idx2]
            pred_winner = 1 if pred1 > pred2 else 2

            if pred_winner == true_winner:
                correct += 1

        acc = (correct / valid) if valid > 0 else 0.5
        results[b_name] = {
            "accuracy": acc,
            "correct": correct,
            "total_pairs": valid
        }
        total_correct += correct
        total_valid_pairs += valid

    overall_acc = (total_correct / total_valid_pairs) if total_valid_pairs > 0 else 0.5
    results["overall"] = {
        "accuracy": overall_acc,
        "correct": total_correct,
        "total_pairs": total_valid_pairs
    }
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file or directory containing .pt checkpoints")
    parser.add_argument("--test_path", type=str, default="data/processed/test.csv")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pairs_per_bin", type=int, default=2000)
    args = parser.parse_args()

    # Find checkpoints
    if os.path.isfile(args.checkpoint):
        checkpoint_paths = [args.checkpoint]
    elif os.path.isdir(args.checkpoint):
        checkpoint_paths = sorted(glob.glob(os.path.join(args.checkpoint, "*.pt")))
        if not checkpoint_paths:
            # Check subdirectories
            checkpoint_paths = sorted(glob.glob(os.path.join(args.checkpoint, "**", "*.pt"), recursive=True))
    else:
        checkpoint_paths = sorted(glob.glob(args.checkpoint))

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files found at: {args.checkpoint}")

    print(f"Found {len(checkpoint_paths)} checkpoint file(s) for comparison evaluation.")

    test_df = pd.read_csv(args.test_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_loader = DataLoader(AQADataset(test_df, tokenizer), batch_size=args.batch_size, shuffle=False)

    # Construct & sample pairs
    sampled_binned_pairs = construct_and_sample_pairs(test_df, seed=args.seed, max_pairs_per_bin=args.max_pairs_per_bin)

    bin_names = list(get_bins().keys())
    
    # Evaluate across all checkpoints
    ckpt_results = []
    for ckpt_path in checkpoint_paths:
        print(f"Evaluating checkpoint: {ckpt_path}")
        model = RoBERTaAQAModel(args.model_name).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        pred_scores = predict_test_scores(model, test_loader, device)
        res = evaluate_pairwise_comparison(pred_scores, sampled_binned_pairs)
        ckpt_results.append((ckpt_path, res))

    # Calculate Random & Oracle baselines
    random_baseline = {b_name: 0.5 for b_name in bin_names}
    random_baseline["overall"] = 0.5

    oracle_baseline = {b_name: 1.0 for b_name in bin_names}
    oracle_baseline["overall"] = 1.0

    # Aggregate stats across checkpoints
    aggregated_results = {}
    for b_key in bin_names + ["overall"]:
        accs = [res[b_key]["accuracy"] for _, res in ckpt_results]
        pair_counts = ckpt_results[0][1][b_key]["total_pairs"]
        aggregated_results[b_key] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "total_pairs": pair_counts
        }

    # Format output JSON
    output_dict = {
        "checkpoint_arg": args.checkpoint,
        "num_checkpoints": len(checkpoint_paths),
        "checkpoints_evaluated": [p for p, _ in ckpt_results],
        "random_baseline": random_baseline,
        "oracle_baseline": oracle_baseline,
        "aggregated_results": aggregated_results,
        "per_checkpoint_results": [
            {
                "checkpoint": p,
                "overall_accuracy": res["overall"]["accuracy"],
                "bin_accuracies": {b_name: res[b_name]["accuracy"] for b_name in bin_names}
            }
            for p, res in ckpt_results
        ]
    }

    # Print clean table
    print("\n================ ARGUMENT QUALITY COMPARISON TASK RESULTS ================")
    print(f"{'Model / Baseline':<28} | " + " | ".join([f"{b:<18}" for b in bin_names]) + f" | {'Overall Acc':<12}")
    print("-" * 145)
    
    # Print Random
    rand_row = f"{'Random Baseline':<28} | " + " | ".join([f"{0.5000:<18.4f}" for _ in bin_names]) + f" | {0.5000:<12.4f}"
    print(rand_row)

    # Print Oracle
    oracle_row = f"{'Oracle Baseline':<28} | " + " | ".join([f"{1.0000:<18.4f}" for _ in bin_names]) + f" | {1.0000:<12.4f}"
    print(oracle_row)

    # Print Evaluated Model
    model_b_strs = [
        f"{aggregated_results[b]['mean_accuracy']:.4f}±{aggregated_results[b]['std_accuracy']:.4f}"
        for b in bin_names
    ]
    ov_str = f"{aggregated_results['overall']['mean_accuracy']:.4f}±{aggregated_results['overall']['std_accuracy']:.4f}"
    model_row = f"{os.path.basename(args.checkpoint):<28} | " + " | ".join([f"{s:<18}" for s in model_b_strs]) + f" | {ov_str:<12}"
    print(model_row)
    print("=" * 145)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"Saved comparison results to {args.output}")

if __name__ == "__main__":
    main()
