import numpy as np
import pandas as pd
import json
import os
import argparse

def evaluate_pairwise_gap_accuracy(y_true, y_pred, sample_pairs=100000, seed=42):
    """
    Evaluates pairwise ordering accuracy across delta bins |y_i - y_j|:
    Bins: < 0.05, 0.05 - 0.15, 0.15 - 0.25, >= 0.25
    """
    np.random.seed(seed)
    N = len(y_true)
    
    # Sample random pairs of indices (i, j)
    idx1 = np.random.randint(0, N, size=sample_pairs)
    idx2 = np.random.randint(0, N, size=sample_pairs)
    
    # Filter out identical pairs
    valid_mask = idx1 != idx2
    idx1, idx2 = idx1[valid_mask], idx2[valid_mask]

    true_diffs = np.abs(y_true[idx1] - y_true[idx2])
    true_order = y_true[idx1] > y_true[idx2]
    pred_order = y_pred[idx1] > y_pred[idx2]

    # Ignore exact ties in true scores
    non_tie_mask = y_true[idx1] != y_true[idx2]
    idx1, idx2 = idx1[non_tie_mask], idx2[non_tie_mask]
    true_diffs = true_diffs[non_tie_mask]
    correct_matches = (true_order[non_tie_mask] == pred_order[non_tie_mask])

    bins = {
        "< 0.05": (true_diffs < 0.05),
        "0.05 - 0.15": (true_diffs >= 0.05) & (true_diffs < 0.15),
        "0.15 - 0.25": (true_diffs >= 0.15) & (true_diffs < 0.25),
        ">= 0.25": (true_diffs >= 0.25)
    }

    results = {}
    for bin_name, mask in bins.items():
        if mask.sum() > 0:
            acc = float(correct_matches[mask].mean())
            count = int(mask.sum())
        else:
            acc = 0.0
            count = 0
        results[bin_name] = {
            "accuracy": acc,
            "count": count
        }

    overall_acc = float(correct_matches.mean())
    results["overall_pairwise_accuracy"] = overall_acc
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_file", type=str, required=True, help="Path to JSON file with true and predicted labels")
    args = parser.parse_args()

    with open(args.preds_file, "r") as f:
        data = json.load(f)

    y_true = np.array(data["y_true"])
    y_pred = np.array(data["y_pred"])

    res = evaluate_pairwise_gap_accuracy(y_true, y_pred)
    print("\n================ Pairwise Quality-Gap Accuracy (RQ2 Analysis) ================")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"Bin {k:12s} -> Accuracy: {v['accuracy']*100:.2f}% (Samples: {v['count']})")
        else:
            print(f"Overall Pairwise Accuracy: {v*100:.2f}%")

if __name__ == "__main__":
    main()
