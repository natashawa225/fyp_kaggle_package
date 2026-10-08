import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(model, dataloader, device):
    """
    Evaluates model performance across global and per-topic metrics.
    
    Returns dict containing:
    - global_pearson
    - global_spearman
    - per_topic_macro_pearson
    - per_topic_macro_spearman
    - per_topic_weighted_pearson
    - per_topic_weighted_spearman
    - mae
    - rmse
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_topics = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            topics = batch['topic']

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, _ = model(input_ids, attention_mask)

            all_preds.extend(logits.detach().cpu().numpy().flatten().tolist())
            all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())
            all_topics.extend(topics)

    preds = np.array(all_preds, dtype=float)
    labels = np.array(all_labels, dtype=float)
    df_eval = pd.DataFrame({'topic': all_topics, 'label': labels, 'pred': preds})

    # Global metrics
    g_pearson = float(pearsonr(labels, preds)[0]) if len(labels) > 1 else 0.0
    g_spearman = float(spearmanr(labels, preds)[0]) if len(labels) > 1 else 0.0
    mae = float(mean_absolute_error(labels, preds))
    rmse = float(np.sqrt(mean_squared_error(labels, preds)))

    # Per-topic metrics
    topic_pearsons = []
    topic_spearmans = []

    for topic_name, group in df_eval.groupby('topic'):
        t_labels = group['label'].values
        t_preds = group['pred'].values
        n_samples = len(group)

        # Handle < 2 samples or zero variance gracefully
        if n_samples < 2 or np.std(t_labels) == 0 or np.std(t_preds) == 0:
            continue

        try:
            r_p, _ = pearsonr(t_labels, t_preds)
            r_s, _ = spearmanr(t_labels, t_preds)

            if not np.isnan(r_p):
                topic_pearsons.append((r_p, n_samples))
            if not np.isnan(r_s):
                topic_spearmans.append((r_s, n_samples))
        except Exception:
            continue

    if topic_pearsons:
        p_vals, p_weights = zip(*topic_pearsons)
        per_topic_macro_pearson = float(np.mean(p_vals))
        per_topic_weighted_pearson = float(np.average(p_vals, weights=p_weights))
    else:
        per_topic_macro_pearson = 0.0
        per_topic_weighted_pearson = 0.0

    if topic_spearmans:
        s_vals, s_weights = zip(*topic_spearmans)
        per_topic_macro_spearman = float(np.mean(s_vals))
        per_topic_weighted_spearman = float(np.average(s_vals, weights=s_weights))
    else:
        per_topic_macro_spearman = 0.0
        per_topic_weighted_spearman = 0.0

    return {
        "global_pearson": g_pearson,
        "global_spearman": g_spearman,
        "per_topic_macro_pearson": per_topic_macro_pearson,
        "per_topic_macro_spearman": per_topic_macro_spearman,
        "per_topic_weighted_pearson": per_topic_weighted_pearson,
        "per_topic_weighted_spearman": per_topic_weighted_spearman,
        "mae": mae,
        "rmse": rmse
    }

def print_evaluation_summary(metrics_dict, title="EVALUATION SUMMARY"):
    """Prints evaluation metrics in a clean table format."""
    print(f"\n================ {title} ================")
    print(f"Global Pearson:              {metrics_dict['global_pearson']:.4f}")
    print(f"Global Spearman:             {metrics_dict['global_spearman']:.4f}")
    print(f"Per-Topic Macro Pearson:     {metrics_dict['per_topic_macro_pearson']:.4f}")
    print(f"Per-Topic Macro Spearman:    {metrics_dict['per_topic_macro_spearman']:.4f}")
    print(f"Per-Topic Weighted Pearson:  {metrics_dict['per_topic_weighted_pearson']:.4f}")
    print(f"Per-Topic Weighted Spearman: {metrics_dict['per_topic_weighted_spearman']:.4f}")
    print(f"MAE:                         {metrics_dict['mae']:.4f}")
    print(f"RMSE:                        {metrics_dict['rmse']:.4f}")
    print("=" * (18 + len(title)))
