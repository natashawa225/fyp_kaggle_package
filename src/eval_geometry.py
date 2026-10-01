import numpy as np
import json
import os
import argparse
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

def analyze_representation_geometry(features, labels, num_pairs=50000, seed=42):
    """
    Evaluates representation geometry (RQ3 Probes):
    1. Spearman correlation between embedding cosine similarity and label quality distance |y_i - y_j|.
    2. Cluster variance for Low (y < 0.3), Medium (0.3 <= y <= 0.7), and High (y > 0.7) quality groups.
    """
    np.random.seed(seed)
    N = features.shape[0]

    # Normalize features to unit sphere
    z = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # 1. Cosine Similarity vs Label Distance Alignment
    idx1 = np.random.randint(0, N, size=num_pairs)
    idx2 = np.random.randint(0, N, size=num_pairs)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    cos_sims = np.sum(z[idx1] * z[idx2], axis=1) # dot product on unit vectors
    label_diffs = np.abs(labels[idx1] - labels[idx2])

    # Expected: negative correlation (higher quality distance -> lower cosine similarity)
    alignment_r, _ = spearmanr(label_diffs, cos_sims)

    # 2. Within-Group Cosine Variance
    low_mask = labels < 0.3
    med_mask = (labels >= 0.3) & (labels <= 0.7)
    high_mask = labels > 0.7

    def compute_group_variance(mask_g):
        if mask_g.sum() < 2:
            return 0.0
        z_g = z[mask_g]
        mean_z = np.mean(z_g, axis=0, keepdims=True)
        mean_z = mean_z / (np.linalg.norm(mean_z) + 1e-8)
        sims = np.dot(z_g, mean_z.T)
        return float(np.mean(1.0 - sims)) # Average cosine distance to group mean centroid

    return {
        "alignment_spearman_rho": float(alignment_r),
        "group_cosine_distances": {
            "low_quality_variance": compute_group_variance(low_mask),
            "med_quality_variance": compute_group_variance(med_mask),
            "high_quality_variance": compute_group_variance(high_mask)
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, required=True, help="Path to .npz file containing 'features' and 'labels'")
    args = parser.parse_args()

    data = np.load(args.features_file)
    features = data["features"]
    labels = data["labels"]

    res = analyze_representation_geometry(features, labels)
    print("\n================ Representation Geometry Analysis (RQ3 Analysis) ================")
    print(f"Alignment Spearman (CosSim vs |Δy|): {res['alignment_spearman_rho']:.4f} (Expected: Negative)")
    print("Group Centroid Cosine Variance:")
    print(f"  Low Quality  (y < 0.3): {res['group_cosine_distances']['low_quality_variance']:.4f}")
    print(f"  Med Quality  (0.3-0.7): {res['group_cosine_distances']['med_quality_variance']:.4f}")
    print(f"  High Quality (y > 0.7): {res['group_cosine_distances']['high_quality_variance']:.4f}")

if __name__ == "__main__":
    main()
