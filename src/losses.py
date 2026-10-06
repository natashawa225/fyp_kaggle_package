import torch
import torch.nn as nn
import torch.nn.functional as F

class DistCLLoss(nn.Module):
    """
    Continuous Distance-Aware Soft-Contrastive Loss (DistCL)
    
    Inputs:
        features: torch.Tensor of shape (N, d) - Hidden embeddings (e.g., [CLS] representations)
        labels: torch.Tensor of shape (N,) - Continuous target quality scores in [0, 1]
    
    Hyperparameters:
        temperature (tau): float - Softmax scaling factor
        sigma: float - RBF kernel distance bandwidth parameter
        kernel_type: str - 'gaussian' or 'exponential'
    """
    def __init__(self, temperature=0.07, sigma=0.15, kernel_type="gaussian"):
        super(DistCLLoss, self).__init__()
        self.temperature = temperature
        self.sigma = sigma
        self.kernel_type = kernel_type

    def forward(self, features, labels):
        device = features.device
        N = features.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Normalize features to unit hypersphere S^(d-1)
        z = F.normalize(features, dim=1) # (N, d)

        # Pairwise Cosine Similarity Matrix: S_ij = z_i . z_j / tau
        sim_matrix = torch.matmul(z, z.T) / self.temperature # (N, N)

        # Pairwise Continuous Label Differences: |y_i - y_j|
        labels = labels.view(-1, 1) # (N, 1)
        label_diffs = torch.abs(labels - labels.T) # (N, N)

        # Compute Weight Kernel w_ij
        if self.kernel_type == "gaussian":
            w = torch.exp(-(label_diffs ** 2) / (2 * (self.sigma ** 2)))
        elif self.kernel_type == "exponential":
            w = torch.exp(-label_diffs / self.sigma)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Remove self-attraction (diagonal)
        mask_self = torch.eye(N, dtype=torch.bool, device=device)
        w = w.masked_fill(mask_self, 0.0)

        # Normalize attraction weights per anchor i: \tilde{w}_ij = w_ij / sum_k(w_ik)
        w_sum = w.sum(dim=1, keepdim=True) + 1e-8
        w_norm = w / w_sum # (N, N)

        # Log-softmax over contrastive denominator (all samples k != i)
        # Numerical stability via logsumexp over non-self entries
        sim_matrix_masked = sim_matrix.masked_fill(mask_self, -1e4)
        log_prob = sim_matrix - torch.logsumexp(sim_matrix_masked, dim=1, keepdim=True)

        # Weighted Soft-Contrastive Loss
        loss_i = -(w_norm * log_prob).sum(dim=1)
        loss = loss_i.mean()

        return loss


class QuartileSupConLoss(nn.Module):
    """
    EMNLP 2023 Quartile-Based Supervised Contrastive Loss (Wang et al., 2023)
    Discretizes sorted batch into 4 quartiles B1, B2, B3, B4.
    Uses B2 and B3 as anchors.
    """
    def __init__(self, temperature=0.1):
        super(QuartileSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        N = features.size(0)
        if N < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        z = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Sort batch by true label
        sorted_indices = torch.argsort(labels, descending=True)
        n_q = N // 4
        if n_q == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Quartile indices: B1 (top), B2, B3, B4 (bottom)
        b2_idx = sorted_indices[n_q : 2 * n_q]
        b3_idx = sorted_indices[2 * n_q : 3 * n_q]

        def compute_subset_loss(anchor_indices):
            loss_sum = 0.0
            count = 0
            for i in anchor_indices:
                # Positives are other samples in the same quartile
                pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
                pos_mask[anchor_indices] = True
                pos_mask[i] = False # remove self

                if pos_mask.sum() == 0:
                    continue

                pos_sim = sim_matrix[i, pos_mask]
                all_sim = sim_matrix[i, torch.arange(N, device=device) != i]

                log_prob = torch.logsumexp(pos_sim, dim=0) - torch.logsumexp(all_sim, dim=0)
                loss_sum -= log_prob
                count += 1
            return loss_sum / max(count, 1)

        l_b2 = compute_subset_loss(b2_idx)
        l_b3 = compute_subset_loss(b3_idx)

        return 0.5 * (l_b2 + l_b3)
