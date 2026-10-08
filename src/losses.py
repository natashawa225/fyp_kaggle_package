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
        kernel_type: str - 'gaussian', 'exponential', or 'linear'
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
        elif self.kernel_type == "linear":
            w = torch.clamp(1.0 - label_diffs, min=0.0)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Remove self-attraction (diagonal)
        mask_self = torch.eye(N, dtype=torch.bool, device=device)
        w = w.masked_fill(mask_self, 0.0)

        # Normalize attraction weights per anchor i: \tilde{w}_ij = w_ij / sum_k(w_ik)
        w_sum = w.sum(dim=1, keepdim=True) + 1e-8
        w_norm = w / w_sum # (N, N)

        # Log-softmax over contrastive denominator (all samples k != i)
        sim_matrix_masked = sim_matrix.masked_fill(mask_self, -1e4)
        log_prob = sim_matrix - torch.logsumexp(sim_matrix_masked, dim=1, keepdim=True)

        # Weighted Soft-Contrastive Loss
        loss_i = -(w_norm * log_prob).sum(dim=1)
        loss = loss_i.mean()

        return loss


class QuartileSupConLoss(nn.Module):
    """
    EMNLP 2023 Quartile-Based Supervised Contrastive Loss (Wang et al., 2023)
    Discretizes sorted batch of size N into 4 quartiles B1, B2, B3, B4.
    Uses B2 and B3 as anchors.
    Positives: same quartile.
    Negatives: all arguments outside anchor's quartile.
    
    Loss = alpha * L_cl^B2 + (1 - alpha) * L_cl^B3
    """
    def __init__(self, temperature=0.1, alpha=0.5):
        super(QuartileSupConLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, features, labels):
        device = features.device
        N = features.size(0)
        if N < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 1. Normalize features to unit hypersphere
        z = F.normalize(features, dim=1)

        # 2. Pairwise Cosine Similarity / tau
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # 3. Sort batch by true label descending
        sorted_indices = torch.argsort(labels, descending=True)
        n_q = N // 4
        if n_q == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Quartile indices: B1 (top), B2, B3, B4 (bottom)
        b2_idx = sorted_indices[n_q : 2 * n_q]
        b3_idx = sorted_indices[2 * n_q : 3 * n_q]

        mask_self = torch.eye(N, dtype=torch.bool, device=device)

        def compute_quartile_loss(anchor_indices):
            anchor_losses = []
            for i in anchor_indices:
                # Positives: other samples in the same quartile
                pos_mask = torch.zeros(N, dtype=torch.bool, device=device)
                pos_mask[anchor_indices] = True
                pos_mask[i] = False # remove self

                num_pos = pos_mask.sum().item()
                if num_pos == 0:
                    continue

                # Denominator: logsumexp over all k != i
                all_sim_i = sim_matrix[i].masked_fill(mask_self[i], -1e4)
                log_denom = torch.logsumexp(all_sim_i, dim=0)

                # Positives numerator: S_ip - log_denom for each positive p
                pos_sim_i = sim_matrix[i, pos_mask]
                log_prob_p = pos_sim_i - log_denom

                # Average over positive pairs for anchor i
                loss_i = -log_prob_p.mean()
                anchor_losses.append(loss_i)

            if not anchor_losses:
                return torch.tensor(0.0, device=device, requires_grad=True)

            return torch.stack(anchor_losses).mean()

        l_b2 = compute_quartile_loss(b2_idx)
        l_b3 = compute_quartile_loss(b3_idx)

        loss_cl = self.alpha * l_b2 + (1.0 - self.alpha) * l_b3
        return loss_cl
