import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

def set_seed(seed, deterministic=False):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class RoBERTaAQAModel(nn.Module):
    def __init__(self, model_name='roberta-base', use_proj=False, proj_dim=128):
        super(RoBERTaAQAModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        self.use_proj = use_proj
        if self.use_proj:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, proj_dim)
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :] # [CLS] embedding
        logits = self.regressor(cls_rep).squeeze(-1)
        if self.use_proj:
            cl_rep = self.projection_head(cls_rep)
            return logits, cl_rep
        return logits, cls_rep
