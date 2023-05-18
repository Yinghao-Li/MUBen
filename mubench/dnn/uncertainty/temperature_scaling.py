"""
Modified from
ethen8181.github.io/machine-learning/model_selection/prob_calibration/deeplearning_prob_calibration.html
"""

import torch
import torch.nn as nn


# Temperature scaling model
class TSModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        # the single temperature scaling parameter, the initialization value doesn't
        # seem to matter that much based on some ad-hoc experimentation
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, batch):
        """forward method that returns softmax-ed confidence scores."""
        self.model.eval()

        logits = self.model(batch)
        scaled_logits = logits / self.temperature

        return scaled_logits

