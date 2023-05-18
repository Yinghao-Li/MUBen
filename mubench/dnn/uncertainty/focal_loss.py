"""
Focal loss with adaptive gamma, as proposed in https://arxiv.org/pdf/2006.15607.pdf
Modified from https://github.com/torrvision/focal_calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma_threshold=0.2, gamma_lower=5, gamma_higher=3):
        """
        Initialize the Focal Loss.
        The parameters are defined in the paper and theoretically do not need to modify

        Parameters
        ----------
        gamma_threshold
        gamma_lower
        gamma_higher
        """
        super().__init__()
        self.gamma_threshold = gamma_threshold
        self.gamma_lower = gamma_lower
        self.gamma_higher = gamma_higher

    def get_gamma(self, pt: torch.Tensor):
        gamma = torch.zeros_like(pt)
        gamma[pt <= self.gamma_threshold] = self.gamma_lower
        gamma[pt > self.gamma_threshold] = self.gamma_higher
        return gamma

    def forward(self, logits, lbs):
        log_pt = F.log_softmax(logits, dim=-1).gather(dim=1, index=lbs.unsqueeze(-1)).view(-1)
        pt = log_pt.exp()
        gamma = self.get_gamma(pt)
        loss = -1 * (1 - pt) ** gamma * log_pt
        return loss
