"""
Focal loss with adaptive gamma, as proposed in https://arxiv.org/pdf/2006.15607.pdf
"""

import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss

__all__ = ["SigmoidFocalLoss"]


class SigmoidFocalLoss(nn.Module):

    def __init__(self, gamma_threshold=0.25, gamma=2, reduction: str = 'none'):
        """
        Initialize the Focal Loss.
        The parameters are defined in the paper and theoretically do not need to modify

        Parameters
        ----------
        gamma_threshold
        """
        super().__init__()
        self.gamma_threshold = gamma_threshold
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = sigmoid_focal_loss(
            inputs=inputs, targets=targets, alpha=self.gamma_threshold, gamma=self.gamma, reduction=self.reduction
        )
        return loss
