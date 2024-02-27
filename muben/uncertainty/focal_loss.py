"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description:

Implements the Focal Loss with adaptive gamma.
This loss function is designed to address class imbalance in classification problems.

# Reference: https://arxiv.org/pdf/2006.15607.pdf
"""

import torch.nn as nn
from torchvision.ops.focal_loss import sigmoid_focal_loss

__all__ = ["SigmoidFocalLoss"]


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma_threshold=0.25, gamma=2, reduction: str = "none"):
        """
        Initialize the SigmoidFocalLoss module.

        The parameters are defined based on the paper and theoretically do not need to be modified
        unless there's a specific use-case that requires a different setting.

        Parameters
        ----------
        gamma_threshold : float
            Threshold for class weights (alpha) in the loss function.
        gamma : float
            Modulating factor to emphasize misclassified examples in the loss.
        reduction : str
            Specifies the reduction to apply to the output. It can be "none" (no reduction),
            "mean" (mean of the loss), or "sum" (sum of the loss).
        """
        super().__init__()
        self.gamma_threshold = gamma_threshold
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass through the SigmoidFocalLoss module.

        Parameters
        ----------
        inputs : torch.Tensor
            Predictions from the model.
        targets : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Computed focal loss based on the inputs and targets.
        """
        loss = sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.gamma_threshold,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss
