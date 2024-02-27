"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description:
Implements Temperature Scaling (TS) for model calibration.

# Reference: https://ethen8181.github.io/machine-learning/model_selection/prob_calibration/deeplearning_prob_calibration.html
"""


import torch
import torch.nn as nn

__all__ = ["TSModel"]


# Temperature scaling model
class TSModel(nn.Module):
    """
    Implements Temperature Scaling (TS) for model calibration.

    TS is a post-processing technique that rescales the logits of a pre-trained model
    using a single scalar (the temperature). When this scalar is learned on a validation
    set, it can improve the model's calibration without affecting its accuracy.

    Args:
        model (nn.Module): Pre-trained model for which calibration is sought.
        n_task (int): Number of tasks for which temperature scaling is performed. Each task
            gets assigned its own temperature.

    Attributes:
        model (nn.Module): Underlying base model.
        temperature (nn.Parameter): Learnable temperature parameters for each task.

    Note:
        The initialization value for temperature doesn't seem to be crucial based
        on preliminary experiments.
    """

    def __init__(self, model, n_task):
        super().__init__()
        # the single temperature scaling parameter, the initialization value doesn't
        # seem to matter that much based on some ad-hoc experimentation
        self.model = model
        # assign one temperature for each task
        self.temperature = nn.Parameter(torch.ones(n_task))

    def forward(self, batch):
        """
        Forward method that returns scaled logits.

        The logits from the base model are divided by the temperature, effectively
        applying the temperature scaling.

        Args:
            batch (torch.Tensor): Input data batch.

        Returns:
            torch.Tensor: Scaled logits.
        """

        # Set the base model to evaluation mode for Temperature Scaling training
        self.model.eval()

        logits = self.model(batch)
        scaled_logits = logits / self.temperature

        return scaled_logits
