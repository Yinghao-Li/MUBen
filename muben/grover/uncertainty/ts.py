"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: Reload the Temperature Scaling UQ method
               to accomotate GROVER's 2-head output format.
"""

import torch
import torch.nn as nn


# Temperature scaling model
class TSModel(nn.Module):
    def __init__(self, model, n_task):
        super().__init__()
        # the single temperature scaling parameter, the initialization value doesn't
        # seem to matter that much based on some ad-hoc experimentation
        self.model = model
        self.atom_temperature = nn.Parameter(
            torch.ones(n_task)
        )  # assign one temperature for each task
        self.bond_temperature = nn.Parameter(
            torch.ones(n_task)
        )  # assign one temperature for each task

    def forward(self, batch):
        """forward method that returns softmax-ed confidence scores."""

        # Set the base model to evaluation mode for Temperature Scaling training
        self.model.eval()

        atom_logits, bond_logits = self.model(batch)
        atom_logits_scaled = atom_logits / self.atom_temperature
        bond_logits_scaled = bond_logits / self.bond_temperature

        return atom_logits_scaled, bond_logits_scaled
