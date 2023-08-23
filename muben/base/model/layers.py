"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: The versatile output layer for all backbone models.
"""


import torch.nn as nn
from ..uncertainty.bbp import BBPOutputLayer
from ..uncertainty.evidential import NIGOutputLayer
from muben.utils.macro import UncertaintyMethods


class OutputLayer(nn.Module):
    def __init__(
        self,
        last_hidden_dim,
        n_output_heads,
        uncertainty_method=UncertaintyMethods.none,
        task_type="classification",
        **kwargs
    ):
        """
        Initialize the model output layer

        Parameters
        ----------
        last_hidden_dim: the dimensionality of the last hidden state
        n_output_heads: the number of output heads
        apply_bbp: whether using Bayesian by Backprop for the output layer
        kwargs: other keyword arguments
        """

        super().__init__()

        self._uncertainty_method = uncertainty_method
        self._task_type = task_type
        if uncertainty_method == UncertaintyMethods.bbp:
            self.output_layer = BBPOutputLayer(
                last_hidden_dim, n_output_heads, **kwargs
            )
        elif (
            uncertainty_method == UncertaintyMethods.evidential
            and task_type == "regression"
        ):
            self.output_layer = NIGOutputLayer(
                last_hidden_dim, n_output_heads, **kwargs
            )
        else:
            self.output_layer = nn.Linear(last_hidden_dim, n_output_heads)

        self.kld = None
        self.initialize()

    def initialize(self):
        if self._uncertainty_method == UncertaintyMethods.bbp:
            self.output_layer.initialize()
            self.kld = None
        elif (
            self._uncertainty_method == UncertaintyMethods.evidential
            and self._task_type == "regression"
        ):
            self.output_layer.initialize()
        else:
            nn.init.xavier_uniform_(self.output_layer.weight)
            self.output_layer.bias.data.fill_(0.01)
        return self

    def forward(self, x, **kwargs):
        if self._uncertainty_method == UncertaintyMethods.bbp:
            logits, self.kld = self.output_layer(x)
        else:
            logits = self.output_layer(x)

        return logits
