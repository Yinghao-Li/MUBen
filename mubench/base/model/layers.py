"""
Yinghao Li @ Georgia Tech

Contains Linear and Bayesian output Layer
"""

import torch.nn as nn
from ..uncertainty.bnn import BBPOutputLayer


class OutputLayer(nn.Module):

    def __init__(self,
                 last_hidden_dim,
                 n_output_heads,
                 apply_bbp=False,
                 **kwargs):
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

        self._apply_bbp = apply_bbp
        if not apply_bbp:
            self.output_layer = nn.Linear(last_hidden_dim, n_output_heads)
        else:
            self.output_layer = BBPOutputLayer(last_hidden_dim, n_output_heads, **kwargs)

        self.kld = None

    def initialize(self):
        if not self._apply_bbp:
            nn.init.xavier_uniform(self.output_layer.weight)
            self.output_layer.bias.data.fill_(0.01)
        else:
            self.output_layer.initialize()
            self.kld = None
        return self

    def forward(self, x, **kwargs):
        if not self._apply_bbp:
            logits = self.output_layer(x)
        else:
            logits, self.kld = self.output_layer(x)

        return logits
