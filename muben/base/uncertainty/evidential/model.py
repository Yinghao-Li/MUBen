"""
# Author: Yinghao Li
# Modified: August 4th, 2023
# ---------------------------------------
# Description: Implementation of evidential model layers.
               Modified from https://github.com/aamini/evidential-deep-learning
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module


class NIGOutputLayer(Module):
    def __init__(self, d_input, d_output, **kwargs):
        super(NIGOutputLayer, self).__init__()
        self.n_tasks = d_output
        self.linear = nn.Linear(d_input, d_output)

    def initialize(self):
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear(x)
        x = x.view(batch_size, -1, 4)
        gamma, nu, alpha, beta = torch.split(x, 1, dim=-1)

        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + 1. + 1e-6
        beta = F.softplus(beta) + 1e-6

        return torch.cat((gamma, nu, alpha, beta), dim=-1)
