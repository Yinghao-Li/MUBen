"""
# Author: Yinghao Li
# Created: July 19th, 2023
# Modified: July 19th, 2023
# ---------------------------------------
# Description: Implementation of evidential model layers.
               Modified from https://github.com/aamini/evidential-deep-learning
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d


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

        nu = F.softplus(nu)
        alpha = F.softplus(alpha) + 1.
        beta = F.softplus(beta)

        return torch.cat((gamma, nu, alpha, beta), dim=-1)


class EvidentialRegressionModel(Module):
    def __init__(self, config, return_tensor=False):
        super(EvidentialRegressionModel, self).__init__()
        self.return_tensor = return_tensor

        self.input_layer = nn.Linear(config.d_feature, config.hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(config.n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(config.hidden_dims[i], config.hidden_dims[i+1]))

        self.nig_layer = NIGOutputLayer(d_input=config.hidden_dims[-1])

        self.dropout = nn.Dropout(config.dropout)
        self.activate = nn.ReLU()

    def forward(self, features: torch.Tensor, **kwargs):

        x = self.input_layer(features)

        for hidden_layer in self.hidden_layers:
            x = self.activate(x)
            x = self.dropout(x)
            x = hidden_layer(x)

        x = self.activate(x)
        x = self.dropout(x)

        nig_outs = self.nig_layer(x)

        results = [r.squeeze() for r in nig_outs]
        if not results[0].shape:
            results = [result.unsqueeze(-1) for result in results]

        return torch.stack(results).to(x.device)


class Conv2DNormal(Module):
    def __init__(self, in_channels, out_tasks, kernel_size, **kwargs):
        super(Conv2DNormal, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * out_tasks
        self.n_tasks = out_tasks
        self.conv = Conv2d(self.in_channels, self.out_channels, kernel_size, **kwargs)

    def forward(self, x):
        output = self.conv(x)
        if len(x.shape) == 3:
            mu, logsigma = torch.split(output, self.n_tasks, dim=0)
        else:
            mu, logsigma = torch.split(output, self.n_tasks, dim=1)

        sigma = F.softplus(logsigma) + 1e-6

        return torch.stack([mu, sigma]).to(x.device)


class Conv2DNormalGamma(Module):
    def __init__(self, in_channels, out_tasks, kernel_size, **kwargs):
        super(Conv2DNormalGamma, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_tasks
        self.conv = Conv2d(in_channels, 4 * out_tasks, kernel_size, **kwargs)

    def forward(self, x):
        output = self.conv(x)

        if len(x.shape) == 3:
            gamma, lognu, logalpha, logbeta = torch.split(output, self.out_channels, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(output, self.out_channels, dim=1)

        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1.
        beta = F.softplus(logbeta)
        return torch.stack([gamma, nu, alpha, beta]).to(x.device)
