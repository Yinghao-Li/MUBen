"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: Modified from https://github.com/JavierAntoran/Bayesian-Neural-Networks
"""

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["BBPOutputLayer"]


def kld_cost(mu_p, sig_p, mu_q, sig_q):
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = (
        0.5
        * (
            2 * torch.log(sig_p / sig_q)
            - 1
            + (sig_q / sig_p).pow(2)
            + ((mu_p - mu_q) / sig_p).pow(2)
        ).sum()
    )
    return kld


class BBPOutputLayer(nn.Module):
    """
    Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
        the moments of each weight's normal distribution.
    The KL divergence is obtained in closed form.
    Only works with gaussian priors.
    """

    def __init__(self, last_hidden_dim, n_output_heads, bbp_prior_sigma=0.1, **kwargs):
        super(BBPOutputLayer, self).__init__()
        self.n_in = last_hidden_dim
        self.n_out = n_output_heads
        self.prior_sigma = bbp_prior_sigma

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(self.n_in, self.n_out))
        self.weight_rho = nn.Parameter(torch.empty(self.n_in, self.n_out))

        self.bias_mu = nn.Parameter(torch.empty(self.n_out))
        self.bias_rho = nn.Parameter(torch.empty(self.n_out))

        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.weight_mu, -0.1, 0.1)
        nn.init.uniform_(self.bias_mu, -0.1, 0.1)

        nn.init.uniform_(self.weight_rho, -3, -2)
        nn.init.uniform_(self.bias_rho, -3, -2)

    def forward(self, x):
        # calculate std
        std_weight = 1e-6 + F.softplus(self.weight_rho, beta=1, threshold=20)
        std_bias = 1e-6 + F.softplus(self.bias_rho, beta=1, threshold=20)

        act_weight_mu = torch.mm(x, self.weight_mu)  # self.W_mu + std_w * eps_W
        act_weight_std = torch.sqrt(torch.mm(x.pow(2), std_weight.pow(2)))

        eps_w = torch.empty_like(act_weight_std).normal_(mean=0, std=1)
        eps_b = torch.empty_like(std_bias).normal_(mean=0, std=1)

        act_w_out = act_weight_mu + act_weight_std * eps_w  # (batch_size, n_output)
        act_b_out = self.bias_mu + std_bias * eps_b

        logits = act_w_out + act_b_out.unsqueeze(0).expand(x.shape[0], -1)

        kld = kld_cost(
            mu_p=0, sig_p=self.prior_sigma, mu_q=self.weight_mu, sig_q=std_weight
        ) + kld_cost(mu_p=0, sig_p=0.1, mu_q=self.bias_mu, sig_q=std_bias)

        return logits, kld
