import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import cached_property
from .layers import utils


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


class GaussianLayer(nn.Module):
    def __init__(self, k=128, edge_types=1024):
        super().__init__()
        self.K = k
        self.means = nn.Embedding(1, k)
        self.stds = nn.Embedding(1, k)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    @cached_property
    def half_log_2pi(self):
        return 0.9189385  # float32 precision

    def gaussian_prob(self, x, mu, sigma):
        """
        Use exp-log-gaussian for numerical stability
        """
        return torch.exp(
            -0.5 * ((x - mu) / sigma) ** 2 - torch.log(sigma) - self.half_log_2pi
        )

    def forward(self, x, edge_type):
        mul = self.mul(edge_type)
        bias = self.bias(edge_type)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.view(-1)
        std = self.stds.weight.view(-1).abs() + 1e-5
        return self.gaussian_prob(x, mean, std)
