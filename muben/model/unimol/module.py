# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import cached_property
from .layers import utils


class NonLinearHead(nn.Module):
    """
    Head for simple classification tasks.

    A two-layer feed-forward neural network with a specified activation function.

    Attributes
    ----------
    linear1 : torch.nn.Linear
        First linear layer.
    linear2 : torch.nn.Linear
        Second linear layer.
    activation_fn : function
        Activation function.
    """

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input tensor.
        out_dim : int
            Desired dimensionality for the output tensor.
        activation_fn : str
            Name of the activation function to use after the first linear layer.
        hidden : int, optional
            Dimensionality of the hidden layer. If not specified, defaults to the input dimension.
        """
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
    """
    Distance head for producing a symmetric matrix from transformer outputs.

    Attributes
    ----------
    dense : torch.nn.Linear
        Linear layer to transform the input tensor.
    layer_norm : torch.nn.LayerNorm
        Layer normalization.
    out_proj : torch.nn.Linear
        Linear layer for output projection.
    activation_fn : function
        Activation function.
    """

    def __init__(
        self,
        heads,
        activation_fn,
    ):
        """
        Parameters
        ----------
        heads : int
            Number of attention heads.
        activation_fn : str
            Name of the activation function to use.
        """
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Forward pass through the DistanceHead.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Symmetric matrix after processing.
        """

        bsz, seq_len, seq_len, _ = x.size()
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


class GaussianLayer(nn.Module):
    """
    Layer to compute gaussian probabilities for input tensors.

    Attributes
    ----------
    K : int
        Dimensionality for gaussian computation.
    means : torch.nn.Embedding
        Embedding for means.
    stds : torch.nn.Embedding
        Embedding for standard deviations.
    mul : torch.nn.Embedding
        Embedding for multiplication factor.
    bias : torch.nn.Embedding
        Embedding for bias.
    half_log_2pi : float
        Cached property for half of the logarithm of 2π.
    """

    def __init__(self, k=128, edge_types=1024):
        """
        Parameters
        ----------
        k : int, optional
            Dimensionality for gaussian computation. Defaults to 128.
        edge_types : int, optional
            Number of edge types. Defaults to 1024.
        """
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
        """Cached property for half of the logarithm of 2π."""
        return 0.9189385  # float32 precision

    def gaussian_prob(self, x, mu, sigma):
        """
        Compute the gaussian probability.
        Use exp-log-gaussian for numerical stability

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        mu : torch.Tensor
            Mean values for gaussian computation.
        sigma : torch.Tensor
            Standard deviation values for gaussian computation.

        Returns
        -------
        torch.Tensor
            Gaussian probabilities for each value in the input tensor.
        """
        return torch.exp(
            -0.5 * ((x - mu) / sigma) ** 2
            - torch.log(sigma)
            - self.half_log_2pi
        )

    def forward(self, x, edge_type):
        """
        Forward pass through the GaussianLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        edge_type : torch.Tensor
            Type of edge for the input tensor.

        Returns
        -------
        torch.Tensor
            Gaussian probabilities for each value in the input tensor.
        """
        mul = self.mul(edge_type)
        bias = self.bias(edge_type)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.view(-1)
        std = self.stds.weight.view(-1).abs() + 1e-5
        return self.gaussian_prob(x, mean, std)
