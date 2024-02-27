# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F


class LayerNorm(torch.nn.Module):
    """
    Custom implementation of Layer Normalization.

    This layer applies layer normalization over a mini-batch of inputs.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Scaling parameter learned during training if elementwise_affine is set to True.
    bias : torch.nn.Parameter
        Offset parameter learned during training if elementwise_affine is set to True.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Parameters
        ----------
        normalized_shape : int or tuple
            Input shape from an expected input of size.
        eps : float, optional
            Value added to the denominator for numerical stability. Default is 1e-5.
        elementwise_affine : bool, optional
            Whether to learn elementwise scaling and offset. Default is True.
        """
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

        def torch_layer_norm(input):
            """Internal function to compute layer normalization."""
            return F.layer_norm(
                input,
                self.normalized_shape,
                self.weight.type(input.dtype),
                self.bias.type(input.dtype),
                self.eps,
            )

        self.func = torch_layer_norm

    def reset_parameters(self):
        """
        Reset the parameters (weight and bias) of the layer normalization.

        The weight is initialized with ones and the bias with zeros.
        """
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):
        return self.func(input)

    def extra_repr(self):
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine=True".format(**self.__dict__)
        )
