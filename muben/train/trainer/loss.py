"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: 

Gaussian Negative Log Likelihood (NLL) Loss with Extended Input

This module provides an extended version of PyTorch's GaussianNLLLoss function
to accept a logit tensor with shape (..., 2) as input. The first channel of this tensor
is treated as the mean while the second channel, after undergoing a softplus operation,
is treated as the variance.
"""


from torch import Tensor
from torch.nn import GaussianNLLLoss as GaussianNLLLossBase
from torch.nn import functional as F

__all__ = ["GaussianNLLLoss"]


# noinspection PyShadowingBuiltins
class GaussianNLLLoss(GaussianNLLLossBase):
    """
    Gaussian Negative Log Likelihood Loss Function with Extended Input.

    Inherits the base GaussianNLLLoss function from PyTorch, but is modified
    to work with an input tensor of shape (..., 2), where the first channel
    represents mean and the second channel (after softplus) represents variance.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the GaussianNLLLoss with the arguments of the base class.
        """
        super().__init__(*args, **kwargs)

    def forward(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        """
        Computes the Gaussian Negative Log Likelihood Loss.

        Parameters
        ----------
        input : Tensor
            Tensor of shape (..., 2), where the first channel represents
            the mean and the second represents the variance (after softplus).
        target : Tensor
            Ground truth tensor.

        Returns
        -------
        Tensor
            The computed loss value.
        """

        # Extract mean and variance from the input tensor
        mean = input[..., 0]
        var = (
            F.softplus(input[..., 1]) + 1e-6
        )  # Ensure variance is non-negative

        # Use the base class's forward method to compute the loss
        return super().forward(input=mean, target=target, var=var)
