
from torch import Tensor
from torch.nn import GaussianNLLLoss as GaussianNLLLossBase
from torch.nn import functional as F


# noinspection PyShadowingBuiltins
class GaussianNLLLoss(GaussianNLLLossBase):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super().__init__(full=full, eps=eps, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs):
        mean = input[..., 0]
        var = F.softplus(input[..., 1]) + 1e-6
        return super().forward(input=mean, target=target, var=var)
