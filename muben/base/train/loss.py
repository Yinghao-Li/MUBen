
from torch import Tensor
from torch.nn import GaussianNLLLoss as GaussianNLLLossBase
from torch.nn import functional as F

__all__ = ["GaussianNLLLoss"]


# noinspection PyShadowingBuiltins
class GaussianNLLLoss(GaussianNLLLossBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs):
        mean = input[..., 0]
        var = F.softplus(input[..., 1]) + 1e-6
        return super().forward(input=mean, target=target, var=var)
