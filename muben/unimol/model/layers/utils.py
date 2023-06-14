# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn.functional as F
from typing import List, Callable

HAS_MULTI_TENSOR = False
HAS_FUSED_ROUNDING = False

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
    HAS_MULTI_TENSOR = False
    HAS_FUSED_ROUNDING = False

logger = logging.getLogger(__name__)


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int):
    return t.reshape(t.shape[:-num_dims] + (-1,))


def masked_mean(mask, value, dim, eps=1e-10):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def one_hot(x, num_classes, dtype=torch.float32):
    x_one_hot = torch.zeros(*x.shape, num_classes, dtype=dtype, device=x.device)
    x_one_hot.scatter_(-1, x.long().unsqueeze(-1), 1)
    return x_one_hot
