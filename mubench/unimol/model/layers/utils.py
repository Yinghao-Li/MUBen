# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import numpy as np
from functools import partial
from typing import List, Callable
import torch
import torch.nn.functional as F

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
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "tanh",
        "linear",
    ]


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


@contextlib.contextmanager
def torch_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return

    def check_seed(s):
        assert type(s) == int or type(s) == np.int32 or type(s) == np.int64

    check_seed(seed)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 1e8)
    state = get_rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        set_rng_state(state)


class CudaEnvironment(object):
    def __init__(self):
        cur_device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
        self.name = prop.name
        self.major = prop.major
        self.minor = prop.minor
        self.total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024

    @staticmethod
    def pretty_print_cuda_env_list(cuda_env_list):
        """
        Given a list of CudaEnviorments, pretty print them
        """
        num_workers = len(cuda_env_list)
        center = "CUDA enviroments for all {} workers".format(num_workers)
        banner_len = 40 - len(center) // 2
        first_line = "*" * banner_len + center + "*" * banner_len
        logger.info(first_line)
        for r, env in enumerate(cuda_env_list):
            logger.info(
                "rank {:3d}: ".format(r)
                + "capabilities = {:2d}.{:<2d} ; ".format(env.major, env.minor)
                + "total memory = {:.3f} GB ; ".format(env.total_memory_in_GB)
                + "name = {:40s}".format(env.name)
            )
        logger.info(first_line)


def csv_str_list(x):
    return x.split(",")


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_str_dict(x, type=dict):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    return x


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


def checkpoint_sequential(
        functions,
        input,
        enabled=True,
):
    def wrap_tuple(a):
        return (a,) if type(a) is not tuple else a

    def exec(func, a):
        return wrap_tuple(func(*a))

    def get_wrap_exec(func):
        def wrap_exec(*a):
            return exec(func, a)

        return wrap_exec

    input = wrap_tuple(input)

    is_grad_enabled = torch.is_grad_enabled()

    if enabled and is_grad_enabled:
        for func in functions:
            input = torch.utils.checkpoint.checkpoint(get_wrap_exec(func), *input)
    else:
        for func in functions:
            input = exec(func, input)
    return input


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int):
    return t.reshape(t.shape[:-num_dims] + (-1,))


def masked_mean(mask, value, dim, eps=1e-10):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x, num_classes, dtype=torch.float32):
    x_one_hot = torch.zeros(*x.shape, num_classes, dtype=dtype, device=x.device)
    x_one_hot.scatter_(-1, x.long().unsqueeze(-1), 1)
    return x_one_hot


def batched_gather(data, inds, dim=0, num_batch_dims=0):
    assert dim < 0 or dim - num_batch_dims >= 0
    ranges = []
    for i, s in enumerate(data.shape[:num_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - num_batch_dims)
    ]
    remaining_dims[dim - num_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        try:
            return fn(tree)
        except:
            raise ValueError(f"cannot apply {fn} on {tree}.")
    else:
        raise ValueError(f"{type(tree)} not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def fp32_to_bf16_sr(t, o):
    r = (torch.rand(size=t.size(), device=t.device, dtype=torch.float32) - 0.5) / 256
    m, e = torch.frexp(t)
    t = t + torch.ldexp(r, e)
    o.data.copy_(t.bfloat16())


@contextlib.contextmanager
def validate_with_ema(trainer, ema=False):
    if not ema:
        yield
        return
    _wrapped_model = trainer._wrapped_model
    trainer._wrapped_model = trainer.ema.model_ema
    try:
        yield
    finally:
        trainer._wrapped_model = _wrapped_model
