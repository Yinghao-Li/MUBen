import torch
import random
import numpy as np
from typing import Optional, List
from functools import cache


class Status:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Batch:
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        for k, v in self.__dict__.items():
            try:
                setattr(self, k, v.to(device))
            except AttributeError:
                pass
        return self

    @cache
    def __len__(self):
        for v in list(self.__dict__.values()):
            if isinstance(v, torch.Tensor):
                return v.shape[0]


def pack_instances(**kwargs) -> List[dict]:
    """
    Convert attribute lists to a list of data instances, each is a dict with attribute names as keys
    and one datapoint attribute values as values
    """

    instance_list = list()
    keys = tuple(kwargs.keys())

    for inst_attrs in zip(*tuple(kwargs.values())):
        inst = dict(zip(keys, inst_attrs))
        instance_list.append(inst)

    return instance_list


def unpack_instances(instance_list: List[dict], attr_names: Optional[List[str]] = None):
    """
    Convert a list of dict-type instances to a list of value lists,
    each contains all values within a batch of each attribute

    Parameters
    ----------
    instance_list: a list of attributes
    attr_names: the name of the needed attributes. Notice that this variable should be specified
        for Python versions that does not natively support ordered dict
    """
    if not attr_names:
        attr_names = list(instance_list[0].keys())
    attribute_lists = [[inst[name] for inst in instance_list] for name in attr_names]

    return attribute_lists


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).
    Modified from PyTorch's original implementation

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    # ^^ safe to call this function even if cuda is not available
    torch.cuda.manual_seed_all(seed)
