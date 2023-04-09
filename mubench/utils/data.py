import torch
import random
import numpy as np
from typing import Optional, List


def unzip_list_of_dicts(instance_list: List[dict], feature_names: Optional[List[str]] = None):
    if not feature_names:
        feature_names = list(instance_list[0].keys())

    features_lists = list()
    for name in feature_names:
        features_lists.append([inst[name] for inst in instance_list])

    return features_lists


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
