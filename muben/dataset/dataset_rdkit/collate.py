"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: Data collator for DNN
"""

import torch
import logging
import numpy as np

from ..dataset import Batch, unpack_instances


logger = logging.getLogger(__name__)


class CollatorRDKit:
    """
    Collator for creating data batches for the Deep Neural Network (DNN).

    The collator takes a list of data instances, processes, and collates them into a batch.
    This is primarily used for training and evaluation of the DNN model.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Collator instance.
        """

    def __call__(self, instance_list: list, *args, **kwargs) -> Batch:
        """
        Create a batch from a list of instances.

        Processes and collates the input list of data instances into a single batch
        suitable for DNN training or evaluation.

        Parameters
        ----------
        instance_list : list
            List of data instances to be processed. Each instance typically contains feature data,
            labels, and masks.

        Returns
        -------
        Batch
            Processed batch of data, containing features, labels, and masks.
        """
        features, lbs, masks = unpack_instances(instance_list)

        feature_batch = torch.from_numpy(np.stack(features)).to(torch.float)
        lbs_batch = torch.from_numpy(np.stack(lbs)).to(torch.float)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(features=feature_batch, lbs=lbs_batch, masks=masks_batch)
