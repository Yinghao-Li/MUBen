"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description:

This module provides a Collator class for the TorchMD-NET dataset. The Collator class 
facilitates the aggregation of individual data instances into a batch format suitable for 
model training and inference.
"""

import torch
import logging
import numpy as np
import itertools

from ..dataset import Batch, unpack_instances

logger = logging.getLogger(__name__)


class Collator3D:
    """
    Collator class for aggregating individual data instances into a batch.

    The Collator class defines the necessary functionality to transform a list of instances
    into a batch that can be used for training or inference with the TorchMD-NET model.

    Attributes
    ----------
    _task : str
        The type of task (e.g., "regression", "classification").
    _lbs_type : torch.dtype
        Data type for labels in the batch. Defaults to torch.float.
    """

    def __init__(self, config):
        """
        Initialize the Collator class.

        Parameters
        ----------
        config : object
            Configuration object that defines the task type of the collator.
        """
        self._task = config.task_type
        self._lbs_type = torch.float

    def __call__(self, instances) -> Batch:
        """
        Convert a list of instances into a batch format.

        This method unpacks the instances and aggregates the individual attributes
        (atoms, coordinates, labels, masks) into a batch format.

        Parameters
        ----------
        instances : list
            List of individual data instances.

        Returns
        -------
        Batch
            Aggregated batch of instances.
        """
        atoms, coords, lbs, masks = unpack_instances(instances)

        mol_ids = torch.tensor(
            list(itertools.chain.from_iterable([[i] * len(a) for i, a in enumerate(atoms)])),
            dtype=torch.long,
        )
        atoms_batch = torch.tensor(list(itertools.chain.from_iterable(atoms)), dtype=torch.long)
        coords_batch = torch.from_numpy(np.concatenate(coords, axis=0)).to(torch.float)

        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(
            atoms=atoms_batch,
            coords=coords_batch,
            mol_ids=mol_ids,
            lbs=lbs_batch,
            masks=masks_batch,
            batch_size=len(lbs_batch),
        )
