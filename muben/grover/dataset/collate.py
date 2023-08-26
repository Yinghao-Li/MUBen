"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: GROVER data collator.
"""

import torch
import logging
import numpy as np

from muben.base.dataset import Batch, unpack_instances
from .molgraph import BatchMolGraph

logger = logging.getLogger(__name__)


class Collator:
    """
    A data collator for GROVER.

    This class is responsible for preparing batches of data from the given instances.
    These batches are then used for training or evaluation with the GROVER model.

    Attributes
    ----------
    _task : str
        The task type extracted from the provided configuration.
    _lbs_type : torch.dtype
        The datatype for labels. Typically set to torch.float.
    """

    def __init__(self, config):
        """
        Initialize the Collator with a given configuration.

        Parameters
        ----------
        config : object
            A configuration object with at least a 'task_type' attribute.
        """
        self._task = config.task_type
        self._lbs_type = torch.float

    def __call__(self, instance_list: list, *args, **kwargs) -> Batch:
        """
        Prepare a batch from a list of instances.

        Given a list of instances, it combines them into a single batch
        ready for training or evaluation. It uses the `BatchMolGraph` utility to
        batch molecule graphs and ensures labels and masks are in the correct datatype.

        Parameters
        ----------
        instance_list : list
            List of data instances, each instance typically consists of a molecule
            graph, associated labels, and masks.

        Returns
        -------
        Batch
            A combined batch of the instances containing batched molecule graphs, labels,
            and masks.
        """
        molecule_graphs, lbs, masks = unpack_instances(instance_list)

        molecule_graphs_batch = BatchMolGraph(molecule_graphs)
        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(
            molecule_graphs=molecule_graphs_batch,
            lbs=lbs_batch,
            masks=masks_batch,
        )
