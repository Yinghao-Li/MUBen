"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: Collate function for the GIN (Graph Isomorphism Network) model.
"""

import torch
import logging
import numpy as np

from torch_geometric.data import Batch as pygBatch
from ..dataset import Batch, unpack_instances

logger = logging.getLogger(__name__)


class Collator2D:
    """
    Collator for the GIN model.

    This class provides a mechanism to collate individual data instances into a single batch for the GIN model.
    """

    def __init__(self, config):
        """
        Initialize the collator.

        Parameters
        ----------
        config : object
            Configuration object containing necessary hyperparameters and settings.
            Expected to have attributes `task_type`.
        """
        self._task = config.task_type
        self._lbs_type = torch.float

    def __call__(self, instances) -> Batch:
        """
        Collate individual instances into a batch.

        This method collates input instances into a single batch that is compatible with the GIN model.

        Parameters
        ----------
        instances : list
            List of instances where each instance comprises a graph, labels, and masks.

        Returns
        -------
        Batch
            A single batch containing batched graphs, labels, and masks, and the batch size.
        """
        graphs, lbs, masks = unpack_instances(instances)
        batched_graphs = pygBatch().from_data_list(graphs)

        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(
            graphs=batched_graphs,
            lbs=lbs_batch,
            masks=masks_batch,
            batch_size=len(lbs_batch),
        )
