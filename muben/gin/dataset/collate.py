"""
# Author: Yinghao Li
# Modified: August 14th, 2023
# ---------------------------------------
# Description: Collate function for GIN model
"""


import torch
import logging
import numpy as np
import itertools

from torch_geometric.data import Batch as pygBatch
from muben.base.dataset import Batch, unpack_instances

logger = logging.getLogger(__name__)


class Collator:

    def __init__(self, config):
        self._task = config.task_type
        self._lbs_type = torch.float

    def __call__(self, instances) -> Batch:
        """
        function call

        Returns
        -------
        a Batch of instances
        """
        graphs, lbs, masks = unpack_instances(instances)
        batched_graphs = pygBatch().from_data_list(graphs)

        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(graphs=batched_graphs,
                     lbs=lbs_batch,
                     masks=masks_batch,
                     batch_size=len(lbs_batch))
