"""
# Author: Yinghao Li
# Modified: August 8th, 2023
# ---------------------------------------
# Description: Collate function for GIN model
"""


import torch
import logging
import numpy as np
import itertools

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
        atom_ids, edge_indices, lbs, masks = unpack_instances(instances)

        mol_ids = torch.tensor(
            list(itertools.chain.from_iterable([[i]*len(a) for i, a in enumerate(atom_ids)])),
            dtype=torch.long
        )

        atoms_batch = torch.tensor(list(itertools.chain.from_iterable(atom_ids)), dtype=torch.long)

        edge_batch = torch.from_numpy(np.concatenate(edge_indices, axis=0)).to(torch.long)
        edge_batch_ids = torch.tensor(
            list(itertools.chain.from_iterable([[i] * len(e) for i, e in enumerate(edge_indices)]))
        )
        edge_batch += edge_batch_ids.unsqueeze(-1)

        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(atom_ids=atoms_batch,
                     edge_indices=edge_batch.T,
                     mol_ids=mol_ids,
                     lbs=lbs_batch,
                     masks=masks_batch,
                     batch_size=len(lbs_batch))
