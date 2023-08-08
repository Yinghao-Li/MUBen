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
        atoms, coords, lbs, masks = unpack_instances(instances)

        mol_ids = torch.tensor(
            list(itertools.chain.from_iterable([[i]*len(a) for i, a in enumerate(atoms)])),
            dtype=torch.long
        )
        atoms_batch = torch.tensor(list(itertools.chain.from_iterable(atoms)), dtype=torch.long)
        coords_batch = torch.from_numpy(np.concatenate(coords, axis=0)).to(torch.float)

        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(atoms=atoms_batch,
                     coords=coords_batch,
                     mol_ids=mol_ids,
                     lbs=lbs_batch,
                     masks=masks_batch)
