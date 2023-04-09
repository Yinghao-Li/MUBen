import torch
import logging
import numpy as np
import torch.nn.functional as F

from seqlbtoolkit.training.dataset import (
    Batch,
)
from mubench.utils.data import unpack_instances

logger = logging.getLogger(__name__)


class Collator:

    def __init__(self, config, atom_pad_idx=0):
        self._atom_pad_idx = atom_pad_idx
        self._task = config.task_type
        self._lbs_type = torch.float \
            if config.task_type == 'regression' or not config.binary_classification_with_softmax \
            else torch.long

    def __call__(self, instances) -> Batch:
        """
        function call

        Returns
        -------
        a Batch of instances
        """
        atoms, coordinates, distances, edge_types, lbs, masks = unpack_instances(instances)

        lengths = [tk.shape[1] for tk in atoms]
        max_length = max(lengths)

        atoms_batch = torch.cat([
            torch.cat([a, a.new(a.shape[0], max_length-a.shape[1]).fill_(self._atom_pad_idx)], dim=1) for a in atoms
        ], dim=0)
        coordinates_batch = torch.cat([
            torch.cat([c, c.new(c.shape[0], max_length-c.shape[1], c.shape[2]).fill_(0)], dim=1) for c in coordinates
        ], dim=0)
        distances_batch = torch.cat([
            F.pad(d, (0, max_length-d.shape[1], 0, max_length-d.shape[1]), value=0) for d in distances
        ], dim=0)
        edge_types_batch = torch.cat([
            F.pad(e, (0, max_length-e.shape[1], 0, max_length-e.shape[1]), value=0) for e in edge_types
        ], dim=0)

        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(atoms=atoms_batch,
                     coordinates=coordinates_batch,
                     distances=distances_batch,
                     edge_types=edge_types_batch,
                     lbs=lbs_batch,
                     masks=masks_batch)
