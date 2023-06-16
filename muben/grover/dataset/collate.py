import torch
import logging
import numpy as np

from muben.base.dataset import (
    Batch,
    unpack_instances
)
from .molgraph import BatchMolGraph

logger = logging.getLogger(__name__)


class Collator:

    def __init__(self, config):
        self._task = config.task_type
        self._lbs_type = torch.float

    def __call__(self, instance_list: list, *args, **kwargs) -> Batch:
        """
        function call

        Parameters
        ----------
        instance_list: a list of instance

        Returns
        -------
        a Batch of instances
        """
        molecule_graphs, lbs, masks = unpack_instances(instance_list)

        molecule_graphs_batch = BatchMolGraph(molecule_graphs)
        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(molecule_graphs=molecule_graphs_batch, lbs=lbs_batch, masks=masks_batch)
