import torch
import logging
import numpy as np

from seqlbtoolkit.training.dataset import (
    Batch,
    instance_list_to_feature_lists
)
from mubench.grover.data.molgraph import BatchMolGraph

logger = logging.getLogger(__name__)


class Collator:

    def __init__(self, config):
        self._task = config.task_type
        self._lbs_type = torch.float \
            if config.task_type == 'regression' or not config.binary_classification_with_softmax \
            else torch.long

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
        molecule_graphs, lbs, masks = instance_list_to_feature_lists(instance_list)

        # TODO: copied from GROVER implementation but may not be the most efficient way to batchify the graphs
        molecule_graphs_batch = BatchMolGraph(molecule_graphs)
        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(molecule_components=molecule_graphs_batch.get_components(), lbs=lbs_batch, masks=masks_batch)
