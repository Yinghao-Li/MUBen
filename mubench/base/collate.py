import torch
import logging
import numpy as np

from seqlbtoolkit.training.dataset import (
    Batch,
    instance_list_to_feature_lists
)

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
        features, smiles, lbs, masks = instance_list_to_feature_lists(instance_list)

        feature_batch = torch.from_numpy(np.stack(features)).to(torch.float)
        lbs_batch = torch.as_tensor(lbs).to(self._lbs_type)
        masks_batch = torch.as_tensor(masks)

        return Batch(features=feature_batch, lbs=lbs_batch, masks=masks_batch)
