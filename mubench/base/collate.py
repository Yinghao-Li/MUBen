import torch
import logging

from seqlbtoolkit.training.dataset import (
    Batch,
    instance_list_to_feature_lists
)

logger = logging.getLogger(__name__)


class Collator:

    def __init__(self, task="classification"):
        assert task in ("classification", "regression")
        self._task = task
        self._lbs_dtype = torch.long if self._task == "classification" else torch.float

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
        features, lbs = instance_list_to_feature_lists(instance_list)

        feature_batch = torch.stack(features)
        lbs_batch = torch.as_tensor(lbs, dtype=self._lbs_dtype)

        return Batch(features=feature_batch, lbs=lbs_batch)
