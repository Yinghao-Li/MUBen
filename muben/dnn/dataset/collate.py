import torch
import logging
import numpy as np

from muben.base.dataset import Batch, unpack_instances


logger = logging.getLogger(__name__)


class Collator:

    def __init__(self, *args, **kwargs):
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
        features, lbs, masks = unpack_instances(instance_list)

        feature_batch = torch.from_numpy(np.stack(features)).to(torch.float)
        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(features=feature_batch, lbs=lbs_batch, masks=masks_batch)
