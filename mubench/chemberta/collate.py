import torch
import logging
import numpy as np
from transformers import AutoTokenizer

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

        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)
        self._pad_id = tokenizer.pad_token_id

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
        atom_ids, lbs, masks = instance_list_to_feature_lists(instance_list)

        atom_lengths = [len(inst) for inst in atom_ids]
        max_atom_length = max(atom_lengths)

        atom_ids_batch = torch.tensor([inst_ids + [self._pad_id] * (max_atom_length - atom_length)
                                       for inst_ids, atom_length in zip(atom_ids, atom_lengths)])
        attn_masks_batch = (atom_ids_batch != self._pad_id).to(torch.long)
        lbs_batch = torch.from_numpy(np.stack(lbs)).to(self._lbs_type)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(atom_ids=atom_ids_batch, attn_masks=attn_masks_batch, lbs=lbs_batch, masks=masks_batch)
