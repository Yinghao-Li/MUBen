"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: Collate function for ChemBERTa.
"""

import torch
import logging
import numpy as np
from transformers import AutoTokenizer

from ..dataset import Batch, unpack_instances

logger = logging.getLogger(__name__)


class Collator:
    """
    Collator class for ChemBERTa.

    The collator is responsible for combining instances into a batch for model processing.
    It handles token padding to ensure consistent tensor shapes across batches. It also constructs attention masks.

    Attributes
    ----------
    _pad_id : int
        The padding token ID.
    """

    def __init__(self, config):
        """
        Initialize the Collator.

        Parameters
        ----------
        config : object
            The configuration object which should have attributes 'task_type' and 'pretrained_model_name_or_path'.
        """
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)
        self._pad_id = tokenizer.pad_token_id

    def __call__(self, instance_list: list, *args, **kwargs) -> Batch:
        """
        Collate instances into a batch.

        Given a list of instances, this method pads the atom_ids to ensure all instances have the same length.
        Attention masks are generated to distinguish actual tokens from padding tokens. The resulting instances
        are combined into a batch.

        Parameters
        ----------
        instance_list : list
            A list of instances where each instance is expected to have atom_ids, labels (lbs), and masks.

        Returns
        -------
        Batch
            A batch containing padded atom_ids, attention masks, labels, and masks.
        """
        atom_ids, lbs, masks = unpack_instances(instance_list)

        atom_lengths = [len(inst) for inst in atom_ids]
        max_atom_length = max(atom_lengths)

        atom_ids_batch = torch.tensor(
            [
                inst_ids + [self._pad_id] * (max_atom_length - atom_length)
                for inst_ids, atom_length in zip(atom_ids, atom_lengths)
            ]
        )
        attn_masks_batch = (atom_ids_batch != self._pad_id).to(torch.long)
        lbs_batch = torch.from_numpy(np.stack(lbs)).to(torch.float)
        masks_batch = torch.from_numpy(np.stack(masks))

        return Batch(
            atom_ids=atom_ids_batch,
            attn_masks=attn_masks_batch,
            lbs=lbs_batch,
            masks=masks_batch,
        )
