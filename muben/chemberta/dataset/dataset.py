"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: Dataset for ChemBERTa.
"""

import logging

from transformers import AutoTokenizer
from muben.base.dataset import pack_instances, Dataset as BaseDataset

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    """
    Dataset class for ChemBERTa.

    Extends the base dataset to create data features specifically tailored for ChemBERTa.
    It tokenizes the data using the provided tokenizer and manages the creation of instances.

    Attributes
    ----------
    _atom_ids : list
        List of tokenized instance IDs.
    """

    def __init__(self):
        super().__init__()

        self._atom_ids = None

    def create_features(self, config):
        """
        Tokenizes the data and creates features for the dataset.

        Uses the tokenizer specified in the config to tokenize the data.
        This method must be called before `get_instances`.

        Parameters
        ----------
        config : object
            Configuration object which should have attribute 'pretrained_model_name_or_path'.

        Returns
        -------
        Dataset
            Returns the dataset instance with populated features.
        """
        tokenizer_name = config.pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenized_instances = tokenizer(
            self._smiles, add_special_tokens=True, truncation=True
        )

        self._atom_ids = tokenized_instances.input_ids

    def get_instances(self):
        """
        Creates data instances using the atom IDs, labels, and masks.

        It utilizes the pack_instances function to combine atom IDs, labels, and masks into structured data instances.

        Returns
        -------
        list
            List of data instances where each instance is a combination of atom IDs, labels, and masks.
        """
        data_instances = pack_instances(
            atom_ids=self._atom_ids, lbs=self.lbs, masks=self.masks
        )
        return data_instances
