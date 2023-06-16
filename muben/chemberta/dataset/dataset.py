import logging

from transformers import AutoTokenizer
from muben.base.dataset import (
    pack_instances,
    Dataset as BaseDataset
)

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._atom_ids = None

    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """
        tokenizer_name = config.pretrained_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenized_instances = tokenizer(self._smiles, add_special_tokens=True, truncation=True)

        self._atom_ids = tokenized_instances.input_ids

    def get_instances(self):

        data_instances = pack_instances(atom_ids=self._atom_ids, lbs=self.lbs, masks=self.masks)
        return data_instances
