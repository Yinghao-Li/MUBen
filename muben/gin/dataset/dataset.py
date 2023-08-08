"""
# Author: Yinghao Li
# Modified: August 8th, 2023
# ---------------------------------------
# Description: GIN dataset.
"""


import logging
from tqdm.auto import tqdm

from muben.utils.chem import smiles_to_2d_graph
from muben.base.dataset import (
    pack_instances,
    Dataset as BaseDataset
)


logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._atom_ids = None
        self._edge_indices = None

    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """

        self._atom_ids = list()
        self._edge_indices = list()
        for smiles in tqdm(self._smiles):
            atom_ids, edge_indices = smiles_to_2d_graph(smiles)
            self._atom_ids.append(atom_ids)
            self._edge_indices.append(edge_indices)

        return self

    def get_instances(self):
        data_instances = pack_instances(
            atom_ids=self._atom_ids, edge_indices=self._edge_indices, lbs=self.lbs, masks=self.masks
        )

        return data_instances
