"""
# Author: Yinghao Li
# Modified: August 8th, 2023
# ---------------------------------------
# Description: GIN dataset.
"""


import logging
import torch
from tqdm.auto import tqdm
from torch_geometric.data import Data

from muben.utils.chem import smiles_to_2d_graph
from muben.base.dataset import pack_instances, Dataset as BaseDataset


logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._graphs = None

    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """

        self._graphs = list()
        for smiles in tqdm(self._smiles):
            atom_ids, edge_indices = smiles_to_2d_graph(smiles)
            data = Data(
                x=torch.tensor(atom_ids, dtype=torch.long),
                edge_index=torch.tensor(edge_indices, dtype=torch.long)
                .t()
                .contiguous(),
            )
            self._graphs.append(data)

        return self

    def get_instances(self):
        data_instances = pack_instances(
            graphs=self._graphs, lbs=self.lbs, masks=self.masks
        )

        return data_instances
