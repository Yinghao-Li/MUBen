"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: Dataset class tailored for the GIN (Graph Isomorphism Network) model.
"""

import logging
import torch
from tqdm.auto import tqdm
from torch_geometric.data import Data

from muben.utils.chem import smiles_to_2d_graph
from ..dataset import pack_instances, Dataset as BaseDataset


logger = logging.getLogger(__name__)


class Dataset2D(BaseDataset):
    """
    Dataset tailored for the GIN model.

    This class provides mechanisms for creating graph features from SMILES strings and generating instances compatible with the GIN model.
    """

    def __init__(self):
        """
        Initialize the GIN dataset.

        Initializes the base dataset and prepares a placeholder for graph representations.
        """
        super().__init__()
        self._graphs = None

    def create_features(self, config):
        """
        Generate graph representations from SMILES strings.

        Iterates over SMILES strings, converts them into graph representations with node (atom) features and edge indices,
        and stores them for further processing.

        Parameters
        ----------
        config : object
            Configuration object containing necessary hyperparameters and settings.

        Returns
        -------
        Dataset
            Self reference for potential method chaining.
        """
        self._graphs = list()
        for smiles in tqdm(self._smiles):
            atom_ids, edge_indices = smiles_to_2d_graph(smiles)
            data = Data(
                x=torch.tensor(atom_ids, dtype=torch.long),
                edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            )
            self._graphs.append(data)

        return self

    def get_instances(self):
        """
        Pack graph data, labels, and masks into data instances.

        Returns
        -------
        list
            List of packed instances, each containing a graph, labels, and masks.
        """
        data_instances = pack_instances(graphs=self._graphs, lbs=self.lbs, masks=self.masks)

        return data_instances
