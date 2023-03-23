from tqdm.auto import tqdm
from seqlbtoolkit.training.dataset import (
    DataInstance,
    feature_lists_to_instance_list,
)

from mubench.base.dataset import Dataset as BaseDataset
from .molgraph import MolGraph


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._molecule_graph = None

    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """
        bond_drop_rate = config.bond_drop_rate
        self._molecule_graph = [MolGraph(smiles, bond_drop_rate).remove_intermediate_attrs()
                                for smiles in tqdm(self._smiles)]

    def get_instances(self):

        data_instances = feature_lists_to_instance_list(
            DataInstance,
            molecule_graph=self._molecule_graph, lbs=self.lbs, masks=self.masks
        )

        return data_instances
