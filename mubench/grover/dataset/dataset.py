from tqdm.auto import tqdm
from seqlbtoolkit.training.dataset import (
    DataInstance,
    feature_lists_to_instance_list,
)

from mubench.base.dataset import Dataset as BaseDataset
from .molgraph import MolGraph, MolGraphAttrs


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
        pass

    def get_instances(self):

        data_instances = feature_lists_to_instance_list(
            DataInstance,
            smiles=self.smiles, lbs=self.lbs, masks=self.masks
        )

        return data_instances
