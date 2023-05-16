from mubench.utils.data import pack_instances
from mubench.base.dataset import Dataset as BaseDataset


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
        data_instances = pack_instances(smiles=self.smiles, lbs=self.lbs, masks=self.masks)
        return data_instances
