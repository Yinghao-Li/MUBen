
from tqdm.auto import tqdm
from multiprocessing import get_context

from muben.utils.data import pack_instances
from muben.base.dataset import Dataset as BaseDataset
from .molgraph import MolGraph, MolGraphAttrs


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._molecule_graphs = None

    # noinspection PyTypeChecker
    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """

        with get_context('fork').Pool(config.num_preprocess_workers) as pool:
            self._molecule_graphs = [g for g in tqdm(
                pool.imap(self.get_mol_attr, self.smiles), total=len(self._smiles)
            )]

    @staticmethod
    def get_mol_attr(smiles):
        return MolGraphAttrs().from_mol_graph(MolGraph(smiles))

    def get_instances(self):
        data_instances = pack_instances(molecule_graphs=self._molecule_graphs, lbs=self.lbs, masks=self.masks)
        return data_instances
