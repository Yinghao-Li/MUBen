import logging
from multiprocessing import get_context
from tqdm.auto import tqdm

from .utils import smiles_to_coords
from .process import ProcessingPipeline
from mubench.base.dataset import Dataset as BaseDataset


logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._atoms = None
        self._cooridnates = None
        self.data_processor = None

    def __getitem__(self, idx):
        atoms, coordinates, distances, edge_types = self.data_processor(
            atoms=self._atoms[idx],
            coordinates=self._cooridnates[idx]
        )
        feature_dict = {
            'atoms': atoms,
            'coordinates': coordinates,
            'distances': distances,
            'edge_types': edge_types,
            'lbs': self.lbs[idx],
            'masks': self.masks[idx]
        }
        return feature_dict

    # noinspection PyMethodOverriding
    def prepare(self, config, partition, dictionary):
        super().prepare(config, partition)

        processor_variant = 'training' if partition == 'train' else 'inference'
        data_processor = ProcessingPipeline(
            dictionary=dictionary,
            max_atoms=config.max_atoms,
            max_seq_len=config.max_seq_len,
            remove_hydrogen_flag=config.remove_hydrogen,
            remove_polar_hydrogen_flag=config.remove_polar_hydrogen
        )
        self.data_processor = getattr(data_processor, f'process_{processor_variant}')
        return self

    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """

        self._atoms = list()
        self._cooridnates = list()

        with get_context('fork').Pool(config.n_feature_generating_threads) as pool:

            for outputs in tqdm(pool.imap(smiles_to_coords, self._smiles), total=len(self._smiles)):
                atoms, coordinates = outputs
                self._atoms.append(atoms)
                self._cooridnates.append(coordinates)

        return self

    def get_instances(self):

        return None

