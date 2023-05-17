import os
import lmdb
import pickle
import logging
from multiprocessing import get_context
from tqdm.auto import tqdm

from .utils import smiles_to_coords
from .process import ProcessingPipeline
from .dictionary import Dictionary
from mubench.base.dataset import Dataset as BaseDataset
from mubench.utils.data import pack_instances

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._partition = None
        self._atoms = None
        self._cooridnates = None
        self.data_processor = None

    def prepare(self, config, partition, dictionary=None):
        self._partition = partition

        if not dictionary:
            dictionary = Dictionary.load()
            dictionary.add_symbol("[MASK]", is_special=True)

        processor_variant = 'training' if partition == 'train' else 'inference'
        data_processor = ProcessingPipeline(
            dictionary=dictionary,
            max_atoms=config.max_atoms,
            max_seq_len=config.max_seq_len,
            remove_hydrogen_flag=config.remove_hydrogen,
            remove_polar_hydrogen_flag=config.remove_polar_hydrogen
        )
        self.data_processor = getattr(data_processor, f'process_{processor_variant}')

        super().prepare(config, partition)

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

        # load feature if UniMol LMDB file exists else generate feature
        unimol_feature_path = os.path.join(config.unimol_feature_dir, f"{self._partition}.lmdb")
        print(unimol_feature_path)
        if os.path.exists(unimol_feature_path):
            logger.info("Loading features form pre-processed Uni-Mol LMDB")
            env = lmdb.open(unimol_feature_path,
                            subdir=False,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            max_readers=256)
            txn = env.begin()
            keys = list(txn.cursor().iternext(values=False))
            for idx in tqdm(keys):
                datapoint_pickled = txn.get(idx)
                data = pickle.loads(datapoint_pickled)
                self._atoms.append(data['atoms'])
                self._cooridnates.append(data['coordinates'])

        else:
            # TODO: there might be some issue with this generation method
            logger.info("Generating Uni-Mol features.")
            with get_context('fork').Pool(config.num_preprocess_workers) as pool:
                for outputs in tqdm(pool.imap(smiles_to_coords, self._smiles), total=len(self._smiles)):
                    atoms, coordinates = outputs
                    self._atoms.append(atoms)
                    self._cooridnates.append(coordinates)

        return self

    def get_instances(self):
        atoms = list()
        coordinates = list()
        distances = list()
        edge_types = list()

        for atom, coord in tqdm(zip(self._atoms, self._cooridnates), total=len(self._atoms)):
            a, c, d, e = self.data_processor(atom, coord)
            atoms.append(a)
            coordinates.append(c)
            distances.append(d)
            edge_types.append(e)

        data_instances = pack_instances(
            atoms=atoms, coordinates=coordinates, distances=distances, edge_types=edge_types,
            lbs=self.lbs, masks=self.masks
        )

        return data_instances
