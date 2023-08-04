"""
# Author: Yinghao Li
# Modified: August 4th, 2023
# ---------------------------------------
# Description: TorchMD-NET dataset.
"""


import os
import lmdb
import pickle
import logging
from functools import partial
from multiprocessing import get_context
from tqdm.auto import tqdm

from muben.utils.chem import smiles_to_coords, smiles_to_atom_ids
from muben.base.dataset import (
    pack_instances,
    Dataset as BaseDataset
)

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._partition = None
        self._atoms = None
        self._cooridnates = None

    def prepare(self, config, partition, **kwargs):
        self._partition = partition
        super().prepare(config, partition, **kwargs)
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
                self._cooridnates.append(data['coordinates'][0])

        else:
            logger.info("Generating 3D Coordinates.")
            s2c = partial(smiles_to_coords, n_conformer=1)
            with get_context('fork').Pool(config.num_preprocess_workers) as pool:
                for outputs in tqdm(pool.imap(s2c, self._smiles), total=len(self._smiles)):
                    _, coordinates = outputs
                    self._cooridnates.append(coordinates[0])

        for smiles, coords in zip(self._smiles, self._cooridnates):
            atom_ids = smiles_to_atom_ids(smiles)
            assert len(atom_ids) == len(coords)
            self._atoms.append(atom_ids)

        return self

    def get_instances(self):
        data_instances = pack_instances(
            atoms=self._atoms, coords=self._cooridnates, lbs=self.lbs, masks=self.masks
        )

        return data_instances
