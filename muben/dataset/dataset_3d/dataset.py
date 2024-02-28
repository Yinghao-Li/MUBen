"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: TorchMD-NET dataset.
"""

import os.path as op
import logging
from functools import partial
from multiprocessing import get_context
from tqdm.auto import tqdm

from muben.utils.chem import (
    smiles_to_coords,
    smiles_to_atom_ids,
    atom_to_atom_ids,
)
from muben.utils.io import load_lmdb, load_unimol_preprocessed
from ..dataset import pack_instances, Dataset as BaseDataset

logger = logging.getLogger(__name__)


class Dataset3D(BaseDataset):
    """
    TorchMD-NET dataset class.

    Extends the BaseDataset class to provide specific functionality for handling TorchMD-NET data.

    Attributes
    ----------
    _partition : str
        The type of data partition (e.g., "train", "test", "valid").
    _atoms : list
        List storing atom ids for each molecule.
    _coordinates : list
        List storing coordinates for each molecule.
    """

    def __init__(self):
        super().__init__()

        self._partition = None
        self._atoms = None
        self._coordinates = None

    def prepare(self, config, partition, **kwargs) -> "Dataset3D":
        """
        Prepare the dataset.

        Parameters
        ----------
        config : object
            Configuration object containing various dataset settings.
        partition : str
            The type of data partition (e.g., "train", "test", "valid").

        Returns
        -------
        Dataset
            The prepared dataset.
        """
        self._partition = partition
        super().prepare(config, partition, **kwargs)
        return self

    def create_features(self, config):
        """
        Create data features for the dataset.

        Generates or loads preprocessed molecular features.

        Parameters
        ----------
        config : object
            Configuration object containing various dataset settings.

        Returns
        -------
        Dataset
            The dataset with created features.
        """

        self._atoms = list()
        self._coordinates = list()

        # load feature if UniMol LMDB file exists else generate feature
        unimol_feature_path = op.join(config.unimol_feature_dir, f"{self._partition}.lmdb")

        if op.exists(config.unimol_feature_dir) and op.exists(unimol_feature_path):
            logger.info("Loading features form pre-processed Uni-Mol LMDB")

            if not config.random_split:
                unimol_atoms, self._coordinates = load_lmdb(unimol_feature_path, ["atoms", "coordinates"])
            else:
                unimol_data = load_unimol_preprocessed(config.unimol_feature_dir)
                id2data_mapping = {
                    idx: (a, c)
                    for idx, a, c in zip(
                        unimol_data["ori_index"],
                        unimol_data["atoms"],
                        unimol_data["coordinates"],
                    )
                }
                unimol_atoms = [id2data_mapping[idx][0] for idx in self._ori_ids]
                self._coordinates = [id2data_mapping[idx][1] for idx in self._ori_ids]
            self._coordinates = [c[0] for c in self._coordinates]
        else:
            logger.info("Generating 3D Coordinates.")
            s2c = partial(smiles_to_coords, n_conformer=1)
            with get_context("fork").Pool(config.num_preprocess_workers) as pool:
                for outputs in tqdm(pool.imap(s2c, self._smiles), total=len(self._smiles)):
                    _, coordinates = outputs
                    self._coordinates.append(coordinates[0])
            unimol_atoms = [None] * len(self._coordinates)

        logger.info("Generating atom ids")

        for smiles, coords, atoms in tqdm(
            zip(self._smiles, self._coordinates, unimol_atoms),
            total=len(self._smiles),
        ):
            atom_ids = smiles_to_atom_ids(smiles)

            if len(atom_ids) != len(coords):
                assert atoms is not None
                atom_ids = atom_to_atom_ids(atoms)
                assert len(atom_ids) == len(coords)

            self._atoms.append(atom_ids)

        return self

    def get_instances(self) -> list[dict]:
        """
        Retrieve data instances from the dataset.

        Returns
        -------
        list
            List of data instances.
        """
        data_instances = pack_instances(
            atoms=self._atoms,
            coords=self._coordinates,
            lbs=self.lbs,
            masks=self.masks,
        )

        return data_instances
