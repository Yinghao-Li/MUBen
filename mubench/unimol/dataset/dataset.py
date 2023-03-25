import logging
import pandas as pd
from multiprocessing import get_context
from tqdm.auto import tqdm
from typing import Optional

from transformers import AutoTokenizer
from seqlbtoolkit.training.dataset import (
    DataInstance,
    feature_lists_to_instance_list,
)

from .utils import smiles_to_coords
from mubench.base.dataset import Dataset as BaseDataset

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._atoms = None
        self._cooridnates = None
        self._mol = None

    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """

        self._atoms = list()
        self._cooridnates = list()
        self._mol = list()

        with get_context('fork').Pool(config.n_threads) as pool:

            for outputs in tqdm(pool.imap(smiles_to_coords, self._smiles), total=len(self._smiles)):
                atoms, coordinates, mol = outputs
                self._atoms.append(atoms)
                self._cooridnates.append(coordinates)
                self._mol.append(mol)

        return self

    def get_instances(self):

        data_instances = feature_lists_to_instance_list(
            DataInstance,
            atom_ids=self._atom_ids, lbs=self.lbs, masks=self.masks
        )

        return data_instances

