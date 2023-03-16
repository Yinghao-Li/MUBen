import os
import logging
import pandas as pd

from typing import List, Union
from ast import literal_eval
from tqdm.auto import tqdm
from seqlbtoolkit.training.dataset import (
    BaseDataset,
    DataInstance,
    feature_lists_to_instance_list,
)
from ..utils.feature_generators import rdkit_2d_features_normalized_generator

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._features = None  # This could be any user-defined data structure.
        self._smiles: Union[List[str], None] = None
        self._lbs: Union[List[List[Union[int, float]]], None] = None
        self._masks: Union[List[List[int]], None] = None
    
    @property
    def features(self):
        return self._features
    
    @property
    def lbs(self):
        return self._lbs

    @property
    def masks(self):
        return self._masks

    def prepare(self, config, partition):
        """
        Prepare dataset for training and test

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test]

        Returns
        -------
        self
        """

        assert partition in ['train', 'valid', 'test'], \
            ValueError(f"Argument `partition` should be one of 'train', 'valid' or 'test'!")

        preprocessed_path = os.path.normpath(os.path.join(
            config.data_dir, "processed", config.model_name, config.uncertainty_method, f"{partition}.pt"
        ))
        # Load Pre-processed dataset if exist
        if os.path.exists(preprocessed_path) and not config.ignore_preprocessed_dataset:
            logger.info(f"Loading dataset {preprocessed_path}")
            self.load(preprocessed_path)
        # else, load dataset from csv and generate features
        else:
            file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.csv"))
            logger.info(f"Loading dataset {file_path}")

            if file_path and os.path.exists(file_path):
                self.read_csv(file_path)
            else:
                raise FileNotFoundError(f"File {file_path} does not exist!")

            self.create_features(feature_type=config.feature_type)

            # Always save pre-processed dataset to disk
            self.save(preprocessed_path)

        self.data_instances = feature_lists_to_instance_list(
            DataInstance,
            features=self._features, smiles=self._smiles, lbs=self._lbs, masks=self._masks
        )

        return self

    def create_features(self, feature_type):
        """
        Create data features

        Returns
        -------
        self
        """
        if feature_type == 'rdkit':
            logger.info("Generating normalized RDKit features")
            self._features = [rdkit_2d_features_normalized_generator(smiles) for smiles in tqdm(self._smiles)]
        else:
            self._features = [None] * len(self._smiles)
        return self

    def read_csv(self, file_path: str):
        """
        Load data
        """

        df = pd.read_csv(file_path)
        self._smiles = df.smiles.tolist()
        self._lbs = df.labels.map(literal_eval)
        self._masks = df.masks.map(literal_eval) if not df.masks.isnull().all() else None

        return self
