import os
import logging
import pandas as pd
import numpy as np

from typing import List, Union
from ast import literal_eval
from tqdm.auto import tqdm
from functools import cached_property
from seqlbtoolkit.training.dataset import (
    BaseDataset,
    DataInstance,
    feature_lists_to_instance_list,
)
from .features import (
    rdkit_2d_features_normalized_generator,
    morgan_binary_features_generator
)

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._features = None
        self._smiles: Union[List[str], None] = None
        self._lbs: Union[np.ndarray, None] = None
        self._masks: Union[np.ndarray, None] = None
    
    @property
    def features(self):
        return self._features

    @property
    def smiles(self) -> List[str]:
        return self._smiles
    
    @property
    def lbs(self) -> np.ndarray:
        return self._lbs

    @cached_property
    def masks(self) -> np.ndarray:
        return self._masks if self._masks is not None else np.ones_like(self.lbs).astype(int)

    def __len__(self):
        return len(self._smiles)

    def update_lbs(self, lbs):
        """
        Update dataset labels and instance list accordingly
        """
        self._lbs = lbs
        self.data_instances = self.get_instances()
        return self

    def prepare(self, config, partition, **kwargs):
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

        method_identifier = f"{config.model_name}-{config.feature_type}" \
            if config.feature_type != 'none' else config.model_name
        preprocessed_path = os.path.normpath(os.path.join(
            config.data_dir, "processed", method_identifier, f"{partition}.pt"
        ))
        # Load Pre-processed dataset if exist
        if os.path.exists(preprocessed_path) and not config.ignore_preprocessed_dataset:
            logger.info(f"Loading pre-processed dataset {preprocessed_path}")
            self.load(preprocessed_path)
        # else, load dataset from csv and generate features
        else:
            file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.csv"))
            logger.info(f"Loading dataset {file_path}")

            if file_path and os.path.exists(file_path):
                self.read_csv(file_path)
            else:
                raise FileNotFoundError(f"File {file_path} does not exist!")

            logger.info("Creating features")
            self.create_features(config)

            # Always save pre-processed dataset to disk
            logger.info("Saving pre-processed dataset")
            self.save(preprocessed_path)

        self.data_instances = self.get_instances()
        return self

    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """
        feature_type = config.feature_type
        if feature_type == 'rdkit':
            logger.info("Generating normalized RDKit features")
            self._features = np.stack([rdkit_2d_features_normalized_generator(smiles) for smiles in tqdm(self._smiles)])
        elif feature_type == 'morgan':
            logger.info("Generating Morgan binary features")
            self._features = np.stack(
                [morgan_binary_features_generator(smiles) for smiles in tqdm(self._smiles)])
        else:
            self._features = np.empty(len(self)) * np.nan
        return self

    def get_instances(self):

        data_instances = feature_lists_to_instance_list(
            DataInstance,
            features=self._features, lbs=self.lbs, masks=self.masks
        )

        return data_instances

    def read_csv(self, file_path: str):
        """
        Load data
        """

        df = pd.read_csv(file_path)
        self._smiles = df.smiles.tolist()
        self._lbs = np.asarray(df.labels.map(literal_eval).to_list())
        self._masks = np.asarray(df.masks.map(literal_eval).to_list()) if not df.masks.isnull().all() else None

        return self
