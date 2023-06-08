import os
import regex
import torch
import logging
import pandas as pd
import numpy as np

from typing import List, Union
from ast import literal_eval
from functools import cached_property
from torch.utils.data import Dataset as TorchDataset


logger = logging.getLogger(__name__)

__all__ = ["Dataset"]


class Dataset(TorchDataset):
    def __init__(self):
        super().__init__()

        self._smiles: Union[List[str], None] = None
        self._lbs: Union[np.ndarray, None] = None
        self._masks: Union[np.ndarray, None] = None

        self.data_instances = None
    
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

    def __getitem__(self, idx):
        return self.data_instances[idx]

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
            self.read_csv(config.data_dir, partition)

            logger.info("Creating features")
            self.create_features(config)

            # Always save pre-processed dataset to disk
            if not config.disable_dataset_saving:
                logger.info("Saving pre-processed dataset")
                self.save(preprocessed_path)

        self.data_instances = self.get_instances()
        return self

    # noinspection PyTypeChecker
    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """
        raise NotImplementedError

    def get_instances(self):

        raise NotImplementedError

    def save(self, file_path: str):
        """
        Save the entire dataset for future usage
        Parameters
        ----------
        file_path: path to the saved file
        Returns
        -------
        self
        """
        attr_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                attr_dict[attr] = value

        os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)
        torch.save(attr_dict, file_path)

        return self

    def load(self, file_path: str):
        """
        Load the entire dataset from disk
        Parameters
        ----------
        file_path: path to the saved file
        Returns
        -------
        self
        """
        attr_dict = torch.load(file_path)

        for attr, value in attr_dict.items():
            if attr not in self.__dict__:
                logger.warning(f"Attribute {attr} is not natively defined in dataset!")

            setattr(self, attr, value)

        return self

    def read_csv(self, data_dir: str, partition: str):
        """
        Load data
        """
        file_path = os.path.normpath(os.path.join(data_dir, f"{partition}.csv"))
        logger.info(f"Loading dataset {file_path}")

        if not (file_path and os.path.exists(file_path)):
            raise FileNotFoundError(f"File {file_path} does not exist!")

        df = pd.read_csv(file_path)
        self._smiles = df.smiles.tolist()
        self._lbs = np.asarray(df.labels.map(literal_eval).to_list())
        self._masks = np.asarray(df.masks.map(literal_eval).to_list()) if not df.masks.isnull().all() else None

        return self
