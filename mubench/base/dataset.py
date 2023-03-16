import os
import torch
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Union
from ast import literal_eval
from seqlbtoolkit.training.dataset import (
    BaseDataset,
    DataInstance,
    feature_lists_to_instance_list,
)

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._features = None
        self._smiles: List[str] = None
        self._lbs: List[List[Union[int, float]]] = None
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

        file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.pt"))

        if file_path and os.path.exists(file_path):
            self.read_csv(file_path)
        else:
            raise FileNotFoundError(f"File {file_path} does not exist!")

        self.data_instances = feature_lists_to_instance_list(
            DataInstance,
            features=self._features, lbs=self._lbs
        )

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
