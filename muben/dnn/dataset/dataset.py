"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: DNN Dataset.
"""


import logging
import numpy as np

from tqdm.auto import tqdm
from multiprocessing import get_context

from muben.base.dataset import pack_instances, Dataset as BaseDataset
from muben.utils.chem import (
    rdkit_2d_features_normalized_generator,
    morgan_binary_features_generator,
)

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._features = None

    @property
    def features(self):
        return self._features

    # noinspection PyTypeChecker
    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """
        feature_type = config.feature_type
        if feature_type == "rdkit":
            logger.info("Generating normalized RDKit features")
            with get_context("fork").Pool(config.num_preprocess_workers) as pool:
                self._features = [
                    f
                    for f in tqdm(
                        pool.imap(rdkit_2d_features_normalized_generator, self._smiles),
                        total=len(self._smiles),
                    )
                ]
        elif feature_type == "morgan":
            logger.info("Generating Morgan binary features")
            with get_context("fork").Pool(config.num_preprocess_workers) as pool:
                self._features = [
                    f
                    for f in tqdm(
                        pool.imap(morgan_binary_features_generator, self._smiles),
                        total=len(self._smiles),
                    )
                ]
        else:
            self._features = np.empty(len(self)) * np.nan
        return self

    def get_instances(self):
        data_instances = pack_instances(
            features=self._features, lbs=self.lbs, masks=self.masks
        )

        return data_instances
