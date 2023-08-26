"""
# Author: Yinghao Li
# Modified: August 26th, 2023
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
    """
    Dataset tailored for the Deep Neural Network (DNN) model.

    This dataset allows for creating features using different methods like RDKit normalized
    or Morgan binary features. After the feature creation, instances are packed
    and made ready for the training and evaluation of the DNN model.
    """

    def __init__(self):
        """
        Initialize the DNN dataset.

        Currently, initializes the internal `_features` storage to None.
        """
        super().__init__()

        self._features = None

    @property
    def features(self):
        """
        Get the features stored in the dataset.

        Returns
        -------
        np.array
            Array containing the features for each data instance.
        """
        return self._features

    # noinspection PyTypeChecker
    def create_features(self, config):
        """
        Create data features based on the provided configuration.

        Depending on the `feature_type` in the configuration, this method
        generates features using RDKit normalized or Morgan binary methods.

        Parameters
        ----------
        config : object
            Configuration object containing necessary parameters for feature generation,
            including `feature_type` and `num_preprocess_workers`.

        Returns
        -------
        Dataset
            Self, to allow for method chaining.
        """
        feature_type = config.feature_type
        if feature_type == "rdkit":
            logger.info("Generating normalized RDKit features")
            with get_context("fork").Pool(
                config.num_preprocess_workers
            ) as pool:
                self._features = [
                    f
                    for f in tqdm(
                        pool.imap(
                            rdkit_2d_features_normalized_generator,
                            self._smiles,
                        ),
                        total=len(self._smiles),
                    )
                ]
        elif feature_type == "morgan":
            logger.info("Generating Morgan binary features")
            with get_context("fork").Pool(
                config.num_preprocess_workers
            ) as pool:
                self._features = [
                    f
                    for f in tqdm(
                        pool.imap(
                            morgan_binary_features_generator, self._smiles
                        ),
                        total=len(self._smiles),
                    )
                ]
        else:
            self._features = np.empty(len(self)) * np.nan
        return self

    def get_instances(self):
        """
        Get packed data instances for the model.

        Packs the features, labels, and masks and returns them as a list of instances.

        Returns
        -------
        list
            List of packed data instances.
        """
        data_instances = pack_instances(
            features=self._features, lbs=self.lbs, masks=self.masks
        )

        return data_instances
