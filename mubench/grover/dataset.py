import os
import logging

from seqlbtoolkit.training.dataset import (
    DataInstance,
    feature_lists_to_instance_list,
)
from ..base.dataset import Dataset as BaseDataset
from .data.molgraph import MolGraph

logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    def __init__(self):
        super().__init__()

        self._molecule_graph = None

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
            config.data_dir, "processed", config.model_name, f"{partition}.pt"
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

            self.create_features(bond_drop_rate=config.bond_drop_rate)

            # Always save pre-processed dataset to disk
            self.save(preprocessed_path)

        self.data_instances = feature_lists_to_instance_list(
            DataInstance,
            molecule_graphs=self._molecule_graph, lbs=self.lbs, masks=self.masks
        )

        return self

    def create_features(self, bond_drop_rate):
        """
        Create data features

        Returns
        -------
        self
        """

        self._molecule_graph = [MolGraph(smiles, bond_drop_rate) for smiles in self._smiles]
