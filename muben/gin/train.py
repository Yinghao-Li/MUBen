"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: Trainer class for the GIN (Graph Isomorphism Network) model.
"""

import logging

from .dataset import Collator
from .model import GIN
from .args import Config

from muben.base.train import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer for the GIN model.

    This class initializes and manages the training, validation, and testing of the GIN model.
    """

    def __init__(
        self,
        config: Config,
        training_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        collate_fn=None,
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        config : Config
            Configuration object containing necessary hyperparameters and settings.
        training_dataset : Dataset, optional
            The dataset to be used for training.
        valid_dataset : Dataset, optional
            The dataset to be used for validation.
        test_dataset : Dataset, optional
            The dataset to be used for testing.
        collate_fn : callable, optional
            Function to collate data into batches. If None, the default collate function will be used.
        """
        collate_fn = collate_fn if collate_fn is not None else Collator(config)

        super().__init__(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn,
        )

    @property
    def config(self) -> Config:
        """
        Get the configuration object.

        Returns
        -------
        Config
            Configuration object containing necessary hyperparameters and settings.
        """
        return self._config

    def initialize_model(self, *args, **kwargs):
        """
        Initialize the GIN model based on the given configuration.

        Additional arguments and keyword arguments are ignored.
        """
        self._model = GIN(
            n_lbs=self.config.n_lbs,
            n_tasks=self.config.n_tasks,
            max_atomic_num=self.config.max_atomic_num,
            n_layers=self.config.n_gin_layers,
            d_hidden=self.config.d_gin_hidden,
            dropout=self.config.dropout,
            uncertainty_method=self.config.uncertainty_method,
            task_type=self.config.task_type,
            bbp_prior_sigma=self.config.bbp_prior_sigma,
        )
