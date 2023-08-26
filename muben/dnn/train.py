"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: Trainer for DNN.
"""


import logging

from .dataset import Collator
from .model import DNN
from .args import Config

from muben.base.train import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class for the Deep Neural Network (DNN).

    This class facilitates the training, validation, and testing of the DNN model.
    It extends the functionality provided by the BaseTrainer from the muben library.
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
        Initialize the Trainer instance.

        Parameters
        ----------
        config : Config
            Configuration parameters for the trainer.
        training_dataset : Dataset, optional
            Dataset to be used for training.
        valid_dataset : Dataset, optional
            Dataset to be used for validation.
        test_dataset : Dataset, optional
            Dataset to be used for testing.
        collate_fn : callable, optional
            Function to collate data into batches. Defaults to the Collator initialized with config.
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
        Get the configuration for the trainer.

        Returns
        -------
        Config
            Configuration parameters for the trainer.
        """
        return self._config

    def initialize_model(self, *args, **kwargs):
        """
        Initialize the DNN model using configurations.

        Any additional arguments or keyword arguments are passed directly to the DNN constructor.
        """
        self._model = DNN(
            d_feature=self.config.d_feature,
            n_lbs=self.config.n_lbs,
            n_tasks=self.config.n_tasks,
            n_hidden_layers=self.config.n_dnn_hidden_layers,
            d_hidden=self.config.d_dnn_hidden,
            p_dropout=self.config.dropout,
            uncertainty_method=self.config.uncertainty_method,
            task_type=self.config.task_type,
            bbp_prior_sigma=self.config.bbp_prior_sigma,
        )
