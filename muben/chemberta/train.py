"""
# Author: Yinghao Li
# Modified: November 29th, 2023
# ---------------------------------------
# Description: ChemBERTa trainer.
"""

from abc import ABC

import logging

from muben.base.train import Trainer as BaseTrainer
from .dataset import Collator
from .model import ChemBERTa
from .args import Config

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    """
    Trainer class for the ChemBERTa model.

    Inherits from the BaseTrainer and provides specialized functionalities
    for training the ChemBERTa model with chemical data.
    """

    def __init__(
        self,
        config,
        training_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        collate_fn=None,
        **kwargs,
    ):
        """
        Initializes the ChemBERTa Trainer.

        Args:
            config (Config): Configuration settings for the trainer.
            training_dataset (Optional[Union[str, list]]): Dataset for training. Default is None.
            valid_dataset (Optional[Union[str, list]]): Dataset for validation. Default is None.
            test_dataset (Optional[Union[str, list]]): Dataset for testing. Default is None.
            collate_fn (Optional[Callable]): Collation function to process batches of data. If not provided,
                                            a default Collator based on the config is used.

        """
        if not collate_fn:
            collate_fn = Collator(config)

        super().__init__(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn,
            **kwargs,
        )

    @property
    def config(self) -> Config:
        """
        Returns the config object used by the trainer.
        """
        return self._config

    def initialize_model(self):
        """
        Initializes the ChemBERTa model with settings from the config.
        """
        self._model = ChemBERTa(
            bert_model_name_or_path=self.config.pretrained_model_name_or_path,
            n_lbs=self.config.n_lbs,
            n_tasks=self.config.n_tasks,
            uncertainty_method=self.config.uncertainty_method,
            task_type=self.config.task_type,
            bbp_prior_sigma=self.config.bbp_prior_sigma,
        )
