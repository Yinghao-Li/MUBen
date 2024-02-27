"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: ChemBERTa trainer.
"""

from abc import ABC

import logging

from ..chemberta.model import ChemBERTa
from .trainer import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    """
    Trainer class for the ChemBERTa model.

    Inherits from the BaseTrainer and provides specialized functionalities
    for training the ChemBERTa model with chemical data.
    """

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
