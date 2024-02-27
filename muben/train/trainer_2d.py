"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: Trainer class for the GIN (Graph Isomorphism Network) model.
"""

import logging

from ..gin.model import GIN
from ..gin.args import Config

from muben.base.train import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer for the GIN model.

    This class initializes and manages the training, validation, and testing of the GIN model.
    """

    def initialize_model(self, *args, **kwargs):
        """
        Initialize the GIN model based on the given configuration.

        Additional arguments and keyword arguments are ignored.
        """
        self._model = self._model_class(
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
