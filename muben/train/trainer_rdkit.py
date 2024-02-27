"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: Trainer for DNN.
"""

import logging

from ..dnn.model import DNN
from ..dnn.args import Config

from muben.base.train import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class for the Deep Neural Network (DNN).

    This class facilitates the training, validation, and testing of the DNN model.
    It extends the functionality provided by the BaseTrainer from the muben library.
    """

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
