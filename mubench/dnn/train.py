"""
Yinghao Li @ Georgia Tech

Base trainer function.
"""

import logging

from .dataset import Collator
from .model import DNN
from .args import Config

from mubench.base.train import Trainer as BaseTrainer
from mubench.utils.macro import UncertaintyMethods

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self,
                 config: Config,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 collate_fn=None):

        collate_fn = collate_fn if collate_fn is not None else Collator(config)

        super().__init__(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn,
        )

    def initialize_model(self, *args, **kwargs):
        self._model = DNN(
            d_feature=self._config.d_feature,
            n_lbs=self._config.n_lbs,
            n_tasks=self._config.n_tasks,
            n_hidden_layers=self._config.n_dnn_hidden_layers,
            d_hidden=self._config.d_dnn_hidden,
            p_dropout=self._config.dropout,
            apply_bbp=self._config.uncertainty_method == UncertaintyMethods.bbp,
            bbp_prior_sigma=self._config.bbp_prior_sigma
        )
