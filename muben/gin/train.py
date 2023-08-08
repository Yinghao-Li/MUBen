"""
# Author: Yinghao Li
# Modified: August 8th, 2023
# ---------------------------------------
# Description: Trainer function for GIN.
"""


import logging

from .dataset import Collator
from .model import GIN
from .args import Config

from muben.base.train import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self,
                 config,
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

    @property
    def config(self) -> Config:
        return self._config

    def initialize_model(self, *args, **kwargs):

        self._model = GIN(
            n_lbs=self.config.n_lbs,
            n_tasks=self.config.n_tasks,
            max_atomic_num=self.config.max_atomic_num,
            n_layers=self.config.n_gin_layers,
            d_hidden=self.config.d_gin_hidden,
            dropout=self.config.dropout,
            uncertainty_method=self.config.uncertainty_method,
            task_type=self.config.task_type,
            bbp_prior_sigma=self.config.bbp_prior_sigma
        )
