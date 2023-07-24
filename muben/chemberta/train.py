from abc import ABC

import logging

from muben.utils.macro import UncertaintyMethods
from muben.base.train import Trainer as BaseTrainer
from .dataset import Collator
from .model import ChemBERTa
from .args import Config

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    def __init__(self,
                 config,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 collate_fn=None):

        if not collate_fn:
            collate_fn = Collator(config)

        super().__init__(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn
        )

    @property
    def config(self) -> Config:
        return self._config

    def initialize_model(self):
        self._model = ChemBERTa(
            bert_model_name_or_path=self.config.pretrained_model_name_or_path,
            n_lbs=self.config.n_lbs,
            n_tasks=self.config.n_tasks,
            uncertainty_method=self.config.uncertainty_method,
            task_type=self.config.task_type,
            bbp_prior_sigma=self.config.bbp_prior_sigma
        )
