from abc import ABC

import logging

from mubench.utils.macro import UncertaintyMethods
from mubench.base.train import Trainer as BaseTrainer
from .dataset import Collator
from .model import ChemBERTa
from .args import Config

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    def __init__(self,
                 config: Config,
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

    def initialize_model(self):
        self._model = ChemBERTa(
            bert_model_name_or_path=self._config.pretrained_model_name_or_path,
            n_lbs=self._config.n_lbs,
            n_tasks=self._config.n_tasks,
            apply_bbp=self._config.uncertainty_method == UncertaintyMethods.bbp
        )
