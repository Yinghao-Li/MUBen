"""
# Author: Yinghao Li
# Modified: August 5th, 2023
# ---------------------------------------
# Description: Trainer function for TorchMD-Net.
"""


import torch
import logging

from .dataset import Collator
from .model import TorchMDNET
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
        ckpt = torch.load(self.config.checkpoint_path, map_location="cpu")

        self._model = TorchMDNET(self.config).load_from_checkpoint(ckpt)
