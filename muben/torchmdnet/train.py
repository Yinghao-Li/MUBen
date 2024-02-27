"""
# Author: Yinghao Li
# Modified: November 29th, 2023
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
    """
    Trainer class for the TorchMD-Net model.
    """

    def __init__(
        self,
        config: Config,
        training_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        collate_fn=None,
        **kwargs,
    ) -> None:
        """
        Initialize the Trainer class.

        Parameters
        ----------
        config : Config
            Configuration object with parameters for the Trainer.
        training_dataset : torch.utils.data.Dataset, optional
            Dataset for training.
        valid_dataset : torch.utils.data.Dataset, optional
            Dataset for validation.
        test_dataset : torch.utils.data.Dataset, optional
            Dataset for testing.
        collate_fn : callable, optional
            A function that collates data samples into batches, defaults to Collator with the provided config.

        """
        collate_fn = collate_fn if collate_fn is not None else Collator(config)

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
        Get the config object for the Trainer.

        Returns
        -------
        Config
            Configuration object with parameters for the Trainer.
        """
        return self._config

    def initialize_model(self, *args, **kwargs) -> None:
        """
        Initialize the TorchMD-Net model from a checkpoint.

        The model is loaded onto the CPU and its weights are initialized from a checkpoint.
        """
        ckpt = torch.load(self.config.checkpoint_path, map_location="cpu")
        self._model = TorchMDNET(self.config).load_from_checkpoint(ckpt)
