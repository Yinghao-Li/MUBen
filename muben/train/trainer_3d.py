"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: Trainer function for TorchMD-Net.
"""

import torch
import logging

from ..torchmdnet.model import TorchMDNET
from ..torchmdnet.args import Config

from .trainer import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class for the TorchMD-Net model.
    """

    def initialize_model(self, *args, **kwargs) -> None:
        """
        Initialize the TorchMD-Net model from a checkpoint.

        The model is loaded onto the CPU and its weights are initialized from a checkpoint.
        """
        ckpt = torch.load(self.config.checkpoint_path, map_location="cpu")
        self._model = TorchMDNET(self.config).load_from_checkpoint(ckpt)
