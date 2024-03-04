"""
# Author: Yinghao Li
# Modified: March 4th, 2024
# ---------------------------------------
# Description: Trainer function for Uni-Mol.
"""

from abc import ABC

import logging

import numpy as np
from torch.optim import AdamW

from .trainer import Trainer as BaseTrainer
from muben.utils.macro import UncertaintyMethods
from muben.uncertainty.sgld import SGLDOptimizer, PSGLDOptimizer
from muben.uncertainty.ts import TSModel

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    """
    Trainer class responsible for training and managing the Uni-Mol model.
    """

    def initialize_optimizer(self, *args, **kwargs):
        """
        Initialize the optimizer for the model.

        Depending on the uncertainty estimation method, it initializes
        either a standard AdamW optimizer or an SGLD optimizer.
        """
        # Original implementation seems set weight decay to 0, which is weird.
        # We'll keep it as default here
        params = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = AdamW(params, lr=self._status.lr, betas=(0.9, 0.99), eps=1e-6)

        # for sgld compatibility
        if self.config.uncertainty_method == UncertaintyMethods.sgld:
            output_param_ids = [id(x[1]) for x in self._model.named_parameters() if "output_layer" in x[0]]
            backbone_params = filter(
                lambda p: id(p) not in output_param_ids,
                self._model.parameters(),
            )
            output_params = filter(lambda p: id(p) in output_param_ids, self._model.parameters())

            self._optimizer = AdamW(
                backbone_params,
                lr=self._status.lr,
                betas=(0.9, 0.99),
                eps=1e-6,
            )
            sgld_optimizer = PSGLDOptimizer if self.config.apply_preconditioned_sgld else SGLDOptimizer
            self._sgld_optimizer = sgld_optimizer(
                output_params,
                lr=self._status.lr,
                norm_sigma=self.config.sgld_prior_sigma,
            )

        return None

    def ts_session(self):
        """
        Reload Temperature Scaling training as the dataset is processed
        differently for training and test in Uni-Mol.
        """
        # update hyper parameters
        self._status.lr = self.config.ts_lr
        self._status.lr_scheduler_type = "constant"
        self._status.n_epochs = self.config.n_ts_epochs
        # Can also set this to None; disable validation
        self._status.valid_epoch_interval = 0

        self.model.to(self._device)
        self.freeze()
        self._ts_model = TSModel(self._model, self.config.n_tasks)

        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss(disable_focal_loss=True)

        logger.info("Training model on validation")
        self._valid_dataset.set_processor_variant("training")
        self.train(use_valid_dataset=True)
        self._valid_dataset.set_processor_variant("inference")

        self.unfreeze()
        return self

    def process_logits(self, logits: np.ndarray):
        """
        Add the conformation aggregation to the parent method.
        """
        preds = super().process_logits(logits)

        if isinstance(preds, np.ndarray):
            pred_instance_shape = preds.shape[1:]

            preds = preds.reshape((-1, self.config.n_conformation, *pred_instance_shape))
            return preds.mean(axis=1)

        elif isinstance(preds, tuple):
            pred_instance_shape = preds[0].shape[1:]

            # this could be improved for deep ensembles
            return tuple(
                [p.reshape((-1, self.config.n_conformation, *pred_instance_shape)).mean(axis=1) for p in preds]
            )

        else:
            raise TypeError(f"Unsupported prediction type {type(preds)}")

    def test_on_training_data(self, load_best_model=True):
        """
        Reload the runction as the dataset is processed differently
        """
        self._training_dataset.set_processor_variant("inference")
        metrics = super().test_on_training_data(load_best_model)
        self._training_dataset.set_processor_variant("training")
        return metrics
