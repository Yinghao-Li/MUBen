from abc import ABC

import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from scipy.special import expit
from typing import Optional, Tuple

from .dataset import Collator
from .args import Config
from .model import load_checkpoint
from .uncertainty.ts import TSModel
from mubench.utils.macro import UncertaintyMethods
from mubench.base.train import Trainer as BaseTrainer
from mubench.base.uncertainty.sgld import SGLDOptimizer, PSGLDOptimizer

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
        logger.info(f"Loading GROVER checkpoint from {self._config.checkpoint_path}")
        self._model = load_checkpoint(self._config)

    def initialize_optimizer(self):
        """
        Builds an Optimizer, copied from GROVER original implementation.
        """

        # Only adjust the learning rate for the GroverFinetuneTask.
        ffn_param_ids = [id(x[1]) for x in self._model.named_parameters()
                         if "grover" not in x[0] and ("ffn" in x or "output_layer" in x[0])]
        base_params = filter(lambda p: id(p) not in ffn_param_ids, self._model.parameters())
        output_params = filter(lambda p: id(p) in ffn_param_ids, self._model.parameters())
        if self._config.fine_tune_coff == 0:
            for param in base_params:
                param.requires_grad = False

        # for sgld compatibility
        if self._config.uncertainty_method != UncertaintyMethods.sgld:
            self._optimizer = AdamW([
                {'params': base_params, 'lr': self._status.lr * self._config.fine_tune_coff},
                {'params': output_params, 'lr': self._status.lr}
            ], lr=self._status.lr, weight_decay=self._config.weight_decay)
        else:
            self._optimizer = AdamW(base_params, lr=self._status.lr, weight_decay=self._config.weight_decay)
            sgld_optimizer = PSGLDOptimizer if self._config.apply_preconditioned_sgld else SGLDOptimizer
            self._sgld_optimizer = sgld_optimizer(
                output_params, lr=self._status.lr, norm_sigma=self._config.sgld_prior_sigma
            )

        return self

    def ts_session(self):
        # update hyper parameters
        self._status.lr = self._config.ts_lr
        self._status.lr_scheduler_type = 'constant'
        self._status.n_epochs = self._config.n_ts_epochs
        self._status.valid_epoch_interval = 0  # Can also set this to None; disable validation

        self.model.to(self._device)
        self.freeze()
        # notice that here TS model is different from the base one
        self._ts_model = TSModel(self._model, self._config.n_tasks)

        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss(disable_focal_loss=True)

        logger.info("Training model on validation")
        self.train(use_valid_dataset=True)
        self.unfreeze()
        return self

    def get_loss(self, logits, batch, n_steps_per_epoch=None) -> torch.Tensor:

        assert isinstance(logits, tuple) and len(logits) == 2, \
            ValueError("GROVER should have 2 return values for training!")

        atom_logits, bond_logits = logits

        atom_loss = super().get_loss(atom_logits, batch)
        bond_loss = super().get_loss(bond_logits, batch)
        loss = atom_loss + bond_loss

        if self._ts_model is None:
            dist = self.get_distance_loss(atom_logits, bond_logits, batch)
            loss += self._config.dist_coff * dist

        # for compatability with bbp
        if self._config.uncertainty_method == UncertaintyMethods.bbp and n_steps_per_epoch is not None:
            kld = (self.model.atom_output_layer.kld + self.model.bond_output_layer.kld) / n_steps_per_epoch / len(batch)
            loss += kld
        return loss

    def get_distance_loss(self, atom_logits, bond_logits, batch):

        # modify data shapes to accommodate different tasks
        if self._config.task_type == 'regression' and self._config.regression_with_variance:
            atom_logits = atom_logits.view(-1, self._config.n_tasks, 2)  # mean and var for the last dimension
            bond_logits = bond_logits.view(-1, self._config.n_tasks, 2)  # mean and var for the last dimension

        loss = F.mse_loss(atom_logits, bond_logits)
        loss = torch.sum(loss * batch.masks) / batch.masks.sum()
        return loss

    def inference(self, dataset, batch_size: Optional[int] = None):

        dataloader = self.get_dataloader(
            dataset,
            batch_size=batch_size if batch_size else self._config.batch_size,
            shuffle=False
        )
        self.model.to(self._device)
        self.eval_mode()

        atom_logits_list = list()
        bond_logits_list = list()

        with torch.no_grad():
            for batch in dataloader:
                batch.to(self._config.device)
                atom_logits, bond_logits = self.model(batch)

                atom_logits_list.append(atom_logits.to(torch.float).detach().cpu())
                bond_logits_list.append(bond_logits.to(torch.float).detach().cpu())

        atom_logits = torch.cat(atom_logits_list, dim=0).numpy()
        bond_logits = torch.cat(bond_logits_list, dim=0).numpy()

        return atom_logits, bond_logits

    def process_logits(self, logits: Tuple[np.ndarray, np.ndarray]):

        atom_logits, bond_logits = logits

        if self._config.task_type == 'classification':

            atom_preds = expit(atom_logits)  # sigmoid function
            bond_preds = expit(bond_logits)  # sigmoid function
            preds = (atom_preds + bond_preds) / 2

        elif self._config.task_type == 'regression':
            logits = (atom_logits + bond_logits) / 2

            if self._config.n_tasks > 1:
                logits = logits.reshape(-1, self._config.n_tasks, 2)

            if self._config.regression_with_variance:
                mean = logits[..., 0]
                var = F.softplus(torch.from_numpy(logits[..., 1])).numpy()
                return mean, var
            else:
                preds = logits
        else:
            raise ValueError(f"Unrecognized task type: {self._config.task_type}")

        return preds
