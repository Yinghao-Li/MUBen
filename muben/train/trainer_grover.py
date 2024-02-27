"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: Trainer for GROVER backbone
"""

from abc import ABC

import torch
import logging
import numpy as np
import torch.nn.functional as F
from scipy.special import expit

from ..args.args_grover import Config
from muben.model import load_grover_checkpoint as load_checkpoint
from muben.model import GROVERTSModel as TSModel
from muben.utils.macro import UncertaintyMethods
from .trainer import Trainer as BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    """
    Trainer tailored for the GROVER model.

    This class provides mechanisms for initializing, training, and inferring using the GROVER model.
    """

    def __init__(
        self,
        config: Config,
        training_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        collate_fn=None,
        **kwargs,
    ):
        """
        Initialize the GROVER Trainer.

        Parameters
        ----------
        config : Config
            Configuration object containing necessary hyperparameters and settings.
        training_dataset : Dataset, optional
            Dataset used for training.
        valid_dataset : Dataset, optional
            Dataset used for validation.
        test_dataset : Dataset, optional
            Dataset used for testing.
        collate_fn : Collator, optional
            Function to collate data. Default is None.

        """
        if not collate_fn:
            collate_fn = Collator(config)

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
        """Get the config object."""
        return self._config

    def initialize_model(self):
        """
        Load GROVER model from checkpoint.
        """
        logger.info(f"Loading GROVER checkpoint from {self.config.checkpoint_path}")
        self._model = load_checkpoint(self.config)

    def ts_session(self):
        """
        Reload the temperature scaling session as we are using a different
        TS model implementation.
        """
        # update hyper parameters
        self._status.lr = self.config.ts_lr
        self._status.lr_scheduler_type = "constant"
        self._status.n_epochs = self.config.n_ts_epochs
        self._status.valid_epoch_interval = 0  # Can also set this to None; disable validation

        self.model.to(self._device)
        self.freeze()
        # notice that here TS model is different from the base one
        self._ts_model = TSModel(self._model, self.config.n_tasks)

        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss(disable_focal_loss=True)

        logger.info("Training model on validation")
        self.train(use_valid_dataset=True)
        self.unfreeze()
        return self

    def get_loss(self, logits, batch, n_steps_per_epoch=None) -> torch.Tensor:
        """
        Calculate the combined loss for atom and bond logits.

        Parameters
        ----------
        logits : tuple
            Tuple containing atom and bond logits.
        batch : Batch
            Batch of data samples.
        n_steps_per_epoch : int, optional
            Number of steps per epoch. Default is None.

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        assert isinstance(logits, tuple) and len(logits) == 2, ValueError(
            "GROVER should have 2 return values for training!"
        )

        atom_logits, bond_logits = logits

        atom_loss = super().get_loss(atom_logits, batch)
        bond_loss = super().get_loss(bond_logits, batch)
        loss = atom_loss + bond_loss

        if self._ts_model is None:
            dist = self.get_distance_loss(atom_logits, bond_logits, batch)
            loss += self.config.dist_coff * dist

        # for compatability with bbp
        if self.config.uncertainty_method == UncertaintyMethods.bbp and n_steps_per_epoch is not None:
            kld = (self.model.atom_output_layer.kld + self.model.bond_output_layer.kld) / n_steps_per_epoch / len(batch)
            loss += kld
        return loss

    def get_distance_loss(self, atom_logits, bond_logits, batch):
        """
        Compute the distance loss between atom and bond logits.

        Parameters
        ----------
        atom_logits : torch.Tensor
            Logits for atom predictions.
        bond_logits : torch.Tensor
            Logits for bond predictions.
        batch : Batch
            Batch of data samples.

        Returns
        -------
        torch.Tensor
            Computed distance loss value.
        """
        # modify data shapes to accommodate different tasks
        if self.config.task_type == "regression" and self.config.regression_with_variance:
            atom_logits = atom_logits.view(-1, self.config.n_tasks, 2)  # mean and var for the last dimension
            bond_logits = bond_logits.view(-1, self.config.n_tasks, 2)  # mean and var for the last dimension

        loss = F.mse_loss(atom_logits, bond_logits)
        loss = torch.sum(loss * batch.masks) / batch.masks.sum()
        return loss

    def inference(self, dataset, **kwargs):
        """
        Perform inference on a given dataset using the trained GROVER model.
        This function is reloaded to accomodate the special output format of GROVER.

        Parameters
        ----------
        dataset : Dataset
            Dataset on which to perform inference.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Predicted atom logits and bond logits.
        """
        dataloader = self.get_dataloader(dataset, batch_size=self.config.batch_size_inference, shuffle=False)
        self.model.to(self._device)

        atom_logits_list = list()
        bond_logits_list = list()

        with torch.no_grad():
            for batch in dataloader:
                batch.to(self.config.device)
                atom_logits, bond_logits = self.model(batch)

                atom_logits_list.append(atom_logits.to(torch.float).detach().cpu())
                bond_logits_list.append(bond_logits.to(torch.float).detach().cpu())

        atom_logits = torch.cat(atom_logits_list, dim=0).numpy()
        bond_logits = torch.cat(bond_logits_list, dim=0).numpy()

        return atom_logits, bond_logits

    def process_logits(self, logits: tuple[np.ndarray, np.ndarray]):
        """
        Process raw logits into prediction probabilities or regression values.
        This function is reloaded to accomodate the special output format of GROVER.

        Parameters
        ----------
        logits : tuple[np.ndarray, np.ndarray]
            Raw logits for atoms and bonds.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Processed predictions. This could be class probabilities, or mean and variance in case of regression.
        """
        atom_logits, bond_logits = logits

        if self.config.task_type == "classification":
            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                atom_logits = atom_logits.reshape((-1, self.config.n_tasks, self.config.n_lbs))
                bond_logits = bond_logits.reshape((-1, self.config.n_tasks, self.config.n_lbs))

                atom_alpha = atom_logits * (atom_logits > 0) + 1
                bond_alpha = bond_logits * (bond_logits > 0) + 1
                atom_probs = atom_alpha / np.sum(atom_alpha, axis=1, keepdims=True)
                bond_probs = bond_alpha / np.sum(bond_alpha, axis=1, keepdims=True)

                return (atom_probs + bond_probs) / 2

            else:
                atom_preds = expit(atom_logits)  # sigmoid function
                bond_preds = expit(bond_logits)  # sigmoid function
                return (atom_preds + bond_preds) / 2

        elif self.config.task_type == "regression":
            logits = (atom_logits + bond_logits) / 2

            if self.config.n_tasks > 1 and len(logits.shape) == 2:
                logits = logits.reshape(-1, self.config.n_tasks, self.config.n_lbs)

            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                gamma, _, alpha, beta = np.split(logits, 4, axis=-1)
                mean = gamma.squeeze(-1)
                var = (beta / (alpha - 1)).squeeze(-1)
                return mean, var

            elif self.config.regression_with_variance:
                mean = logits[..., 0]
                var = F.softplus(torch.from_numpy(logits[..., 1])).numpy()
                return mean, var

            else:
                return logits
        else:
            raise ValueError(f"Unrecognized task type: {self.config.task_type}")
