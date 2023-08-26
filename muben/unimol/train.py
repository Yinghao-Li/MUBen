"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: Trainer function for Uni-Mol.
"""


from abc import ABC

import logging

import torch
import numpy as np
from torch.optim import AdamW

from .dataset import Collator, Dictionary
from .model import UniMol
from .args import Config
from muben.utils.macro import UncertaintyMethods
from muben.base.train import Trainer as BaseTrainer
from muben.base.uncertainty.sgld import SGLDOptimizer, PSGLDOptimizer
from muben.base.uncertainty.ts import TSModel

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    """
    Trainer class responsible for training and managing the Uni-Mol model.
    """

    def __init__(
        self,
        config,
        training_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        collate_fn=None,
        dictionary=None,
    ):
        """
        Initialize the Trainer.

        Parameters
        ----------
        config : Config
            Configuration object containing various parameters.
        training_dataset : Dataset, optional
            Dataset for training.
        valid_dataset : Dataset, optional
            Dataset for validation.
        test_dataset : Dataset, optional
            Dataset for testing.
        collate_fn : callable, optional
            Function to collate data samples into batches.
        dictionary : Dictionary, optional
            Dictionary containing symbols used in the training data.
        """
        # this should be before super.__init__ as we require self.dictionary to initialize model
        if dictionary is None:
            self.dictionary = Dictionary.load()
            self.dictionary.add_symbol("[MASK]", is_special=True)
        else:
            self.dictionary = dictionary

        if not collate_fn:
            collate_fn = Collator(config, atom_pad_idx=self.dictionary.pad())

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

    def initialize_model(self):
        """
        Load the UniMol model from a checkpoint.
        """
        self._model = UniMol(self.config, self.dictionary)

        state = load_checkpoint_to_cpu(self.config.checkpoint_path)
        model_loading_info = self._model.load_state_dict(
            state["model"], strict=False
        )
        logger.info(model_loading_info)
        return self

    def initialize_optimizer(self):
        """
        Initialize the optimizer for the model.

        Depending on the uncertainty estimation method, it initializes
        either a standard AdamW optimizer or an SGLD optimizer.
        """
        # Original implementation seems set weight decay to 0, which is weird.
        # We'll keep it as default here
        params = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = AdamW(
            params, lr=self._status.lr, betas=(0.9, 0.99), eps=1e-6
        )

        # for sgld compatibility
        if self.config.uncertainty_method == UncertaintyMethods.sgld:
            output_param_ids = [
                id(x[1])
                for x in self._model.named_parameters()
                if "output_layer" in x[0]
            ]
            backbone_params = filter(
                lambda p: id(p) not in output_param_ids,
                self._model.parameters(),
            )
            output_params = filter(
                lambda p: id(p) in output_param_ids, self._model.parameters()
            )

            self._optimizer = AdamW(
                backbone_params,
                lr=self._status.lr,
                betas=(0.9, 0.99),
                eps=1e-6,
            )
            sgld_optimizer = (
                PSGLDOptimizer
                if self.config.apply_preconditioned_sgld
                else SGLDOptimizer
            )
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

            preds = preds.reshape(
                (-1, self.config.n_conformation, *pred_instance_shape)
            )
            return preds.mean(axis=1)

        elif isinstance(preds, tuple):
            pred_instance_shape = preds[0].shape[1:]

            # this could be improved for deep ensembles
            return tuple(
                [
                    p.reshape(
                        (-1, self.config.n_conformation, *pred_instance_shape)
                    ).mean(axis=1)
                    for p in preds
                ]
            )

        else:
            raise TypeError(f"Unsupported prediction type {type(preds)}")


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """
    Load a checkpoint to CPU.

    If present, the function also applies overrides to arguments present
    in the checkpoint.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    arg_overrides : dict, optional
        Dictionary of arguments to be overridden in the loaded state.

    Returns
    -------
    dict
        Loaded state dictionary.
    """
    local_path = path
    with open(local_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))

    if (
        "args" in state
        and state["args"] is not None
        and arg_overrides is not None
    ):
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

    return state
