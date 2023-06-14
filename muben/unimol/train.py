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
    def __init__(self,
                 config: Config,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 collate_fn=None,
                 dictionary=None):

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
            collate_fn=collate_fn
        )

    def initialize_model(self):
        self._model = UniMol(self._config, self.dictionary)

        state = load_checkpoint_to_cpu(self._config.checkpoint_path)
        model_loading_info = self._model.load_state_dict(state['model'], strict=False)
        logger.info(model_loading_info)
        return self

    def initialize_optimizer(self):
        # Original implementation seems set weight decay to 0, which is weird.
        # We'll keep it as default here
        params = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = AdamW(params, lr=self._status.lr, betas=(0.9, 0.99), eps=1E-6)

        # for sgld compatibility
        if self._config.uncertainty_method == UncertaintyMethods.sgld:
            output_param_ids = [id(x[1]) for x in self._model.named_parameters() if "output_layer" in x[0]]
            base_params = filter(lambda p: id(p) not in output_param_ids, self._model.parameters())
            output_params = filter(lambda p: id(p) in output_param_ids, self._model.parameters())

            self._optimizer = AdamW(base_params, lr=self._status.lr, betas=(0.9, 0.99), eps=1E-6)
            sgld_optimizer = PSGLDOptimizer if self._config.apply_preconditioned_sgld else SGLDOptimizer
            self._sgld_optimizer = sgld_optimizer(
                output_params, lr=self._status.lr, norm_sigma=self._config.sgld_prior_sigma
            )

        return None

    def ts_session(self):
        # update hyper parameters
        self._status.lr = self._config.ts_lr
        self._status.lr_scheduler_type = 'constant'
        self._status.n_epochs = self._config.n_ts_epochs
        self._status.valid_epoch_interval = 0  # Can also set this to None; disable validation

        self.model.to(self._device)
        self.freeze()
        self._ts_model = TSModel(self._model, self._config.n_tasks)

        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss(disable_focal_loss=True)

        logger.info("Training model on validation")
        self._valid_dataset.set_processor_variant('training')
        self.train(use_valid_dataset=True)
        self._valid_dataset.set_processor_variant('inference')

        self.unfreeze()
        return self

    def process_logits(self, logits: np.ndarray):

        preds = super().process_logits(logits)

        if isinstance(preds, np.ndarray):
            pred_instance_shape = preds.shape[1:]

            preds = preds.reshape((-1, self._config.n_conformation, *pred_instance_shape))
            preds = preds.mean(axis=1)

        elif isinstance(preds, tuple):
            pred_instance_shape = preds[0].shape[1:]

            # this could be improved for deep ensembles
            preds = tuple([p.reshape(
                (-1, self._config.n_conformation, *pred_instance_shape)
            ).mean(axis=1) for p in preds])

        else:
            raise TypeError(f"Unsupported prediction type {type(preds)}")

        return preds


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).
    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """
    local_path = path
    with open(local_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))

    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

    return state
