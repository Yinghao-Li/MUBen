from abc import ABC

import logging
import torch
import numpy as np
from torch.optim import AdamW

from ..base.train import Trainer as BaseTrainer
from .dataset import Collator, Dictionary
from .model import UniMol
from .args import Config

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

        if self.config.only_polar > 0:
            self.config.remove_polar_hydrogen = True
        elif self.config.only_polar < 0:
            self.config.remove_polar_hydrogen = False
        else:
            self.config.remove_hydrogen = True

    def initialize_model(self):
        self._model = UniMol(self.config, self.dictionary)

        state = load_checkpoint_to_cpu(self.config.checkpoint_path)
        model_loading_info = self._model.load_state_dict(state['model'], strict=False)
        logger.info(model_loading_info)
        return self

    def initialize_optimizer(self):
        # Original implementation seems set weight decay to 0, which is weird.
        # We'll keep it as default here
        self._optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr, betas=(0.9, 0.99), eps=1E-6
        )
        return None

    def normalize_logits(self, logits: np.ndarray) -> np.ndarray:

        # TODO: should keep preds of all conformations during result saving

        preds = super().normalize_logits(logits)
        pred_instance_shape = preds.shape[1:]

        preds = preds.reshape((-1, self.config.n_conformation, *pred_instance_shape))
        preds = preds.mean(axis=1)

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
