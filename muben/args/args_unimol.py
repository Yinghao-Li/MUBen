"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: Arguments and configuration for Uni-Mol
"""

import os.path as op
import logging
from typing import Optional
from dataclasses import field, dataclass

from .args import Arguments as BaseArguments, Config as BaseConfig
from muben.utils.macro import MODEL_NAMES

logger = logging.getLogger(__name__)


@dataclass
class Arguments(BaseArguments):
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- Reload model arguments to adjust default values ---
    model_name: Optional[str] = field(
        default="Uni-Mol",
        metadata={
            "help": "The name of the model to be used.",
            "choices": MODEL_NAMES,
        },
    )
    # --- Dataset arguments ---
    unimol_feature_folder: Optional[str] = field(
        default=".",
        metadata={"help": "The folder containing files with pre-defined uni-mol atoms and coordinates"},
    )

    # --- Reload training arguments to adjust default values ---
    lr_scheduler_type: Optional[str] = field(
        default="polynomial",
        metadata={
            "help": "Learning rate scheduler with warm ups defined in `transformers`, Please refer to "
            "https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules for details",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    grad_norm: Optional[float] = field(
        default=1,
        metadata={"help": "Gradient norm. Default is 0 (do not clip gradient)"},
    )

    # --- update model parameters from Uni-Mol ---
    checkpoint_path: Optional[str] = field(
        default="./models/unimol_base.pt",
        metadata={"help": "Path to the pre-trained model"},
    )

    # --- Arguments from Uni-Mol original implementation ---
    batch_size: Optional[int] = field(default=32, metadata={"help": "Batch size"})

    max_atoms: Optional[int] = field(
        default=256,
        metadata={"help": "the maximum number of atoms in the input molecules."},
    )

    max_seq_len: Optional[int] = field(default=512, metadata={"help": "Maximum length of the atom tokens."})

    only_polar: Optional[int] = field(
        default=0,
        metadata={"help": "1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen "},
    )

    dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The `pooler dropout` argument in the original implementation. "
            "Controls the dropout ratio of the classification layers."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.unimol_feature_dir = op.join(self.unimol_feature_folder, self.dataset_name)

        self.pooler_dropout = self.dropout

        # Set flags to indicate whether remove hydrogen or polar hydrogen
        # according to the `only_polar` value
        self.remove_hydrogen = False
        self.remove_polar_hydrogen = False

        if self.only_polar > 0:
            self.remove_polar_hydrogen = True
        elif self.only_polar < 0:
            self.remove_polar_hydrogen = False
        else:
            self.remove_hydrogen = True

        self._assign_default_model_args()

    def _assign_default_model_args(self):
        """
        Default hyper-parameters of the pre-trained Uni-Mol model, should not be changed.
        """
        # Model architecture, should not be changed
        self.encoder_layers = 15
        self.encoder_embed_dim = 512
        self.encoder_ffn_embed_dim = 2048
        self.encoder_attention_heads = 64

        # Fix the dropout ratio to the original implementation
        self.dropout = 0.1
        self.emb_dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.pooler_dropout = getattr(self, "pooler_dropout", 0.0)

        self.max_seq_len = getattr(self, "max_seq_len", 512)

        self.activation_fn = getattr(self, "activation_fn", "gelu")
        self.pooler_activation_fn = getattr(self, "pooler_activation_fn", "Tanh")

        self.post_ln = getattr(self, "post_ln", False)
        self.masked_token_loss = getattr(self, "masked_token_loss", -1.0)
        self.masked_coord_loss = getattr(self, "masked_coord_loss", -1.0)
        self.masked_dist_loss = getattr(self, "masked_dist_loss", -1.0)
        self.x_norm_loss = getattr(self, "x_norm_loss", -1.0)
        self.delta_pair_repr_norm_loss = getattr(self, "delta_pair_repr_norm_loss", -1.0)


@dataclass
class Config(Arguments, BaseConfig):
    """
    Configuration class for Uni-Mol.
    """

    # The number of conformations; default is 11 and should not be changed
    n_conformation = 11
