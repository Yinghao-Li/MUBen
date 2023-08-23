"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: GROVER arguments.
"""

import logging
from typing import Optional
from dataclasses import field
from dataclasses import dataclass

from muben.base.args import Config as BaseConfig, Arguments as BaseArguments
from muben.utils.macro import MODEL_NAMES

logger = logging.getLogger(__name__)


@dataclass
class Arguments(BaseArguments):
    """
    Grover fine-tuning/prediction arguments
    """

    # --- Reload model arguments to adjust default values ---
    model_name: Optional[str] = field(
        default="GROVER", metadata={"help": "Name of the model", "choices": MODEL_NAMES}
    )

    checkpoint_path: Optional[str] = field(
        default="./models/grover_base.pt",
        metadata={"help": "Path to model checkpoint (.pt file)"},
    )

    seed: Optional[int] = field(
        default=0,
        metadata={
            "help": "Random seed to use when splitting data into train/val/test sets."
            "When `num_folds` > 1, the first fold uses this seed and all"
            "subsequent folds add 1 to the seed."
        },
    )

    # Training arguments
    lr_scheduler_type: Optional[str] = field(
        default="linear",
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

    # Model arguments
    dropout: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout probability"}
    )
    ffn_hidden_size: Optional[int] = field(
        default=200,
        metadata={
            "help": "Hidden dim for higher-capacity FFN (defaults to hidden_size)"
        },
    )
    ffn_num_layers: Optional[int] = field(
        default=2, metadata={"help": "Number of layers in FFN after MPN encoding"}
    )

    dist_coff: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dist coefficient for output of two branches."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.disable_dataset_saving = True


@dataclass
class Config(Arguments, BaseConfig):
    # activation function, will be overwritten during model loading
    activation = "PReLU"
    # weight decay, will be overwritten during model loading
    weight_decay = 1e-7
    # Drop out bond in molecules; notice that this argument is no longer used.
    bond_drop_rate = 0
    # Enable distinct fine tune learning rate for fc and other layer
    fine_tune_coff = 1
