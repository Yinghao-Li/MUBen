"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: TorchMD-NET arguments.
"""

import os.path as op
import logging
from typing import Optional
from dataclasses import dataclass, field
from muben.args.args import Arguments as BaseArguments, Config as BaseConfig
from muben.utils.macro import MODEL_NAMES

logger = logging.getLogger(__name__)


@dataclass
class Arguments(BaseArguments):

    # --- Reload model arguments to adjust default values ---
    model_name: Optional[str] = field(
        default="TorchMD-NET",
        metadata={"help": "The name of the model to be used.", "choices": MODEL_NAMES},
    )
    unimol_feature_folder: Optional[str] = field(
        default=".",
        metadata={"help": "The folder containing files with pre-defined uni-mol atoms and coordinates"},
    )

    # --- update model parameters ---
    checkpoint_path: Optional[str] = field(
        default="./models/torchmd-net.ckpt",
        metadata={"help": "Path to the pre-trained model"},
    )

    # --- reload training parameters ---
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

    def __post_init__(self):
        super().__post_init__()
        self.unimol_feature_dir = op.join(self.unimol_feature_folder, self.dataset_name)


@dataclass
class Config(Arguments, BaseConfig):
    # Default hyperparameters of the checkpoint, should not be changed
    embedding_dimension = 256
    num_layers = 8
    num_rbf = 64
    rbf_type = "expnorm"
    trainable_rbf = False
    activation = "silu"
    attn_activation = "silu"
    neighbor_embedding = True
    num_heads = 8
    distance_influence = "both"
    cutoff_lower = 0.0
    cutoff_upper = 5.0
    max_z = 100
    max_num_neighbors = 32
    layernorm_on_vec = "whitened"
