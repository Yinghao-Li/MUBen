"""
# Author: Yinghao Li
# Modified: August 4th, 2023
# ---------------------------------------
# Description: TorchMD-NET arguments.
"""


import os.path as op
import logging
from typing import Optional
from dataclasses import dataclass, field
from muben.base.args import (
    Arguments as BaseArguments,
    Config as BaseConfig
)
from muben.utils.macro import MODEL_NAMES


logger = logging.getLogger(__name__)


@dataclass
class Arguments(BaseArguments):
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- Reload model arguments to adjust default values ---
    model_name: Optional[str] = field(
        default='TorchMD-NET', metadata={
            'help': "The name of the model to be used.",
            "choices": MODEL_NAMES
        }
    )
    unimol_feature_folder: Optional[str] = field(
        default='.', metadata={'help': "The folder containing files with pre-defined uni-mol atoms and coordinates"}
    )

    # --- update model parameters ---
    checkpoint_path: Optional[str] = field(
        default='./models/torchmd-net.ckpt', metadata={'help': "Path to the pre-trained model"}
    )

    lr: Optional[float] = field(
        default=0.0004, metadata={'help': ''}
    )

    lr_patience: Optional[int] = field(
        default=15, metadata={'help': ''}
    )

    lr_min: Optional[float] = field(
        default=1e-07, metadata={'help': ''}
    )

    lr_factor: Optional[float] = field(
        default=0.8, metadata={'help': ''}
    )

    weight_decay: Optional[float] = field(
        default=0.0, metadata={'help': ''}
    )

    ema_alpha_y: Optional[float] = field(
        default=1.0, metadata={'help': ''}
    )

    ema_alpha_dy: Optional[float] = field(
        default=1.0, metadata={'help': ''}
    )

    num_nodes: Optional[int] = field(
        default=1, metadata={'help': ''}
    )

    precision: Optional[int] = field(
        default=32, metadata={'help': ''}
    )

    log_dir: Optional[str] = field(
        default='experiments/', metadata={'help': ''}
    )

    distributed_backend: Optional[str] = field(
        default='ddp', metadata={'help': ''}
    )

    redirect: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    job_id: Optional[str] = field(
        default='finetuning', metadata={'help': ''}
    )

    pretrained_model: Optional[str] = field(
        default='./checkpoints/torchmdnet.ckpt', metadata={'help': ''}
    )

    dataset: Optional[str] = field(
        default='QM9', metadata={'help': ''}
    )

    dataset_root: Optional[str] = field(
        default='data/qm9', metadata={'help': ''}
    )

    dataset_arg: Optional[str] = field(
        default='homo', metadata={'help': ''}
    )

    energy_weight: Optional[float] = field(
        default=1.0, metadata={'help': ''}
    )

    force_weight: Optional[float] = field(
        default=1.0, metadata={'help': ''}
    )

    position_noise_scale: Optional[float] = field(
        default=0.005, metadata={'help': ''}
    )

    denoising_weight: Optional[float] = field(
        default=0.1, metadata={'help': ''}
    )

    denoising_only: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    model: Optional[str] = field(
        default='equivariant-transformer', metadata={'help': ''}
    )

    output_model: Optional[str] = field(
        default='Scalar', metadata={'help': ''}
    )

    prior_model: Optional[str] = field(
        default=None, metadata={'help': ''}
    )

    output_model_noise: Optional[str] = field(
        default='VectorOutput', metadata={'help': ''}
    )

    embedding_dimension: Optional[int] = field(
        default=256, metadata={'help': ''}
    )

    num_layers: Optional[int] = field(
        default=8, metadata={'help': ''}
    )

    num_rbf: Optional[int] = field(
        default=64, metadata={'help': ''}
    )

    activation: Optional[str] = field(
        default='silu', metadata={'help': ''}
    )

    rbf_type: Optional[str] = field(
        default='expnorm', metadata={'help': ''}
    )

    trainable_rbf: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    neighbor_embedding: Optional[bool] = field(
        default=True, metadata={'help': ''}
    )

    aggr: Optional[str] = field(
        default='add', metadata={'help': ''}
    )

    distance_influence: Optional[str] = field(
        default='both', metadata={'help': ''}
    )

    attn_activation: Optional[str] = field(
        default='silu', metadata={'help': ''}
    )

    num_heads: Optional[int] = field(
        default=8, metadata={'help': ''}
    )

    layernorm_on_vec: Optional[str] = field(
        default='whitened', metadata={'help': ''}
    )

    derivative: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    cutoff_lower: Optional[float] = field(
        default=0.0, metadata={'help': ''}
    )

    cutoff_upper: Optional[float] = field(
        default=5.0, metadata={'help': ''}
    )

    atom_filter: Optional[int] = field(
        default=-1, metadata={'help': ''}
    )

    max_z: Optional[int] = field(
        default=100, metadata={'help': ''}
    )

    max_num_neighbors: Optional[int] = field(
        default=32, metadata={'help': ''}
    )

    def __post_init__(self):
        super().__post_init__()
        self.unimol_feature_dir = op.join(self.unimol_feature_folder, self.dataset_name)


@dataclass
class Config(Arguments, BaseConfig):
    pass
