import logging
from typing import Optional
from dataclasses import field, dataclass

from ..base.args import (
    Arguments as BaseArguments,
    Config as BaseConfig
)
from ..utils.macro import MODEL_NAMES

logger = logging.getLogger(__name__)


@dataclass
class Arguments(BaseArguments):
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- Reload model arguments to adjust default values ---
    model_name: Optional[str] = field(
        default='Uni-Mol', metadata={
            'help': "Name of the model",
            "choices": MODEL_NAMES
        }
    )

    # --- Arguments from Uni-Mol original implementation ---
    no_progress_bar: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    log_interval: Optional[int] = field(
        default=50, metadata={'help': ''}
    )

    log_format: Optional[str] = field(
        default='simple', metadata={'help': ''}
    )

    tensorboard_logdir: Optional[str] = field(
        default='', metadata={'help': ''}
    )

    cpu: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    fp16: Optional[bool] = field(
        default=True, metadata={'help': ''}
    )

    bf16: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    bf16_sr: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    allreduce_fp32_grad: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    fp16_no_flatten_grads: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    fp16_init_scale: Optional[int] = field(
        default=4, metadata={'help': ''}
    )

    fp16_scale_window: Optional[int] = field(
        default=256, metadata={'help': ''}
    )

    fp16_scale_tolerance: Optional[float] = field(
        default=0.0, metadata={'help': ''}
    )

    min_loss_scale: Optional[float] = field(
        default=0.0001, metadata={'help': ''}
    )

    threshold_loss_scale: Optional[float] = field(
        default=None, metadata={'help': ''}
    )

    user_dir: Optional[str] = field(
        default='../unimol', metadata={'help': ''}
    )

    empty_cache_freq: Optional[int] = field(
        default=0, metadata={'help': ''}
    )

    all_gather_list_size: Optional[int] = field(
        default=16384, metadata={'help': ''}
    )

    suppress_crashes: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    profile: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    ema_decay: Optional[float] = field(
        default=-1.0, metadata={'help': ''}
    )

    validate_with_ema: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    loss: Optional[str] = field(
        default='finetune_cross_entropy', metadata={'help': ''}
    )

    optimizer: Optional[str] = field(
        default='adam', metadata={'help': ''}
    )

    lr_scheduler: Optional[str] = field(
        default='fixed', metadata={'help': ''}
    )

    task: Optional[str] = field(
        default='mol_finetune', metadata={'help': ''}
    )

    num_workers: Optional[int] = field(
        default=8, metadata={'help': ''}
    )

    skip_invalid_size_inputs_valid_test: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    batch_size: Optional[int] = field(
        default=32, metadata={'help': ''}
    )

    required_batch_size_multiple: Optional[int] = field(
        default=8, metadata={'help': ''}
    )

    data_buffer_size: Optional[int] = field(
        default=10, metadata={'help': ''}
    )

    train_subset: Optional[str] = field(
        default='train', metadata={'help': ''}
    )

    valid_subset: Optional[str] = field(
        default='test', metadata={'help': ''}
    )

    validate_interval: Optional[int] = field(
        default=1, metadata={'help': ''}
    )

    validate_interval_updates: Optional[int] = field(
        default=0, metadata={'help': ''}
    )

    validate_after_updates: Optional[int] = field(
        default=0, metadata={'help': ''}
    )

    fixed_validation_seed: Optional[int] = field(
        default=None, metadata={'help': ''}
    )

    disable_validation: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    batch_size_valid: Optional[int] = field(
        default=32, metadata={'help': ''}
    )

    max_valid_steps: Optional[int] = field(
        default=None, metadata={'help': ''}
    )

    curriculum: Optional[int] = field(
        default=0, metadata={'help': ''}
    )

    distributed_world_size: Optional[int] = field(
        default=1, metadata={'help': ''}
    )

    distributed_rank: Optional[int] = field(
        default=0, metadata={'help': ''}
    )

    distributed_backend: Optional[str] = field(
        default='nccl', metadata={'help': ''}
    )

    distributed_init_method: Optional[str] = field(
        default=None, metadata={'help': ''}
    )

    distributed_port: Optional[int] = field(
        default=-1, metadata={'help': ''}
    )

    device_id: Optional[int] = field(
        default=0, metadata={'help': ''}
    )

    distributed_no_spawn: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    ddp_backend: Optional[str] = field(
        default='c10d', metadata={'help': ''}
    )

    bucket_cap_mb: Optional[int] = field(
        default=25, metadata={'help': ''}
    )

    fix_batches_to_gpus: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    find_unused_parameters: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    fast_stat_sync: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    broadcast_buffers: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    nprocs_per_node: Optional[int] = field(
        default=1, metadata={'help': ''}
    )

    path: Optional[str] = field(
        default='./models/unimol.pt', metadata={'help': ''}
    )

    quiet: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    model_overrides: Optional[str] = field(
        default='{}', metadata={'help': ''}
    )

    results_path: Optional[str] = field(
        default='./infer_demo', metadata={'help': ''}
    )

    arch: Optional[str] = field(
        default='unimol_base', metadata={'help': ''}
    )

    mode: Optional[str] = field(
        default='train', metadata={'help': ''}
    )

    data: Optional[str] = field(
        default='./', metadata={'help': ''}
    )

    task_name: Optional[str] = field(
        default='demo', metadata={'help': ''}
    )

    classification_head_name: Optional[str] = field(
        default='demo', metadata={'help': ''}
    )

    num_classes: Optional[int] = field(
        default=2, metadata={'help': ''}
    )

    reg: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    no_shuffle: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    conf_size: Optional[int] = field(
        default=11, metadata={'help': ''}
    )

    remove_hydrogen: Optional[bool] = field(
        default=True, metadata={'help': ''}
    )

    remove_polar_hydrogen: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    max_atoms: Optional[int] = field(
        default=256, metadata={'help': ''}
    )

    dict_name: Optional[str] = field(
        default='dict.txt', metadata={'help': ''}
    )

    only_polar: Optional[int] = field(
        default=0, metadata={'help': ''}
    )

    adam_betas: Optional[str] = field(
        default='(0.9, 0.999)', metadata={'help': ''}
    )

    adam_eps: Optional[float] = field(
        default=1e-08, metadata={'help': ''}
    )

    weight_decay: Optional[float] = field(
        default=0.0, metadata={'help': ''}
    )

    force_anneal: Optional[int] = field(
        default=None, metadata={'help': ''}
    )

    lr_shrink: Optional[float] = field(
        default=0.1, metadata={'help': ''}
    )

    warmup_updates: Optional[int] = field(
        default=0, metadata={'help': ''}
    )

    no_seed_provided: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    encoder_layers: Optional[int] = field(
        default=15, metadata={'help': ''}
    )

    encoder_embed_dim: Optional[int] = field(
        default=512, metadata={'help': ''}
    )

    encoder_ffn_embed_dim: Optional[int] = field(
        default=2048, metadata={'help': ''}
    )

    encoder_attention_heads: Optional[int] = field(
        default=64, metadata={'help': ''}
    )

    dropout: Optional[float] = field(
        default=0.1, metadata={'help': ''}
    )

    emb_dropout: Optional[float] = field(
        default=0.1, metadata={'help': ''}
    )

    attention_dropout: Optional[float] = field(
        default=0.1, metadata={'help': ''}
    )

    activation_dropout: Optional[float] = field(
        default=0.0, metadata={'help': ''}
    )

    pooler_dropout: Optional[float] = field(
        default=0.0, metadata={'help': ''}
    )

    max_seq_len: Optional[int] = field(
        default=512, metadata={'help': ''}
    )

    activation_fn: Optional[str] = field(
        default='gelu', metadata={'help': ''}
    )

    pooler_activation_fn: Optional[str] = field(
        default='tanh', metadata={'help': ''}
    )

    post_ln: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    masked_token_loss: Optional[float] = field(
        default=-1.0, metadata={'help': ''}
    )

    masked_coord_loss: Optional[float] = field(
        default=-1.0, metadata={'help': ''}
    )

    masked_dist_loss: Optional[float] = field(
        default=-1.0, metadata={'help': ''}
    )

    x_norm_loss: Optional[float] = field(
        default=-1.0, metadata={'help': ''}
    )

    delta_pair_repr_norm_loss: Optional[float] = field(
        default=-1.0, metadata={'help': ''}
    )

    distributed_num_procs: Optional[int] = field(
        default=1, metadata={'help': ''}
    )

    def __post_init__(self):
        super().__post_init__()


class Config(Arguments, BaseConfig):

    pretrained_model_name_or_path = "DeepChem/ChemBERTa-77M-MLM"
