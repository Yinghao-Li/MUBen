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

    # --- Reload training arguments to adjust default values ---
    lr_scheduler_type: Optional[str] = field(
        default='polynomial', metadata={
            'help': "Learning rate scheduler with warm ups defined in `transformers`, Please refer to "
                    "https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules for details",
            'choices': ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
        }
    )
    grad_norm: Optional[float] = field(
        default=1, metadata={"help": "Gradient norm. Default is 0 (do not clip gradient)"}
    )

    # --- update model parameters from Uni-Mol ---
    checkpoint_path: Optional[str] = field(
        default='', metadata={'help': "Path to the pre-trained model"}
    )
    n_feature_generating_threads: Optional[int] = field(
        default=8, metadata={'help': "Number of feature generation threads"}
    )

    # --- Arguments from Uni-Mol original implementation ---
    ema_decay: Optional[float] = field(
        default=-1.0, metadata={'help': ''}
    )

    validate_with_ema: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    batch_size: Optional[int] = field(
        default=32, metadata={'help': ''}
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

    weight_decay: Optional[float] = field(
        default=0.0, metadata={'help': ''}
    )

    force_anneal: Optional[int] = field(
        default=None, metadata={'help': ''}
    )

    no_seed_provided: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )

    dropout: Optional[float] = field(
        default=0.1, metadata={'help': 'The `pooler dropout` argument in the original implementation. '
                                       'Controls the dropout ratio of the classification layers.'}
    )

    max_seq_len: Optional[int] = field(
        default=512, metadata={'help': ''}
    )

    def __post_init__(self):
        super().__post_init__()

        self.pooler_dropout = self.dropout


@dataclass
class Config(Arguments, BaseConfig):

    n_conformation = 11
