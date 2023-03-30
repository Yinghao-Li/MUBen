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
            'help': "The name of the model to be used.",
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
    batch_size: Optional[int] = field(
        default=32, metadata={'help': 'Batch size'}
    )

    batch_size_inference: Optional[int] = field(
        default=32, metadata={'help': 'Validation Batch size, should be a smaller value than `batch_size` since '
                                      'we append the conformation dimension (11) to batch size during inference.'}
    )

    max_atoms: Optional[int] = field(
        default=256, metadata={'help': 'the maximum number of atoms in the input molecules.'}
    )

    max_seq_len: Optional[int] = field(
        default=512, metadata={'help': 'Maximum length of the atom tokens.'}
    )

    only_polar: Optional[int] = field(
        default=0, metadata={'help': "1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen "}
    )

    dropout: Optional[float] = field(
        default=0.1, metadata={'help': 'The `pooler dropout` argument in the original implementation. '
                                       'Controls the dropout ratio of the classification layers.'}
    )

    def __post_init__(self):
        super().__post_init__()

        self.pooler_dropout = self.dropout


@dataclass
class Config(Arguments, BaseConfig):

    n_conformation = 11
