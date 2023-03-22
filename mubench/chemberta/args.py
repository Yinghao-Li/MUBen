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
        default='ChemBERTa', metadata={
            'help': "Name of the model",
            "choices": MODEL_NAMES
        }
    )

    # --- Reload training arguments to adjust default values ---
    batch_size: Optional[int] = field(
        default=8, metadata={'help': "Batch size."}
    )
    n_epochs: Optional[int] = field(
        default=50, metadata={'help': "How many epochs to train the model."}
    )
    lr: Optional[float] = field(
        default=5e-5, metadata={'help': "Learning Rate."}
    )

    def __post_init__(self):
        super().__post_init__()


class Config(Arguments, BaseConfig):

    pretrained_model_name_or_path = "DeepChem/ChemBERTa-77M-MLM"
