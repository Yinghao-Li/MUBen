"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: Special arguments and configurations for ChemBERTa.
"""

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
        default="ChemBERTa",
        metadata={"help": "Name of the model", "choices": MODEL_NAMES},
    )


@dataclass
class Config(Arguments, BaseConfig):
    pretrained_model_name_or_path = "DeepChem/ChemBERTa-77M-MLM"
