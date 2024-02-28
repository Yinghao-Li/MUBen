"""
# Author: Yinghao Li
# Modified: February 28th, 2024
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
class ArgumentsLinear(BaseArguments):

    # --- Reload model arguments to adjust default values ---
    model_name: Optional[str] = field(
        default="ChemBERTa",
        metadata={"help": "Name of the model", "choices": MODEL_NAMES},
    )
    pretrained_model_name_or_path: str = field(
        default="DeepChem/ChemBERTa-77M-MLM",
        metadata={"help": "The name or path of the Huggingface model to be used."},
    )


@dataclass
class ConfigLinear(ArgumentsLinear, BaseConfig):
    pass
