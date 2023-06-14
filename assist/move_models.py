"""
Run the basic model and training process
"""

import sys
import shutil
import logging
import os.path as op

from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from muben.utils.io import set_logging, set_log_path, init_dir
from muben.utils.macro import (
    DATASET_NAMES,
    MODEL_NAMES,
    UncertaintyMethods,
    FINGERPRINT_FEATURE_TYPES
)


logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    dataset_names: Optional[str] = field(
        default=None,
        metadata={
            "nargs": "*",
            "choices": DATASET_NAMES,
            "help": "A list of dataset names."
        }
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "choices": MODEL_NAMES,
            "help": "A list of model names"
        }
    )
    feature_type: Optional[str] = field(
        default="none",
        metadata={
            "choices": FINGERPRINT_FEATURE_TYPES,
            "help": "Feature type that the DNN model uses."
        }
    )
    seeds: Optional[int] = field(
        default=None,
        metadata={
            "nargs": "*",
            "help": "A list of random seeds of individual runs."
        }
    )
    result_folder: Optional[str] = field(
        default=".", metadata={"help": "The folder which holds the results."}
    )
    overwrite_folder: Optional[bool] = field(
        default=False, metadata={'help': 'Whether overwrite existing outputs.'}
    )

    def __post_init__(self):
        if self.dataset_names is None:
            self.dataset_names: list[str] = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: list[str] = [self.dataset_names]

        if self.model_name == "DNN":
            assert self.feature_type != 'none', ValueError("Invalid feature type for DNN!")
            self.model_name = f"{self.model_name}-{self.feature_type}"

        if self.seeds is None:
            self.seeds: list[int] = [0, 1, 2]
        elif isinstance(self.seeds, int):
            self.seeds: list[int] = [self.seeds]


def main(args: Arguments):

    for dataset_name in args.dataset_names:
        for seed in args.seeds:
            from_dir = op.join(args.result_folder, dataset_name, args.model_name, UncertaintyMethods.ensembles)
            to_dir = op.join(args.result_folder, dataset_name, args.model_name, UncertaintyMethods.none)

            from_dir = op.join(from_dir, f"seed-{seed}")
            to_dir = op.join(to_dir, f"seed-{seed}")

            from_path = op.join(from_dir, f"model_best.ckpt")
            to_path = op.join(to_dir, f"model_best.ckpt")

            if not op.exists(from_path):
                logger.warning(f"Model {from_path} does not exist!")
                continue

            init_dir(to_dir, clear_original_content=args.overwrite_folder)

            logger.info(f"Moving {from_path} to {to_path}")
            shutil.copy(from_path, to_path)

    return None


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        arguments, = parser.parse_args_into_dataclasses()

    set_logging()

    main(args=arguments)

