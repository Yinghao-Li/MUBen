"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: move the models from Deep Ensembles to the `none` folders.
"""


import sys
import glob
import shutil
import logging
import os.path as op

from typing import Optional
from dataclasses import dataclass, field

from muben.utils.io import set_logging, init_dir
from muben.utils.argparser import ArgumentParser
from muben.utils.macro import DATASET_NAMES, MODEL_NAMES, FINGERPRINT_FEATURE_TYPES


logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    src_folder: Optional[str] = field(
        default=None, metadata={"help": "Source folder that stores the results."}
    )
    tgt_folder: Optional[str] = field(
        default=None, metadata={"help": "Target folder that stores the results."}
    )
    dataset_names: Optional[str] = field(
        default=None,
        metadata={
            "nargs": "*",
            "choices": DATASET_NAMES,
            "help": "A list of dataset names.",
        },
    )
    model_name: Optional[str] = field(
        default=None, metadata={"choices": MODEL_NAMES, "help": "A list of model names"}
    )
    feature_type: Optional[str] = field(
        default="none",
        metadata={
            "choices": FINGERPRINT_FEATURE_TYPES,
            "help": "Feature type that the DNN model uses.",
        },
    )
    overwrite: Optional[bool] = field(
        default=False, metadata={"help": "Whether overwrite existing outputs."}
    )

    def __post_init__(self):
        if self.dataset_names is None:
            self.dataset_names: list[str] = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: list[str] = [self.dataset_names]

        if self.model_name == "DNN":
            assert self.feature_type != "none", ValueError(
                "Invalid feature type for DNN!"
            )
            self.model_name = f"{self.model_name}-{self.feature_type}"


def main(args: Arguments):
    for dataset_name in args.dataset_names:
        src_dir = op.join(args.src_folder, dataset_name, args.model_name)
        src_paths = glob.glob(op.join(src_dir, "**", "preds", "*.pt"), recursive=True)

        for src_path in src_paths:
            tgt_path = src_path.replace(args.src_folder, args.tgt_folder)
            tgt_dir = op.dirname(tgt_path)
            init_dir(tgt_dir, clear_original_content=args.overwrite)

            logger.info(f"Moving {src_path} to {tgt_path}")
            shutil.copy(src_path, tgt_path)

    return None


if __name__ == "__main__":
    # --- set up arguments ---
    parser = ArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    set_logging()

    main(args=arguments)
