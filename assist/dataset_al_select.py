"""
# Author: Yinghao Li
# Modified: November 17th, 2023
# ---------------------------------------
# Description: select initial data points for active learning.
"""

import sys
import logging
import pandas as pd
import numpy as np
import os.path as osp
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from transformers import set_seed

from muben.utils.macro import DATASET_NAMES
from muben.utils.io import set_logging, logging_args, save_json
from muben.utils.argparser import ArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    data_folder: str = field(default="./data/files/", metadata={"help": "where to save constructed dataset."})
    dataset_names: str = field(
        default=None,
        metadata={"nargs": "*", "help": "The name of the dataset to construct."},
    )
    n_init_instances: int = field(
        default=100,
        metadata={"help": "Number of initial instances to select for each class."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})
    log_path: Optional[str] = field(default=None, metadata={"help": "Path to save the log file."})
    overwrite_output: Optional[bool] = field(default=False, metadata={"help": "Whether overwrite existing outputs."})

    def __post_init__(self):
        if not self.dataset_names:
            self.dataset_names: List[str] = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: List[str] = [self.dataset_names]


def main(args: Arguments):
    set_seed(args.seed)
    for dataset_name in args.dataset_names:
        assert dataset_name in DATASET_NAMES, ValueError(f"Undefined dataset: {dataset_name}")

        output_path = osp.join(args.data_folder, dataset_name, f"al-{args.n_init_instances}.json")
        if osp.exists(output_path) and not args.overwrite_output:
            logger.warning(f"File {output_path} already exists. Skip.")

        logger.info(f"Processing dataset {dataset_name}")
        logger.info(f"Loading dataset")

        # read data points
        data = pd.read_csv(osp.join(args.data_folder, dataset_name, "train.csv"))
        n_training_instances = len(data)

        # random select training instances
        ids_list = list(range(n_training_instances))
        np.random.shuffle(ids_list)
        ids_list = ids_list[: args.n_init_instances]
        sorted_ids_list = sorted(ids_list)

        # save selected instances
        save_json(sorted_ids_list, output_path)

    logger.info("Done.")
    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = osp.basename(__file__)
    if _current_file_name.endswith(".py"):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = ArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=osp.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_path", None):
        arguments.log_path = osp.join("./logs", f"{_current_file_name}", f"{_time}.log")

    set_logging(log_path=arguments.log_path)
    logging_args(arguments)

    main(args=arguments)
