"""
Generate/Update meta files without reloading datasets
"""
import os
import sys
import glob
import logging
import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from ast import literal_eval
from transformers import HfArgumentParser

from seqlbtoolkit.io import set_logging, logging_args, save_json
from mubench.utils.macro import DATASET_NAMES

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
            "help": "The name of the dataset to construct."
        }
    )
    data_dir: Optional[str] = field(
        default=".", metadata={"help": "where to save constructed dataset."}
    )
    tasks: Optional[str] = field(
        default=None,
        metadata={
            "nargs": "*",
            "help": "Tasks of each dataset. Will be inferred from the dataset if left empty."
        }
    )
    log_path: Optional[str] = field(
        default=None, metadata={"help": "Path to save the log file."}
    )

    def __post_init__(self):
        if not self.dataset_names:
            self.dataset_names = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: List[str] = [self.dataset_names]

        if self.tasks is None:
            self.tasks: list = [None] * len(self.dataset_names)


def main(args: Arguments):
    for dataset_name, task in zip(args.dataset_names, args.tasks):
        assert dataset_name in DATASET_NAMES, ValueError(f"Undefined dataset: {dataset_name}")

        logger.info(f"Processing dataset {dataset_name}")

        dataset_dir = os.path.join(args.data_dir, dataset_name)
        split_dirs = glob.glob(os.path.join(dataset_dir, "split-*"))

        for split_dir in split_dirs:
            df = pd.read_csv(os.path.join(split_dir, 'test.csv'))
            lbs = df.labels.map(literal_eval).tolist()
            n_tasks = len(lbs[0])

            if not task:
                task = 'classification' if isinstance(lbs[0][0], int) else 'regression'
                logger.info(f"The task for dataset {dataset_name} is inferred as {task}.")

            logger.info("Getting meta information")
            meta_dict = {
                'task_type': task,
                'n_tasks': n_tasks,
                'classes': None if task == 'regression' else np.unique(np.asarray(lbs)).tolist()
            }

            logger.info("Saving metadata.")
            save_json(meta_dict, os.path.join(split_dir, 'meta.json'), collapse_level=2)

    logger.info("Done.")
    return None


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        arguments, = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_path", None):
        arguments.log_path = os.path.join('./logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=arguments.log_path)
    logging_args(arguments)

    main(args=arguments)
