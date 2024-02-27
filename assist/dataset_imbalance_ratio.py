"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: calculate the imbalance ratio of the datasets.
"""

import sys
import logging
import os.path as op
import numpy as np
from dataclasses import dataclass, field
from muben.dataset.dataset import Dataset
from muben.utils.io import set_logging
from muben.utils.macro import CLASSIFICATION_DATASET
from muben.utils.argparser import ArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    dataset_folder: str = field(metadata={"help": "The folder containing all datasets"})


def main(args: Arguments):
    for dataset_name in CLASSIFICATION_DATASET:
        dataset_dir = op.join(args.dataset_folder, dataset_name)

        training_dataset = Dataset().read_csv(data_dir=dataset_dir, partition="train")
        valid_dataset = Dataset().read_csv(data_dir=dataset_dir, partition="valid")
        test_dataset = Dataset().read_csv(data_dir=dataset_dir, partition="test")

        lbs = np.concatenate((training_dataset.lbs, valid_dataset.lbs, test_dataset.lbs), axis=0)

        ratios = list()
        for lbs_ in lbs.T:
            n_pos = np.sum(lbs_ == 1)
            n_neg = np.sum(lbs_ == 0)
            pos_ratio = n_pos / (n_pos + n_neg)
            ratios.append(pos_ratio if pos_ratio > 0.5 else (1 - pos_ratio))

        logger.info(f"{dataset_name}: Mean: {np.mean(ratios):.4f}; Max: {np.max(ratios):.4f}")


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
    main(arguments)
