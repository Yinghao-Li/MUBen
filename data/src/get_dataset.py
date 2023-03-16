"""
Construct dataset from raw data files
"""
import os
import sys
import torch
import logging
import glob
import pandas as pd
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field

import dgllife
from dgllife.utils import (
    ScaffoldSplitter,
    RandomSplitter
)
from transformers import HfArgumentParser

from seqlbtoolkit.io import set_logging, logging_args, init_dir, save_json
from macro import DATASET_NAMES, SPLITTING

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
    output_dir: Optional[str] = field(
        default=".", metadata={"help": "where to save constructed dataset."}
    )
    dataset_splitting_random_seeds: Optional[int] = field(
        default=None,
        metadata={
            "nargs": "*",
            "help": "Random seeds used if the dataset is randomly split."
        }
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
    overwrite_output: Optional[bool] = field(
        default=False, metadata={'help': 'Whether overwrite existing outputs.'}
    )
    keep_cache: Optional[bool] = field(
        default=False, metadata={'help': 'Whether keep the dgllife cache file.'}
    )

    def __post_init__(self):
        if not self.dataset_names:
            self.dataset_names = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: List[str] = [self.dataset_names]

        if self.dataset_splitting_random_seeds is None:
            self.dataset_splitting_random_seeds: List[int] = [0, 1, 2, 3, 4]
        elif isinstance(self.dataset_splitting_random_seeds, int):
            self.dataset_splitting_random_seeds: List[int] = [self.dataset_splitting_random_seeds]

        if self.tasks is None:
            self.tasks: list = [None] * len(self.dataset_names)


def construct_dataset(args: Arguments):
    for dataset_name, task in zip(args.dataset_names, args.tasks):
        assert dataset_name in DATASET_NAMES, ValueError(f"Undefined dataset: {dataset_name}")

        logger.info(f"Processing dataset {dataset_name}")
        logger.info(f"Loading dataset")

        dataset = getattr(dgllife.data, dataset_name)()

        if not task:
            task = 'classification' if dataset.labels.dtype in (torch.int8, torch.int16, torch.int, torch.long) \
                or len(torch.unique(dataset.labels)) < len(dataset) / 10 else 'regression'
            logger.info(f"The task for dataset {dataset_name} is inferred as {task}.")
        dataset.labels = dataset.labels if task == 'regression' else dataset.labels.to(torch.long)

        logger.info("Getting meta information")
        meta_dict = {
            'task_type': task,
            'n_tasks': dataset.labels.shape[1],
            'classes': None if task == 'regression' else torch.unique(dataset.labels).tolist(),
        }

        logger.info(f"Splitting dataset with {SPLITTING[dataset_name]} split strategy.")

        use_scaffold_splitting = SPLITTING[dataset_name] == 'scaffold'
        splitter = ScaffoldSplitter() if use_scaffold_splitting else RandomSplitter()

        if use_scaffold_splitting:
            training_instances, valid_instances, test_instances = get_splits(dataset, splitter)

            save_dir = os.path.join(args.output_dir, dataset_name, "split-0")
            init_dir(save_dir, args.overwrite_output)
            logger.info(f"Saving dataset to {save_dir}")

            for instances, partition in zip((training_instances, valid_instances, test_instances),
                                            ('train', 'valid', 'test')):
                save_csv(instances, os.path.join(save_dir, f'{partition}.csv'))
            save_json(meta_dict, os.path.join(save_dir, 'meta.json'), collapse_level=2)

        else:
            for seed in args.dataset_splitting_random_seeds:
                logger.info(f"Dataset splitting random seed: {seed}")
                training_instances, valid_instances, test_instances = get_splits(dataset, splitter, seed)

                save_dir = os.path.join(args.output_dir, dataset_name, f"split-{seed}")
                init_dir(save_dir, args.overwrite_output)
                logger.info(f"Saving dataset to {save_dir}")

                for instances, partition in zip((training_instances, valid_instances, test_instances),
                                                ('train', 'valid', 'test')):
                    save_csv(instances, os.path.join(save_dir, f'{partition}.csv'))
                save_json(meta_dict, os.path.join(save_dir, 'meta.json'), collapse_level=2)

    if not args.keep_cache:
        logger.info("Clearing cache")
        cached_files = glob.glob('*_dglgraph.bin')
        for f in cached_files:
            os.remove(f)

    logger.info("Done.")
    return None


def get_splits(dataset, splitter, random_seed=None):
    if random_seed is None:
        train, valid, test = splitter.train_val_test_split(dataset)
    else:
        train, valid, test = splitter.train_val_test_split(dataset, random_state=random_seed)

    training_smiles = [dataset.smiles[idx] for idx in train.indices]
    training_labels = dataset.labels[train.indices, :]

    valid_smiles = [dataset.smiles[idx] for idx in valid.indices]
    valid_labels = dataset.labels[valid.indices, :]

    test_smiles = [dataset.smiles[idx] for idx in test.indices]
    test_labels = dataset.labels[test.indices, :]

    try:
        training_label_masks = dataset.mask[train.indices, :].to(torch.long).tolist()
        valid_label_masks = dataset.mask[valid.indices, :].to(torch.long).tolist()
        test_label_masks = dataset.mask[test.indices, :].to(torch.long).tolist()
    except (AttributeError, TypeError):
        training_label_masks = None
        valid_label_masks = None
        test_label_masks = None

    training_instances = {
        "smiles": training_smiles,
        "labels": training_labels.tolist(),
        "masks": training_label_masks
    }
    valid_instances = {
        "smiles": valid_smiles,
        "labels": valid_labels.tolist(),
        "masks": valid_label_masks
    }
    test_instances = {
        "smiles": test_smiles,
        "labels": test_labels.tolist(),
        "masks": test_label_masks
    }
    return training_instances, valid_instances, test_instances


def save_csv(instances, output_path):
    df = pd.DataFrame(instances)
    df.to_csv(output_path)


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
        arguments.log_path = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=arguments.log_path)
    logging_args(arguments)

    construct_dataset(args=arguments)
