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

from dgllife.utils import (
    ScaffoldSplitter,
    RandomSplitter
)
from dgllife.data import MoleculeCSVDataset

from transformers import HfArgumentParser

from muben.utils.io import set_logging, logging_args, init_dir, save_json
from muben.utils.macro import SPLITTING

logger = logging.getLogger(__name__)


QM8_TASKS = [
    "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0",
    "f2-PBE0", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM",
    "f1-CAM", "f2-CAM"
]


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    dataset_path: Optional[str] = field(
        default='./qm8.csv',
        metadata={
            "help": "Path to the qm8.csv dataset on Molecule Net."
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
    force_scaffold: Optional[bool] = field(
        default=False,
        metadata={"help": "Force to use scaffold splitting even when the suggested method is random"}
    )
    n_jobs: Optional[int] = field(
        default=1,
        metadata={'help': 'How many processes to use to process the dataset.'}
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
        self.dataset_name = 'QM8'
        self.task = 'regression'

        if self.dataset_splitting_random_seeds is None:
            self.dataset_splitting_random_seeds: List[int] = [0, 1, 2, 3, 4]
        elif isinstance(self.dataset_splitting_random_seeds, int):
            self.dataset_splitting_random_seeds: List[int] = [self.dataset_splitting_random_seeds]


def main(args: Arguments):
    dataset_name = args.dataset_name
    task = args.task

    logger.info(f"Loading dataset")
    df = pd.read_csv(args.dataset_path)
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=None,
                                 node_featurizer=None,
                                 edge_featurizer=None,
                                 smiles_column='smiles',
                                 cache_file_path='./qm8_dglgraph.bin',
                                 task_names=QM8_TASKS,
                                 load=False,
                                 log_every=1000,
                                 init_mask=False,
                                 n_jobs=args.n_jobs)

    dataset.labels = dataset.labels if task == 'regression' else dataset.labels.to(torch.long)

    logger.info("Getting meta information")
    meta_dict = {
        'task_type': task,
        'n_tasks': dataset.labels.shape[1],
        'classes': None if task == 'regression' else torch.unique(dataset.labels).tolist(),
        'properties': QM8_TASKS
    }

    logger.info(f"Splitting dataset with {SPLITTING[dataset_name]} split strategy.")

    use_scaffold_splitting = SPLITTING[dataset_name] == 'scaffold' or args.force_scaffold
    splitter = ScaffoldSplitter() if use_scaffold_splitting else RandomSplitter()

    if use_scaffold_splitting:
        training_instances, valid_instances, test_instances = get_splits(dataset, splitter)

        save_dir = os.path.join(args.output_dir, dataset_name, "scaffold")
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
        arguments.log_path = os.path.join('./logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_path=arguments.log_path)
    logging_args(arguments)

    main(args=arguments)
