"""
Construct dataset from raw data files
"""
import os
import sys
import logging
import pandas as pd
import lmdb
import pickle
import numpy as np
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from transformers import HfArgumentParser

from macro import DATASET_NAMES, CLASSIFICATION_DATASET, REGRESSION_DATASET
from mubench.utils.io import set_logging, logging_args, save_json, init_dir

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
    unimol_data_path: Optional[str] = field(
        default='./UniMol/',
        metadata={"help": "Path to the pre-processed UniMol dataset."}
    )
    output_dir: Optional[str] = field(
        default=".", metadata={"help": "where to save constructed dataset."}
    )
    log_path: Optional[str] = field(
        default=None, metadata={"help": "Path to save the log file."}
    )
    overwrite_output: Optional[bool] = field(
        default=False, metadata={'help': 'Whether overwrite existing outputs.'}
    )

    def __post_init__(self):

        if not self.dataset_names:
            self.dataset_names: List[str] = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: List[str] = [self.dataset_names]


def main(args: Arguments):
    for dataset_name in args.dataset_names:
        skip_dataset = False
        assert dataset_name in DATASET_NAMES, ValueError(f"Undefined dataset: {dataset_name}")

        if dataset_name in CLASSIFICATION_DATASET:
            task = 'classification'
        elif dataset_name in REGRESSION_DATASET:
            task = 'regression'
        else:
            raise ValueError(f"Task undefined for dataset {dataset_name}")

        logger.info(f"Processing dataset {dataset_name}")
        logger.info(f"Loading dataset")

        output_dir = os.path.join(args.output_dir, dataset_name)
        init_dir(output_dir, args.overwrite_output)

        for partition in ('train', 'valid', 'test'):

            logger.info(partition)

            # read data points
            lmdb_path = os.path.join(args.unimol_data_path, dataset_name, f"{partition}.lmdb")
            if not os.path.exists(lmdb_path):
                logger.warning(f"Path {lmdb_path} does not exist!")
                skip_dataset = True
                break
            env = lmdb.open(
                lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
            txn = env.begin()
            keys = list(txn.cursor().iternext(values=False))

            smiles = list()
            lbs = list()
            for idx in tqdm(keys):
                datapoint_pickled = txn.get(idx)
                data = pickle.loads(datapoint_pickled)
                smiles.append(data['smi'])
                lbs.append(data['target'])

            lbs = np.asarray(lbs)
            lbs = lbs.astype(int) if task == 'classification' else lbs
            masks = np.ones_like(lbs)
            if task == 'classification':
                masks[lbs == -1] = 0

            insts = {
                "smiles": smiles,
                "labels": lbs.tolist(),
                "masks": masks.tolist()
            }
            save_csv(insts, os.path.join(output_dir, f"{partition}.csv"))
        
        if skip_dataset:
            continue

        meta_dict = {
            'task_type': task,
            'n_tasks': lbs.shape[-1],
            'classes': None if task == 'regression' else [0, 1],
        }
        save_json(meta_dict, os.path.join(output_dir, "meta.json"), collapse_level=2)

    logger.info("Done.")
    return None


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
