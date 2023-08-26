"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: construct dataset from the Uni-Mol datasets.
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
from sklearn.model_selection import train_test_split

from transformers import set_seed

from muben.utils.macro import DATASET_NAMES, CLASSIFICATION_DATASET, REGRESSION_DATASET
from muben.utils.io import set_logging, logging_args, save_json, init_dir, load_lmdb
from muben.utils.argparser import ArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    dataset_names: Optional[str] = field(
        default=None,
        metadata={"nargs": "*", "help": "The name of the dataset to construct."},
    )
    unimol_data_dir: Optional[str] = field(
        default="./data/UniMol/",
        metadata={"help": "Path to the pre-processed UniMol dataset."},
    )
    output_dir: Optional[str] = field(
        default="./data/files/", metadata={"help": "where to save constructed dataset."}
    )
    seed: Optional[int] = field(
        default=None,
        metadata={
            "nargs": "*",
            "help": "Random seed. Leave it unspecified to keep the original scaffold split of Uni-Mol. "
            "`seed` can also be assigned to multiple values to create multiple datasets.",
        },
    )
    log_path: Optional[str] = field(
        default=None, metadata={"help": "Path to save the log file."}
    )
    overwrite_output: Optional[bool] = field(
        default=False, metadata={"help": "Whether overwrite existing outputs."}
    )

    def __post_init__(self):
        if not self.dataset_names:
            self.dataset_names: List[str] = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: List[str] = [self.dataset_names]

        if isinstance(self.seed, int):
            self.seed: List[int] = [self.seed]


def main(args: Arguments):
    for dataset_name in args.dataset_names:
        skip_dataset = False
        assert dataset_name in DATASET_NAMES, ValueError(
            f"Undefined dataset: {dataset_name}"
        )

        if dataset_name in CLASSIFICATION_DATASET:
            task = "classification"
        elif dataset_name in REGRESSION_DATASET:
            task = "regression"
        else:
            raise ValueError(f"Task undefined for dataset {dataset_name}")

        if args.seed is None:
            logger.info(f"Processing dataset {dataset_name}")
            logger.info(f"Loading dataset")

            output_dir = os.path.join(args.output_dir, dataset_name)
            init_dir(output_dir, args.overwrite_output)

            for partition in ("train", "valid", "test"):
                logger.info(partition)

                # read data points
                lmdb_path = os.path.join(
                    args.unimol_data_dir, dataset_name, f"{partition}.lmdb"
                )
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
                    smiles.append(data["smi"])
                    lbs.append(data["target"])

                lbs = np.asarray(lbs)
                lbs = lbs.astype(int) if task == "classification" else lbs
                masks = np.ones_like(lbs)
                if task == "classification":
                    masks[lbs == -1] = 0

                insts = {
                    "smiles": smiles,
                    "labels": lbs.tolist(),
                    "masks": masks.tolist(),
                }
                save_csv(insts, os.path.join(output_dir, f"{partition}.csv"))

            if skip_dataset:
                continue

            meta_dict = {
                "task_type": task,
                "n_tasks": lbs.shape[-1],
                "classes": None if task == "regression" else [0, 1],
            }
            save_json(
                meta_dict, os.path.join(output_dir, "meta.json"), collapse_level=2
            )

        else:
            for seed in args.seed:
                set_seed(seed)

                logger.info(f"Processing dataset {dataset_name}")
                logger.info(f"Loading dataset")

                output_dir = os.path.join(args.output_dir, dataset_name, f"seed-{seed}")
                init_dir(output_dir, args.overwrite_output)

                smiles = list()
                lbs = list()
                ori_ids = list()

                lengths = {p: 0 for p in ("train", "valid", "test")}

                for partition in ("train", "valid", "test"):
                    # read data points
                    lmdb_path = os.path.join(
                        args.unimol_data_dir, dataset_name, f"{partition}.lmdb"
                    )
                    results = load_lmdb(lmdb_path, ["smi", "target", "ori_index"])
                    if results == -1:
                        skip_dataset = True
                        break

                    smi, tgt, ori_id = results

                    smiles += smi
                    lbs += tgt
                    ori_ids += ori_id

                    lengths[partition] = len(smi)

                if skip_dataset:
                    continue

                lbs = np.asarray(lbs)
                lbs = lbs.astype(int) if task == "classification" else lbs
                masks = np.ones_like(lbs)

                if task == "classification":
                    masks[lbs == -1] = 0

                split_and_save(smiles, lbs, masks, ori_ids, lengths, seed, output_dir)

                meta_dict = {
                    "task_type": task,
                    "n_tasks": lbs.shape[-1],
                    "classes": None if task == "regression" else [0, 1],
                    "random_split": True,
                }
                save_json(
                    meta_dict, os.path.join(output_dir, "meta.json"), collapse_level=2
                )

    logger.info("Done.")
    return None


def save_csv(instances, output_path):
    df = pd.DataFrame(instances)
    df.to_csv(output_path)


def split_and_save(smiles, lbs, masks, ori_ids, lengths, seed, output_dir):
    assert len(smiles) == len(lbs) == len(masks) == len(ori_ids)

    # train:t, valid:v, test:s
    ts, vs, tl, vl, tm, vm, tid, vid = train_test_split(
        smiles, lbs, masks, ori_ids, test_size=lengths["valid"], random_state=seed
    )
    ts, ss, tl, sl, tm, sm, tid, sid = train_test_split(
        ts, tl, tm, tid, test_size=lengths["test"], random_state=seed
    )

    training_instances = {
        "smiles": ts,
        "labels": tl.tolist(),
        "masks": tm.tolist(),
        "ori_ids": tid,
    }
    save_csv(training_instances, os.path.join(output_dir, f"train.csv"))
    valid_instances = {
        "smiles": vs,
        "labels": vl.tolist(),
        "masks": vm.tolist(),
        "ori_ids": vid,
    }
    save_csv(valid_instances, os.path.join(output_dir, f"valid.csv"))
    test_instances = {
        "smiles": ss,
        "labels": sl.tolist(),
        "masks": sm.tolist(),
        "ori_ids": sid,
    }
    save_csv(test_instances, os.path.join(output_dir, f"test.csv"))
    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith(".py"):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = ArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_path", None):
        arguments.log_path = os.path.join(
            "./logs", f"{_current_file_name}", f"{_time}.log"
        )

    set_logging(log_path=arguments.log_path)
    logging_args(arguments)

    main(args=arguments)
