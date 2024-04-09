"""
# Author: Yinghao Li
# Modified: April 9th, 2024
# ---------------------------------------
# Description:

Group the test datasets by their similarity to the training dataset.
"""

import sys
import random
import logging
import pandas as pd
import numpy as np
import os.path as osp
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field

from transformers import set_seed, HfArgumentParser

from muben.utils.macro import DATASET_NAMES
from muben.utils.io import set_logging, logging_args, save_json

from itertools import product
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from multiprocessing import get_context

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def compute_scaffolds(smiles_list):
    scaffolds = list()
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        scaffolds.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)))
    return scaffolds


def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES string: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)


def calculate_similarity(pair):
    fp1, fp2 = pair
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def generate_fingerprints(smiles_list):
    fps = list()
    with get_context("fork").Pool(8) as pool:
        # fingerprints = pool.map(get_fingerprint, smiles_list)
        for fp in tqdm(pool.imap(get_fingerprint, smiles_list), total=len(smiles_list)):
            fps.append(fp)
    return fps


def compute_similarities(set1_fps, set2_fps):
    print("Generating fingerprint pairs")
    fps_tuples = list(product(set1_fps, set2_fps))

    print("Computing similarities")
    sims = list()
    with get_context("fork").Pool(16) as pool:
        for sim in tqdm(pool.imap(calculate_similarity, fps_tuples), total=len(fps_tuples)):
            sims.append(sim)
    return sims


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
    n_groups: int = field(
        default=5,
        metadata={"help": "Number of groups to divide the test dataset."},
    )
    log_path: Optional[str] = field(default=None, metadata={"help": "Path to save the log file."})
    seed: int = field(default=42, metadata={"help": "Random seed"})

    def __post_init__(self):
        if not self.dataset_names:
            self.dataset_names: List[str] = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: List[str] = [self.dataset_names]


def main(args: Arguments):
    set_seed(args.seed)
    for dataset_name in args.dataset_names:
        assert dataset_name in DATASET_NAMES, ValueError(f"Undefined dataset: {dataset_name}")

        logger.info(f"Processing dataset {dataset_name}")
        logger.info(f"Loading dataset")

        # read data points
        training_data = pd.read_csv(osp.join(args.data_folder, dataset_name, "train.csv"))
        training_smiles = training_data["smiles"].tolist()

        test_data = pd.read_csv(osp.join(args.data_folder, dataset_name, "test.csv"))
        test_smiles = test_data["smiles"].tolist()

        logger.info(f"Computing training scaffolds")
        training_scaffolds = compute_scaffolds(training_smiles)
        training_unique_scaffolds = list(set(training_scaffolds))
        training_unique_scaffolds = [sf for sf in training_unique_scaffolds if sf]

        # training_unique_scaffolds = random.sample(training_unique_scaffolds, 3)

        logger.info(f"Generating training fingerprints")
        training_fps = generate_fingerprints(training_unique_scaffolds)

        logger.info(f"Generating test fingerprints")
        test_fps = generate_fingerprints(test_smiles)

        logger.info(f"Computing similarities")
        similarity_scores = compute_similarities(training_fps, test_fps)

        sims = np.array(similarity_scores).reshape(len(training_unique_scaffolds), len(test_smiles))
        sims_avg = np.mean(sims, axis=0)

        sorted_indices = np.argsort(sims_avg)

        group_size = len(test_smiles) // args.n_groups
        for i in range(args.n_groups):
            if i == args.n_groups - 1:
                group = sorted_indices[i * group_size :]
            else:
                group = sorted_indices[i * group_size : (i + 1) * group_size]

            output_path = osp.join(args.data_folder, dataset_name, f"test-group-{i}.json")
            save_json(group.tolist(), output_path)

    logger.info("Done.")
    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = osp.basename(__file__)
    if _current_file_name.endswith(".py"):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (arguments,) = parser.parse_json_file(json_file=osp.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        (arguments,) = parser.parse_yaml_file(json_file=osp.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_path", None):
        arguments.log_path = osp.join("./logs", f"{_current_file_name}", f"{_time}.log")

    set_logging(log_path=arguments.log_path)
    logging_args(arguments)

    main(args=arguments)
