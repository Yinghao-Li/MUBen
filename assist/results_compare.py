import sys
import logging
import os.path as op
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from muben.utils.macro import (
    CLASSIFICATION_DATASET,
    REGRESSION_DATASET,
    FINGERPRINT_FEATURE_TYPES,
    MODEL_NAMES
)
from muben.utils.io import init_dir, set_logging
from muben.utils.argparser import ArgumentParser

logger = logging.getLogger(__name__)

CLASSIFICATION_UNCERTAINTY = [
    'none',
    'TemperatureScaling',
    'FocalLoss',
    'MCDropout',
    'SWAG',
    'BBP',
    'SGLD',
    'DeepEnsembles'
]
REGRESSION_UNCERTAINTY = [
    'none',
    'MCDropout',
    'SWAG',
    'BBP',
    'SGLD',
    'DeepEnsembles'
]
CLASSIFICATION_METRICS = ['roc-auc', 'ece', 'nll', 'brier']
REGRESSION_METRICS = ['rmse', 'mae', 'nll', 'ce']

DATASETS=['bace', 'bbbp', 'clintox', 'esol', 'freesolv', 'lipo', 'tox21', 'qm7']
MODELS=["ChemBERTa", "GROVER", "Uni-Mol"]
# MODELS=["DNN-rdkit", "ChemBERTa", "GROVER", "Uni-Mol"]

@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---

    model_names: Optional[str] = field(
        default=None,
        metadata={
            "nargs": "*",
            "help": "A list of model names."
        }
    )
    feature_type: Optional[str] = field(
        default="rdkit",
        metadata={
            "choices": FINGERPRINT_FEATURE_TYPES,
            "help": "Feature type that the DNN model uses."
        }
    )
    ori_result_dir: Optional[str] = field(
        default="./output/RESULTS/", metadata={"help": "Directory the scores."}
    )
    tgt_result_dir: Optional[str] = field(
        default="./output/RESULTS/", metadata={"help": "Directory the scores."}
    )
    output_dir: Optional[str] = field(
        default="./output/RESULTS/", metadata={"help": "Where to save the ranks."}
    )

    def __post_init__(self):
        if isinstance(self.model_names, str):
            self.model_names: list[str] = [self.model_names]

        for idx, model_name in enumerate(self.model_names):
            if model_name == "DNN":
                assert self.feature_type != 'none', ValueError("Invalid feature type for DNN!")
                self.model_names[idx] = f"DNN-{self.feature_type}"


def main():
    diff_dfs = list()
    for dataset_name in CLASSIFICATION_DATASET:
        if dataset_name not in DATASETS:
            continue

        for model_name in MODELS:

            file_name = f"{model_name}-{dataset_name}.csv"
            scaffold_path = op.join(scaffold_result_dir, file_name)
            random_path = op.join(random_result_dir, file_name)

            scaffold_df = pd.read_csv(scaffold_path, index_col='method')
            random_df = pd.read_csv(random_path, index_col='method')

            scaffold_df = scaffold_df.reindex([f'{model_name}-{unc}' for unc in CLASSIFICATION_UNCERTAINTY])
            random_df = random_df.reindex([f'{model_name}-{unc}' for unc in CLASSIFICATION_UNCERTAINTY])

            for metrics in CLASSIFICATION_METRICS:
                scaffold_df[metrics] = [m for m in scaffold_df[f'{metrics}-mean']]
                random_df[metrics] = [m for m in random_df[f'{metrics}-mean']]
            scaffold_df = scaffold_df[CLASSIFICATION_METRICS]
            random_df = random_df[CLASSIFICATION_METRICS]

            diff_df = (random_df - scaffold_df) / scaffold_df
            diff_dfs.append(diff_df)

    merged_df = pd.concat(diff_dfs, ignore_index=True)
    results = [f'{merged_df.mean()[m] * 100:.2f}' for m in CLASSIFICATION_METRICS]
    results = " & ".join(results)
    logger.info(results)

    diff_dfs = list()
    for dataset_name in REGRESSION_DATASET:
        if dataset_name not in DATASETS:
            continue

        for model_name in MODELS:

            file_name = f"{model_name}-{dataset_name}.csv"
            scaffold_path = op.join(scaffold_result_dir, file_name)
            random_path = op.join(random_result_dir, file_name)

            scaffold_df = pd.read_csv(scaffold_path, index_col='method')
            random_df = pd.read_csv(random_path, index_col='method')

            scaffold_df = scaffold_df.reindex([f'{model_name}-{unc}' for unc in REGRESSION_UNCERTAINTY])
            random_df = random_df.reindex([f'{model_name}-{unc}' for unc in REGRESSION_UNCERTAINTY])

            for metrics in REGRESSION_METRICS:
                scaffold_df[metrics] = [m for m in scaffold_df[f'{metrics}-mean']]
                random_df[metrics] = [m for m in random_df[f'{metrics}-mean']]
            scaffold_df = scaffold_df[REGRESSION_METRICS]
            random_df = random_df[REGRESSION_METRICS]

            diff_df = (random_df - scaffold_df) / scaffold_df
            diff_dfs.append(diff_df)

    merged_df = pd.concat(diff_dfs, ignore_index=True)
    results = [f'{merged_df.mean()[m] * 100:.2f}' for m in REGRESSION_METRICS]
    results = " & ".join(results)
    logger.info(results)


if __name__ == '__main__':
    # --- set up arguments ---
    parser = ArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        arguments, = parser.parse_args_into_dataclasses()

    set_logging(log_path=None)

    main(arguments)
