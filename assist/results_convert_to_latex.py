"""
Automatically generate the result tables in the appendix.
"""

import sys
import os.path as op
import pandas as pd
from muben.utils.macro import CLASSIFICATION_DATASET, REGRESSION_DATASET
from muben.utils.io import init_dir

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from muben.utils.macro import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    dataset_mapping,
    metrics_mapping
)


MODEL_NAMES = [
    "DNN-rdkit",
    "ChemBERTa",
    "GROVER",
    "Uni-Mol"
]
model_mapping = {
    "DNN-rdkit": 'DNN',
    "ChemBERTa": "ChemBERTa",
    "GROVER": "GROVER",
    "Uni-Mol": "Uni-Mol"
}
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
uncertainty_mapping = {
    'none': 'Deterministic',
    'TemperatureScaling': 'Temperature',
    'FocalLoss': 'Focal Loss',
    'MCDropout': 'MC Dropout',
    'SWAG': 'SWAG',
    'BBP': 'BBP',
    'SGLD': 'SGLD',
    'DeepEnsembles': 'Ensembles'
}


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    result_dir: Optional[str] = field(
        default="./output/RESULTS/", metadata={"help": "The folder which holds the results."}
    )
    output_dir: Optional[str] = field(
        default="latex", metadata={"help": "Directory to save the latex tables."}
    )


def main(args: Arguments):
    score_path = op.join(args.result_dir, 'scores')

    # for classification datasets
    mapped_metrics = [metrics_mapping[m] for m in CLASSIFICATION_METRICS]
    for dataset in CLASSIFICATION_DATASET:
        dfs = list()
        for model_name in MODEL_NAMES:
            result_file = op.join(score_path, f"{model_name}-{dataset}.csv")
            df = pd.read_csv(result_file, index_col='method')
            for metrics in CLASSIFICATION_METRICS:
                df[metrics_mapping[metrics]] = [
                    (f"{m:.4f}" if m == m else '-') + " " + (f"({s:.4f})" if s == s else '(-)')
                    for m, s in zip(df[f'{metrics}-mean'], df[f'{metrics}-std'])
                ]
            df = df[mapped_metrics]
            df = df.reindex([f'{model_name}-{unc}' for unc in CLASSIFICATION_UNCERTAINTY])
            index_mapping = {f"{model_name}-{unc}": uncertainty_mapping[unc] for unc in CLASSIFICATION_UNCERTAINTY}
            df = df.rename(index=index_mapping)
            index = pd.MultiIndex.from_tuples([(model_name, unc) for unc in df.index])
            df = df.set_index(index)
            dfs.append(df)

        data_df = pd.concat(dfs)
        init_dir(args.output_dir, clear_original_content=False)
        data_df.to_latex(op.join(args.output_dir, f"tb.result.dataset.{dataset}.tex"),
                         caption=f"Test results on {dataset_mapping[dataset]} "
                                 f"in the format of ``metric mean (standard deviation)''.",
                         label=f"apptb:result.dataset.{dataset}",
                         position='tbh')

    # for regression datasets
    mapped_metrics = [metrics_mapping[m] for m in REGRESSION_METRICS]
    for dataset in REGRESSION_DATASET:
        dfs = list()
        for model_name in MODEL_NAMES:
            result_file = op.join(score_path, f"{model_name}-{dataset}.csv")
            df = pd.read_csv(result_file, index_col='method')
            for metrics in REGRESSION_METRICS:
                df[metrics_mapping[metrics]] = [
                    (f"{m:.4f}" if m == m else '-') + " " + (f"({s:.4f})" if s == s else '(-)')
                    for m, s in zip(df[f'{metrics}-mean'], df[f'{metrics}-std'])
                ]
            df = df[mapped_metrics]
            df = df.reindex([f'{model_name}-{unc}' for unc in REGRESSION_UNCERTAINTY])
            index_mapping = {f"{model_name}-{unc}": uncertainty_mapping[unc] for unc in REGRESSION_UNCERTAINTY}
            df = df.rename(index=index_mapping)
            index = pd.MultiIndex.from_tuples([(model_name, unc) for unc in df.index])
            df = df.set_index(index)
            dfs.append(df)

        data_df = pd.concat(dfs)
        init_dir(args.output_dir, clear_original_content=False)
        data_df.to_latex(op.join(args.output_dir, f"tb.result.dataset.{dataset}.tex"),
                         caption=f"Test results on {dataset_mapping[dataset]} "
                                 f"in the format of ``metric mean (standard deviation)''.",
                         label=f"apptb:result.dataset.{dataset}",
                         position='tbh')
    return None


if __name__ == '__main__':
    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        arguments, = parser.parse_args_into_dataclasses()

    main(arguments)
