"""
# Author: Yinghao Li
# Modified: April 11th, 2024
# ---------------------------------------
# Description: automatically generate the result tables in the appendix.
"""

import sys
import os.path as op
import pandas as pd
from muben.utils.macro import CLASSIFICATION_DATASET, REGRESSION_DATASET
from muben.utils.io import init_dir

from dataclasses import dataclass, field
from typing import Optional

from muben.utils.argparser import ArgumentParser
from muben.utils.macro import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    MODEL_NAMES,
    DATASET_NAMES,
    FINGERPRINT_FEATURE_TYPES,
    dataset_mapping,
    METRICS_MAPPING,
)


model_mapping = {
    "DNN-rdkit": "DNN",
    "ChemBERTa": "ChemBERTa",
    "GROVER": "GROVER",
    "Uni-Mol": "Uni-Mol",
    "TorchMD-NET": "TorchMD-NET",
    "GIN": "GIN",
}
CLASSIFICATION_UNCERTAINTY = [
    "none",
    "TemperatureScaling",
    "FocalLoss",
    "MCDropout",
    "SWAG",
    "BBP",
    "SGLD",
    "DeepEnsembles",
]
REGRESSION_UNCERTAINTY = ["none", "MCDropout", "SWAG", "BBP", "SGLD", "DeepEnsembles"]
uncertainty_mapping = {
    "none": "Deterministic",
    "TemperatureScaling": "Temperature",
    "FocalLoss": "Focal Loss",
    "MCDropout": "MC Dropout",
    "SWAG": "SWAG",
    "BBP": "BBP",
    "SGLD": "SGLD",
    "DeepEnsembles": "Ensembles",
}


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    result_dir: Optional[str] = field(
        default="./output/RESULTS/",
        metadata={"help": "The folder which holds the results."},
    )
    output_dir: Optional[str] = field(default="latex", metadata={"help": "Directory to save the latex tables."})
    dataset_names: Optional[str] = field(default=None, metadata={"nargs": "*", "help": "A list of dataset names."})
    model_names: Optional[str] = field(default=None, metadata={"nargs": "*", "help": "A list of model names."})
    feature_type: Optional[str] = field(
        default="rdkit",
        metadata={
            "choices": FINGERPRINT_FEATURE_TYPES,
            "help": "Feature type that the DNN model uses.",
        },
    )
    filename_prefix: Optional[str] = field(default="tb3.result.dataset", metadata={"help": "Prefix of the file names."})
    label_prefix: Optional[str] = field(
        default="apptb:result.dataset", metadata={"help": "Prefix of the table labels."}
    )

    def __post_init__(self):
        if self.model_names is None:
            self.model_names = MODEL_NAMES
        elif isinstance(self.model_names, str):
            self.model_names: list[str] = [self.model_names]

        for idx, model_name in enumerate(self.model_names):
            if model_name == "DNN":
                assert self.feature_type != "none", ValueError("Invalid feature type for DNN!")
                self.model_names[idx] = f"DNN-{self.feature_type}"

        if self.dataset_names is None:
            self.dataset_names: list[str] = DATASET_NAMES
        elif self.dataset_names == "classification":
            self.dataset_names: list[str] = CLASSIFICATION_DATASET
        elif self.dataset_names == "regression":
            self.dataset_names: list[str] = REGRESSION_DATASET
        elif isinstance(self.dataset_names, str):
            self.dataset_names: list[str] = [self.dataset_names]


def main(args: Arguments):
    score_path = op.join(args.result_dir, "scores")

    # for classification datasets
    mapped_metrics = [METRICS_MAPPING[m] for m in CLASSIFICATION_METRICS]
    for dataset in CLASSIFICATION_DATASET:
        if dataset not in args.dataset_names:
            continue
        dfs = list()
        for model_name in args.model_names:
            result_file = op.join(score_path, f"{model_name}-{dataset}.csv")
            df = pd.read_csv(result_file, index_col="method")
            for metrics in CLASSIFICATION_METRICS:
                try:
                    df[METRICS_MAPPING[metrics]] = [
                        (f"{m:.4f}" if m == m else "-") + " $\\pm$ " + (f"{s:.4f}" if s == s else "-")
                        for m, s in zip(df[f"{metrics}-mean"], df[f"{metrics}-std"])
                    ]
                except KeyError:
                    df[METRICS_MAPPING[metrics]] = ["- $\\pm$ -" for _ in df[f"roc-auc-mean"]]
            df = df[mapped_metrics]
            df = df.reindex([f"{model_name}-{unc}" for unc in CLASSIFICATION_UNCERTAINTY])
            index_mapping = {f"{model_name}-{unc}": uncertainty_mapping[unc] for unc in CLASSIFICATION_UNCERTAINTY}
            df = df.rename(index=index_mapping)
            index = pd.MultiIndex.from_tuples([(model_name, unc) for unc in df.index])
            df = df.set_index(index)
            dfs.append(df)

        data_df = pd.concat(dfs)
        init_dir(args.output_dir, clear_original_content=False)
        data_df.to_latex(
            op.join(args.output_dir, f"{args.filename_prefix}.{dataset}.tex"),
            caption=f"Test results on {dataset_mapping[dataset]} "
            f"in the format of ``metric mean $\\pm$ standard deviation''.",
            label=f"{args.label_prefix}.{dataset}",
            position="tbh",
        )

    # for regression datasets
    mapped_metrics = [METRICS_MAPPING[m] for m in REGRESSION_METRICS]
    for dataset in REGRESSION_DATASET:
        if dataset not in args.dataset_names:
            continue
        dfs = list()
        for model_name in args.model_names:
            result_file = op.join(score_path, f"{model_name}-{dataset}.csv")
            df = pd.read_csv(result_file, index_col="method")
            for metrics in REGRESSION_METRICS:
                try:
                    df[METRICS_MAPPING[metrics]] = [
                        (f"{m:.4f}" if m == m else "-") + " $\\pm$ " + (f"{s:.4f}" if s == s else "-")
                        for m, s in zip(df[f"{metrics}-mean"], df[f"{metrics}-std"])
                    ]
                except KeyError:
                    df[METRICS_MAPPING[metrics]] = ["- $\\pm$ -" for _ in df[f"rmse-mean"]]
            df = df[mapped_metrics]
            df = df.reindex([f"{model_name}-{unc}" for unc in REGRESSION_UNCERTAINTY])
            index_mapping = {f"{model_name}-{unc}": uncertainty_mapping[unc] for unc in REGRESSION_UNCERTAINTY}
            df = df.rename(index=index_mapping)
            index = pd.MultiIndex.from_tuples([(model_name, unc) for unc in df.index])
            df = df.set_index(index)
            dfs.append(df)

        data_df = pd.concat(dfs)
        init_dir(args.output_dir, clear_original_content=False)
        data_df.to_latex(
            op.join(args.output_dir, f"{args.filename_prefix}.{dataset}.tex"),
            caption=f"Test results on {dataset_mapping[dataset]} "
            f"in the format of ``metric mean $\\pm$ standard deviation''.",
            label=f"{args.label_prefix}.{dataset}",
            position="tbh",
        )
    return None


if __name__ == "__main__":
    # --- set up arguments ---
    parser = ArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    main(arguments)
