"""
# Author: Yinghao Li
# Modified: September 28th, 2023
# ---------------------------------------
# Description: compute and save the ranks of UQ methods
"""

import sys
import glob
import os.path as op
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional
from muben.utils.macro import CLASSIFICATION_DATASET, REGRESSION_DATASET, FINGERPRINT_FEATURE_TYPES, MODEL_NAMES
from muben.utils.io import init_dir


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
CLASSIFICATION_METRICS = ["roc-auc", "prc-auc", "ece", "mce", "nll", "brier"]
REGRESSION_METRICS = ["rmse", "mae", "nll", "ce"]
LARGER_BETTER_LOOKUP = {
    "roc-auc": True,
    "prc-auc": True,
    "ece": False,
    "mce": False,
    "nll": False,
    "brier": False,
    "rmse": False,
    "mae": False,
    "ce": False,
}


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    model_names: Optional[str] = field(default=None, metadata={"nargs": "*", "help": "A list of model names."})
    feature_type: Optional[str] = field(
        default="rdkit",
        metadata={
            "choices": FINGERPRINT_FEATURE_TYPES,
            "help": "Feature type that the DNN model uses.",
        },
    )
    result_dir: Optional[str] = field(default="./output/primary/RESULTS", metadata={"help": "Directory the scores."})
    output_dir: Optional[str] = field(default="./output/primary/RESULTS", metadata={"help": "Where to save the ranks."})

    def __post_init__(self):
        if self.model_names is None:
            self.model_names = MODEL_NAMES
        elif isinstance(self.model_names, str):
            self.model_names: list[str] = [self.model_names]

        for idx, model_name in enumerate(self.model_names):
            if model_name == "DNN":
                assert self.feature_type != "none", ValueError("Invalid feature type for DNN!")
                self.model_names[idx] = f"DNN-{self.feature_type}"


def get_ranks(values, smaller_is_better=True):
    arr = np.array(values)
    if smaller_is_better:
        order = arr.argsort()
    else:
        order = arr.argsort()[::-1]
    rnks = order.argsort()
    return rnks + 1


def main(args: Arguments):
    score_path = op.join(args.result_dir, "scores")
    result_files = list(glob.glob(op.join(score_path, "*.csv")))

    # Store all results in a data structure
    cla_dt_md_unc_mtr = dict()
    reg_dt_md_unc_mtr = dict()
    for result_file in result_files:
        file_name = op.basename(result_file)
        model_name = "-".join(file_name.split("-")[:-1])
        if model_name not in args.model_names:
            continue
        dataset_name = file_name.split("-")[-1].split(".")[0]

        if dataset_name in CLASSIFICATION_DATASET:
            dataset_model_uncertainty_metric = cla_dt_md_unc_mtr
        elif dataset_name in REGRESSION_DATASET:
            dataset_model_uncertainty_metric = reg_dt_md_unc_mtr
        else:
            print("Invalid dataset name!")
            continue

        if dataset_name not in dataset_model_uncertainty_metric.keys():
            dataset_model_uncertainty_metric[dataset_name] = dict()

        df = pd.read_csv(result_file)

        uncertainty_methods = df.method
        uncertainty_methods = [um.split("-")[-1] for um in uncertainty_methods]

        metric_means = [k for k in df.keys() if k.endswith("-mean")]
        metric_names = [m[:-5] for m in metric_means]
        column_headers = ["method"] + metric_means

        uncertainty_metric = dict()
        for item in df[column_headers].to_numpy():
            uncertainty_method = item[0].split("-")[-1]
            metric_values = item[1:]
            assert len(metric_values) == len(metric_names)

            uncertainty_metric[uncertainty_method] = {n: v for n, v in zip(metric_names, metric_values)}

        dataset_model_uncertainty_metric[dataset_name][model_name] = uncertainty_metric

    cla_fl_dt_md_unc_mtr = pd.json_normalize(cla_dt_md_unc_mtr, sep="_").to_dict()
    reg_fl_dt_md_unc_mtr = pd.json_normalize(reg_dt_md_unc_mtr, sep="_").to_dict()

    cla_dt_fl_md_unc_mtr = {
        dt: pd.json_normalize(cla_md_unc_mtr, sep="_").to_dict() for dt, cla_md_unc_mtr in cla_dt_md_unc_mtr.items()
    }
    reg_dt_fl_md_unc_mtr = {
        dt: pd.json_normalize(reg_md_unc_mtr, sep="_").to_dict() for dt, reg_md_unc_mtr in reg_dt_md_unc_mtr.items()
    }

    # calculate ranks
    classification_datasets = list(cla_dt_fl_md_unc_mtr.keys())
    regression_datasets = list(reg_dt_fl_md_unc_mtr.keys())

    cla_metric_ranks = {dt: {mtr: dict() for mtr in CLASSIFICATION_METRICS} for dt in classification_datasets}
    reg_metric_ranks = {dt: {mtr: dict() for mtr in REGRESSION_METRICS} for dt in regression_datasets}

    cla_metric_ranks_mean = {mtr: dict() for mtr in CLASSIFICATION_METRICS}
    reg_metric_ranks_mean = {mtr: dict() for mtr in REGRESSION_METRICS}

    # classification
    for dt, mtr_vals in cla_metric_ranks.items():
        for mtr in mtr_vals:
            md_unc_scores = {
                "_".join(k.split("_")[1:-1]): val[0]
                for k, val in cla_fl_dt_md_unc_mtr.items()
                if k.startswith(dt) and k.endswith(mtr)
            }
            ranks = get_ranks(
                list(md_unc_scores.values()),
                smaller_is_better=not LARGER_BETTER_LOOKUP[mtr],
            )
            md_unc_ranks = {k: r for k, r in zip(md_unc_scores, ranks)}
            mtr_vals[mtr] = md_unc_ranks

            dict1 = cla_metric_ranks_mean[mtr]
            dict2 = md_unc_ranks
            cla_metric_ranks_mean[mtr] = {
                i: dict1.get(i, 0) + dict2.get(i, 0) / len(classification_datasets) for i in md_unc_ranks.keys()
            }

    # regression
    for dt, mtr_vals in reg_metric_ranks.items():
        for mtr in mtr_vals:
            md_unc_scores = {
                "_".join(k.split("_")[1:-1]): val[0]
                for k, val in reg_fl_dt_md_unc_mtr.items()
                if k.startswith(dt) and k.endswith(mtr)
            }
            ranks = get_ranks(
                list(md_unc_scores.values()),
                smaller_is_better=not LARGER_BETTER_LOOKUP[mtr],
            )
            md_unc_ranks = {k: r for k, r in zip(md_unc_scores, ranks)}
            mtr_vals[mtr] = md_unc_ranks

            dict1 = reg_metric_ranks_mean[mtr]
            dict2 = md_unc_ranks
            reg_metric_ranks_mean[mtr] = {
                i: dict1.get(i, 0) + dict2.get(i, 0) / len(regression_datasets) for i in md_unc_ranks.keys()
            }

    # save ranks
    output_dir = op.join(args.result_dir, "ranks")
    init_dir(output_dir)

    for dataset in classification_datasets:
        dataset_metric_ranks = cla_metric_ranks[dataset]
        df = pd.DataFrame(dataset_metric_ranks)
        df.to_csv(op.join(output_dir, f"{dataset}.csv"))

    for dataset in regression_datasets:
        dataset_metric_ranks = reg_metric_ranks[dataset]
        df = pd.DataFrame(dataset_metric_ranks)
        df.to_csv(op.join(output_dir, f"{dataset}.csv"))

    df = pd.DataFrame(cla_metric_ranks_mean)
    df.to_csv(op.join(output_dir, "mean_classification.csv"))

    df = pd.DataFrame(reg_metric_ranks_mean)
    df.to_csv(op.join(output_dir, "mean_regression.csv"))

    # get reciprocal ranks
    cla_metric_rrs = {dt: {mtr: dict() for mtr in CLASSIFICATION_METRICS} for dt in classification_datasets}
    reg_metric_rrs = {dt: {mtr: dict() for mtr in REGRESSION_METRICS} for dt in regression_datasets}

    cla_metric_rrs_mean = {mtr: dict() for mtr in CLASSIFICATION_METRICS}
    reg_metric_rrs_mean = {mtr: dict() for mtr in REGRESSION_METRICS}

    for dt, mtr_vals in cla_metric_rrs.items():
        for mtr in mtr_vals:
            md_unc_scores = {
                "_".join(k.split("_")[1:-1]): val[0]
                for k, val in cla_fl_dt_md_unc_mtr.items()
                if k.startswith(dt) and k.endswith(mtr)
            }
            rrs = get_ranks(
                list(md_unc_scores.values()),
                smaller_is_better=not LARGER_BETTER_LOOKUP[mtr],
            )
            md_unc_rrs = {k: 1 / r for k, r in zip(md_unc_scores, rrs)}
            mtr_vals[mtr] = md_unc_rrs

            dict1 = cla_metric_rrs_mean[mtr]
            dict2 = md_unc_rrs
            cla_metric_rrs_mean[mtr] = {
                i: dict1.get(i, 0) + dict2.get(i, 0) / len(classification_datasets) for i in md_unc_rrs.keys()
            }

    for dt, mtr_vals in reg_metric_rrs.items():
        for mtr in mtr_vals:
            md_unc_scores = {
                "_".join(k.split("_")[1:-1]): val[0]
                for k, val in reg_fl_dt_md_unc_mtr.items()
                if k.startswith(dt) and k.endswith(mtr)
            }
            rrs = get_ranks(
                list(md_unc_scores.values()),
                smaller_is_better=not LARGER_BETTER_LOOKUP[mtr],
            )
            md_unc_rrs = {k: 1 / r for k, r in zip(md_unc_scores, rrs)}
            mtr_vals[mtr] = md_unc_rrs

            dict1 = reg_metric_rrs_mean[mtr]
            dict2 = md_unc_rrs
            reg_metric_rrs_mean[mtr] = {
                i: dict1.get(i, 0) + dict2.get(i, 0) / len(regression_datasets) for i in md_unc_rrs.keys()
            }

    # save mrr results
    output_dir = op.join(args.result_dir, "mrrs")
    init_dir(output_dir)

    for dataset in classification_datasets:
        dataset_metric_mrrs = cla_metric_rrs[dataset]
        df = pd.DataFrame(dataset_metric_mrrs)
        df.to_csv(op.join(output_dir, f"{dataset}-mrr.csv"))

    for dataset in regression_datasets:
        dataset_metric_mrrs = reg_metric_rrs[dataset]
        df = pd.DataFrame(dataset_metric_mrrs)
        df.to_csv(op.join(output_dir, f"{dataset}-mrr.csv"))

    df = pd.DataFrame(cla_metric_rrs_mean)
    df.to_csv(op.join(output_dir, "mrr_classification.csv"))

    df = pd.DataFrame(reg_metric_rrs_mean)
    df.to_csv(op.join(output_dir, "mrr_regression.csv"))

    return None


if __name__ == "__main__":
    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    main(arguments)
