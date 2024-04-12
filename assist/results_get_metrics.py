"""
# Author: Yinghao Li
# Modified: April 11th, 2024
# ---------------------------------------
# Description: Calculate metrics of UQ methods from the saved results.
"""

import sys
import glob
import logging
import os.path as osp

import numpy as np
import pandas as pd

from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from muben.utils.io import set_logging, init_dir, load_results
from muben.utils.macro import (
    DATASET_NAMES,
    CLASSIFICATION_DATASET,
    REGRESSION_DATASET,
    MODEL_NAMES,
    UncertaintyMethods,
    FINGERPRINT_FEATURE_TYPES,
)
from muben.utils.metrics import classification_metrics, regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    dataset_names: Optional[str] = field(default=None, metadata={"nargs": "*", "help": "A list of dataset names."})
    model_name: Optional[str] = field(default=None, metadata={"choices": MODEL_NAMES, "help": "model name"})
    feature_type: Optional[str] = field(
        default="none",
        metadata={
            "choices": FINGERPRINT_FEATURE_TYPES,
            "help": "Feature type that the DNN model uses.",
        },
    )
    uncertainty_methods: Optional[str] = field(
        default=None,
        metadata={
            "nargs": "*",
            "help": "A list of uncertainty methods of which you want to calculate the metrics.",
        },
    )
    result_folder: Optional[str] = field(default=".", metadata={"help": "The folder which holds the results."})
    report_folder: Optional[str] = field(default=".", metadata={"help": "The folder to save the report."})
    test_subset_ids_file_name: Optional[str] = field(
        default=None, metadata={"help": "The file name of the test subset ids, no suffix."}
    )
    log_path: Optional[str] = field(default=None, metadata={"help": "Path to save the log file."})
    overwrite_output: Optional[bool] = field(default=False, metadata={"help": "Whether overwrite existing outputs."})
    result_seeds: Optional[int] = field(
        default=None,
        metadata={
            "nargs": "*",
            "help": "the seeds the models are trained with. This argument is used to locate results.",
        },
    )

    def __post_init__(self):
        if self.model_name == "DNN":
            assert self.feature_type != "none", ValueError("Invalid feature type for DNN!")
            self.model_name = f"{self.model_name}-{self.feature_type}"

        if self.result_seeds is None:
            self.result_seeds: list[int] = [0, 1, 2]
        elif isinstance(self.result_seeds, int):
            self.result_seeds: list[int] = [self.result_seeds]

        if self.dataset_names is None:
            self.dataset_names: list[str] = DATASET_NAMES
        elif self.dataset_names == "classification":
            self.dataset_names: list[str] = CLASSIFICATION_DATASET
        elif self.dataset_names == "regression":
            self.dataset_names: list[str] = REGRESSION_DATASET
        elif isinstance(self.dataset_names, str):
            self.dataset_names: list[str] = [self.dataset_names]

        if self.uncertainty_methods is None:
            self.uncertainty_methods: list[str] = UncertaintyMethods.options()
        elif isinstance(self.uncertainty_methods, str):
            self.uncertainty_methods: list[str] = [self.uncertainty_methods]

        if not self.test_subset_ids_file_name:
            self.test_subset_ids_file_name = "preds"
        elif self.test_subset_ids_file_name.endswith(".json"):
            self.test_subset_ids_file_name = self.test_subset_ids_file_name[:-5]


def main(args: Arguments):
    for dataset_name in args.dataset_names:
        logger.info(f"Processing dataset: {dataset_name}...")

        uncertainty_results = dict()
        for uncertainty_method in args.uncertainty_methods:
            logger.info(f"Processing UQ method: {uncertainty_method}...")

            result_dir = osp.join(args.result_folder, dataset_name, args.model_name, uncertainty_method)

            if uncertainty_method != UncertaintyMethods.ensembles:
                results_for_seeds = list()

                for seed in args.result_seeds:
                    seeded_result_dir = osp.join(result_dir, f"seed-{seed}")
                    test_result_paths = glob.glob(
                        osp.join(seeded_result_dir, f"preds-{args.test_subset_ids_file_name}", "*.pt")
                    )

                    if not test_result_paths:
                        continue

                    preds, variances, lbs, masks = load_results(test_result_paths)

                    if variances is not None:  # regression
                        metrics = regression_metrics(preds, variances, lbs, masks)
                    else:  # classification
                        metrics = classification_metrics(preds, lbs, masks)

                    result_dict = {k: v["macro-avg"] for k, v in metrics.items()}
                    results_for_seeds.append(result_dict)

                if not results_for_seeds:
                    logger.warning(
                        f"Directory {result_dir} does not contain any model prediction! "
                        f"Will skip metric logging for {uncertainty_method}"
                    )
                    continue

                results_aggr = aggregate_seeded_results(results_for_seeds)
                uncertainty_results[uncertainty_method] = results_aggr

            else:
                seeded_result_dirs = glob.glob(osp.join(result_dir, "*"))
                test_result_paths = [
                    osp.join(sid, f"preds-{args.test_subset_ids_file_name}", "0.pt") for sid in seeded_result_dirs
                ]

                if not seeded_result_dirs:
                    logger.warning(
                        f"Directory {result_dir} does not contain any model prediction! "
                        f"Will skip metric logging for {uncertainty_method}"
                    )
                    continue

                preds, variances, lbs, masks = load_results(test_result_paths)

                if variances is not None:  # regression
                    metrics = regression_metrics(preds, variances, lbs, masks)
                else:  # classification
                    metrics = classification_metrics(preds, lbs, masks)

                result_dict = {k: v["macro-avg"] for k, v in metrics.items()}
                results_aggr = {k: {"mean": v, "std": np.NaN} for k, v in result_dict.items()}

                uncertainty_results[uncertainty_method] = results_aggr

        report_dir = args.report_folder
        if args.test_subset_ids_file_name:
            report_dir = osp.join(report_dir, args.test_subset_ids_file_name)
        save_results(uncertainty_results, report_dir, args.model_name, dataset_name)

    return None


def aggregate_seeded_results(results_for_seeds: list[dict[str, float]]):
    assert results_for_seeds

    results_aggr = {
        metric: [r.get(metric, np.NaN) for r in results_for_seeds] for metric in list(results_for_seeds[0].keys())
    }
    for k in results_aggr:
        mean = np.nanmean(results_aggr[k])
        std = np.nanstd(results_aggr[k])
        results_aggr[k] = {"mean": mean, "std": std}

    return results_aggr


def save_results(results, report_dir, model_name, dataset_name):
    if not results:
        logger.warning("Result dict is empty. No results is saved!")
        return None

    uncertainty_names = list(results.keys())
    metrics = list(list(results.values())[0].keys())
    columns_headers = ["method"] + [f"{metric}-mean" for metric in metrics] + [f"{metric}-std" for metric in metrics]
    columns = {k: list() for k in columns_headers}
    columns["method"] = [f"{model_name}-{un}" for un in uncertainty_names]
    for uncertainty in uncertainty_names:
        metric_dict = results[uncertainty]
        for metric in metrics:
            value = metric_dict.get(metric, None)
            if value:
                columns[f"{metric}-mean"].append(value["mean"])
                columns[f"{metric}-std"].append(value["std"])
            else:
                columns[f"{metric}-mean"].append(np.NaN)
                columns[f"{metric}-std"].append(np.NaN)

    df = pd.DataFrame(columns)

    init_dir(report_dir, clear_original_content=False)
    df.to_csv(osp.join(report_dir, f"{model_name}-{dataset_name}.csv"))
    return None


if __name__ == "__main__":
    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (arguments,) = parser.parse_json_file(osp.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (arguments,) = parser.parse_yaml_file(osp.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    set_logging()
    main(args=arguments)
