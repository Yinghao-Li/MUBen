"""
# Author: Yinghao Li
# Modified: April 11th, 2024
# ---------------------------------------
# Description: Plot classification calibration curves.
"""

import sys
import pandas as pd
import os.path as osp
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.ticker import StrMethodFormatter

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

from muben.utils.io import load_results, init_dir
from muben.utils.macro import (
    DATASET_NAMES,
    MODEL_NAMES,
    UncertaintyMethods,
    FINGERPRINT_FEATURE_TYPES,
    REGRESSION_METRICS,
    METRICS_MAPPING,
)

REGRESSION_UNCERTAINTY = ["none", "MCDropout", "SWAG", "BBP", "SGLD", "DeepEnsembles"]
UQ_MAPPING = {
    "none": "Deterministic",
    "TemperatureScaling": "Temperature",
    "FocalLoss": "Focal Loss",
    "MCDropout": "MC Dropout",
    "SWAG": "SWAG",
    "BBP": "BBP",
    "SGLD": "SGLD",
    "DeepEnsembles": "Ensembles",
}


class MetricVector:
    def __init__(self, features: dict[str, list[str]]):
        self.feature_names = list(features.keys())

        for feature_name, options in features.items():
            setattr(self, feature_name, options)
            setattr(self, f"{feature_name}2id", {f: i for i, f in enumerate(options)})

        feature_dims = [len(options) for options in features.values()]
        self.metrics = np.full(feature_dims, np.nan)

    @property
    def shape(self):
        return self.metrics.shape

    def get(self, feature_ops: dict[str, str]):
        feature_indices = [getattr(self, f"{fn}2id")[feature_ops[fn]] for fn in self.feature_names]
        return self.metrics[tuple(feature_indices)]

    def set(self, feature_ops: dict[str, str], metric: float):
        feature_indices = [getattr(self, f"{fn}2id")[feature_ops[fn]] for fn in self.feature_names]
        self.metrics[tuple(feature_indices)] = metric

    def apply_func_along_features(self, func, feature_names: list[str], **kwargs):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        feature_ids = [self.feature_names.index(feature_name) for feature_name in feature_names]

        new_container = copy.deepcopy(self)
        new_container.metrics = func(self.metrics, axis=tuple(feature_ids), **kwargs)

        for feature_name in feature_names:
            delattr(new_container, feature_name)
            delattr(new_container, f"{feature_name}2id")
        new_container.feature_names = [fn for fn in self.feature_names if fn not in feature_names]

        return new_container

    def __str__(self) -> str:
        features_dict = {fn: getattr(self, fn) for fn in self.feature_names}
        txt = f"[features]:\n{features_dict}\n\n"
        txt += f"[metrics]:\n{self.metrics}"
        return txt

    def __repr__(self) -> str:
        features_dict = {fn: getattr(self, fn) for fn in self.feature_names}
        txt = f"[features]:\n{features_dict.__repr__()}\n\n"
        txt += f"[metrics]:\n{self.metrics.__repr__()}"
        return txt


def plot_by_metric_backbone(model_names, metric_vecs_w_backbone, metric, output_dir):
    model_curves = []
    for model in model_names:
        model_curve = []
        for metric_vec in metric_vecs_w_backbone:
            model_curve.append(metric_vec.get({"backbone": model, "metric": metric}))
        model_curves.append(model_curve)

    # plot figure
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    palette = sns.color_palette("coolwarm_r", len(model_curves))
    ax.set_prop_cycle(cycler("color", palette))

    marker = iter(["o", "^", "x", "*", "v", "X"])
    for model_curve, model_name in zip(model_curves, model_names):
        ax.plot(
            range(1, len(model_curve) + 1),
            model_curve,
            alpha=0.95,
            linewidth=2,
            zorder=10,
            label=f"_{model_name}",
        )
        ax.scatter(
            range(1, len(model_curve) + 1),
            model_curve,
            marker=next(marker),
            s=60,
            alpha=0.95,
            zorder=100,
            label=model_name,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)

    # ax.set_xlabel("Test Group ID", fontsize=12)
    # ax.set_ylabel(f"{METRICS_MAPPING[metric]}", fontsize=12)
    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.3f}"))

    if metric == "nll":
        plt.legend(prop={"size": 12})

    init_dir(output_dir, clear_original_content=False)
    f_name = f"test-distr-uqavg-{metric}.pdf"
    fig.savefig(osp.join(output_dir, f_name), bbox_inches="tight")


def plot_by_metric_uq(uq_names, metric_vecs_w_uq, metric, output_dir):
    uq_curves = []
    for uq in uq_names:
        uq_curve = []
        for metric_vec in metric_vecs_w_uq:
            uq_curve.append(metric_vec.get({"uq": uq, "metric": metric}))
        uq_curves.append(uq_curve)

    # plot figure
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    palette = sns.color_palette("coolwarm_r", len(uq_curves))
    ax.set_prop_cycle(cycler("color", palette))

    marker = iter(["o", "^", "x", "*", "v", "X"])
    for uq_curve, uq_name in zip(uq_curves, uq_names):
        ax.plot(
            range(1, len(uq_curve) + 1),
            uq_curve,
            alpha=0.95,
            linewidth=2,
            zorder=10,
            label=f"_{UQ_MAPPING[uq_name]}",
        )
        ax.scatter(
            range(1, len(uq_curve) + 1),
            uq_curve,
            marker=next(marker),
            s=60,
            alpha=0.95,
            zorder=100,
            label=UQ_MAPPING[uq_name],
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)

    # ax.set_xlabel("Test Group ID", fontsize=12)
    # ax.set_ylabel(f"{METRICS_MAPPING[metric]}", fontsize=12)
    # ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.3f}"))

    if metric == "nll":
        plt.legend(prop={"size": 12})

    init_dir(output_dir, clear_original_content=False)
    f_name = f"test-distr-backboneavg-{metric}.pdf"
    fig.savefig(osp.join(output_dir, f_name), bbox_inches="tight")


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"choices": DATASET_NAMES, "help": "Dataset Name"},
    )
    model_names: Optional[str] = field(
        default=None, metadata={"nargs": "*", "choices": MODEL_NAMES, "help": "A list of model names"}
    )
    feature_type: Optional[str] = field(
        default="none",
        metadata={
            "choices": FINGERPRINT_FEATURE_TYPES,
            "help": "Feature type that the DNN model uses.",
        },
    )
    report_dir: Optional[str] = field(
        default="./reports/distribution", metadata={"help": "The folder which holds the results."}
    )
    test_subset_ids_file_base_name: Optional[str] = field(
        default=None, metadata={"help": "The file name of the test subset ids, no suffix indices."}
    )
    test_subset_ids_file_ids: Optional[int] = field(
        default=None, metadata={"nargs": "*", "help": "The file name of the test subset ids suffix."}
    )
    output_dir: Optional[str] = field(default="plots", metadata={"help": "Directory to save the plots."})

    average_over_backbone: Optional[bool] = field(default=False, metadata={"help": "Average over base models."})
    average_over_uq: Optional[bool] = field(
        default=False, metadata={"help": "Average over uncertainty quantification methods."}
    )

    def __post_init__(self):
        if isinstance(self.model_names, str):
            self.model_names = [self.model_names]

        if self.test_subset_ids_file_ids is None:
            self.test_subset_ids_file_ids = [4, 3, 2, 1, 0]
        elif isinstance(self.test_subset_ids_file_ids, int):
            raise ValueError("Number of test subset ids files must be a list of integers (N >= 2).")


def main(args: Arguments):
    # load results
    metric_vecs_backbone = list()
    metric_vecs_uq = list()

    for test_subset_ids_file_id in args.test_subset_ids_file_ids:
        report_dir = osp.join(args.report_dir, f"{args.test_subset_ids_file_base_name}-{test_subset_ids_file_id}")

        metric_vec = MetricVector(
            features={
                "backbone": args.model_names,
                "uq": REGRESSION_UNCERTAINTY,
                "metric": REGRESSION_METRICS,
            }
        )
        for model_name in args.model_names:
            if model_name == "DNN":
                model_name_ = f"{model_name}-{args.feature_type}"
            else:
                model_name_ = model_name
            file_name = f"{model_name_}-{args.dataset_name}.csv"

            file_path = osp.join(report_dir, file_name)

            df = pd.read_csv(file_path, index_col="method")
            df = df.reindex([f"{model_name_}-{unc}" for unc in REGRESSION_UNCERTAINTY])

            for metrics in REGRESSION_METRICS:
                df[metrics] = [m for m in df[f"{metrics}-mean"]]

            df = df[REGRESSION_METRICS]

            for uncertainty_method in REGRESSION_UNCERTAINTY:
                for metric in REGRESSION_METRICS:
                    metric_vec.set(
                        {"backbone": model_name, "uq": uncertainty_method, "metric": metric},
                        df.loc[f"{model_name_}-{uncertainty_method}"][metric],
                    )

            metric_vec_backbone = metric_vec.apply_func_along_features(np.mean, "uq")
            metric_vec_uq = metric_vec.apply_func_along_features(np.mean, "backbone")

        metric_vecs_backbone.append(metric_vec_backbone)
        metric_vecs_uq.append(metric_vec_uq)

    for metric in REGRESSION_METRICS:
        plot_by_metric_backbone(args.model_names, metric_vecs_backbone, metric, args.output_dir)
        plot_by_metric_uq(REGRESSION_UNCERTAINTY, metric_vecs_uq, metric, args.output_dir)


if __name__ == "__main__":
    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (arguments,) = parser.parse_json_file(osp.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        (arguments,) = parser.parse_yaml_file(osp.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    main(args=arguments)
