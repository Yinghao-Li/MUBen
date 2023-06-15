
import sys
import glob
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional

from muben.utils.io import load_results, init_dir
from muben.utils.macro import (
    DATASET_NAMES,
    MODEL_NAMES,
    UncertaintyMethods,
    FINGERPRINT_FEATURE_TYPES
)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "choices": DATASET_NAMES,
            "help": "A list of dataset names."
        }
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "choices": MODEL_NAMES,
            "help": "A list of model names"
        }
    )
    feature_type: Optional[str] = field(
        default="none",
        metadata={
            "choices": FINGERPRINT_FEATURE_TYPES,
            "help": "Feature type that the DNN model uses."
        }
    )
    result_dir: Optional[str] = field(
        default=".", metadata={"help": "The folder which holds the results."}
    )
    result_seed: Optional[int] = field(
        default=0, metadata={"help": "Seed of the results"}
    )
    n_subsamples: Optional[int] = field(
        default=None, metadata={"help": "Sub-sampling n datapoints for each UQ method."}
    )
    output_dir: Optional[str] = field(
        default="plots", metadata={"help": "Directory to save the plots."}
    )

    def __post_init__(self):
        if self.model_name == "DNN":
            assert self.feature_type != 'none', ValueError("Invalid feature type for DNN!")
            self.model_name = f"{self.model_name}-{self.feature_type}"

        self.none_result_dir = op.join(
            self.result_dir, self.dataset_name, self.model_name,
            UncertaintyMethods.none, f"seed-{self.result_seed}", "preds"
        )
        self.sgld_result_dir = op.join(
            self.result_dir, self.dataset_name, self.model_name,
            UncertaintyMethods.sgld, f"seed-{self.result_seed}", "preds"
        )


def main(args: Arguments):

    # load results
    none_result_paths = glob.glob(op.join(args.none_result_dir, "*"))
    sgld_result_paths = glob.glob(op.join(args.sgld_result_dir, "*"))

    sgld_preds, sgld_variances, lbs, _ = load_results(sgld_result_paths)
    none_preds, none_variances, lbs, _ = load_results(none_result_paths)

    sgld_preds = sgld_preds.reshape(-1)
    sgld_variances = sgld_variances.reshape(-1)

    none_preds = none_preds.reshape(-1)
    none_variances = none_variances.reshape(-1)

    lbs = lbs.reshape(-1)

    # sub-sampling result points
    if args.n_subsamples:
        sampled_ids = np.random.choice(len(lbs), args.n_subsamples)

        sgld_preds = sgld_preds[sampled_ids]
        sgld_variances = sgld_variances[sampled_ids]

        none_preds = none_preds[sampled_ids]
        none_variances = none_variances[sampled_ids]

        lbs = lbs[sampled_ids]

    sgld_abs_error = np.abs(sgld_preds - lbs)
    none_abs_error = np.abs(none_preds - lbs)

    # plot figure
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    ref_x = np.linspace(min(sgld_abs_error.min(), none_abs_error.min()),
                        max(sgld_abs_error.max(), none_abs_error.max()), 100)

    ax.plot(ref_x, ref_x, color='black', linestyle="dashed", alpha=1, linewidth=1, zorder=10, label="y=x")
    ax.plot(ref_x, ref_x * 2, color='black', linestyle="dotted", alpha=1, linewidth=1, zorder=10, label="y=2x")
    ax.plot(ref_x, ref_x * 3, color='black', linestyle="dashdot", alpha=1, linewidth=1, zorder=10, label="y=3x")
    ax.scatter(np.sqrt(sgld_variances), sgld_abs_error, color='goldenrod', marker='^', s=20, alpha=0.6, zorder=100,
               label="Deterministic")
    ax.scatter(np.sqrt(none_variances), none_abs_error, color='royalblue', s=20, alpha=0.6, zorder=100, label="SGLD")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Absolute Prediction Error', fontsize=12)
    ax.set_xlabel('Predicted Standard Deviation', fontsize=12)
    ax.set_xlim(0, ref_x[-1] / 3)
    ax.set_ylim(0, ref_x[-1])
    plt.legend()

    init_dir(args.output_dir, clear_original_content=False)
    # use png in case the number of points to plot is too large
    f_name = f'regression.dev.{args.model_name}.{args.dataset_name}.png'
    fig.savefig(op.join(args.output_dir, f_name), bbox_inches='tight')


if __name__ == '__main__':

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        arguments, = parser.parse_args_into_dataclasses()

    main(args=arguments)
