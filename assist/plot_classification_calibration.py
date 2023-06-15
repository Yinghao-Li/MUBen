
import sys
import glob
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
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
        self.focal_result_dir = op.join(
            self.result_dir, self.dataset_name, self.model_name,
            UncertaintyMethods.focal, f"seed-{self.result_seed}", "preds"
        )
        self.ts_result_dir = op.join(
            self.result_dir, self.dataset_name, self.model_name,
            UncertaintyMethods.focal, f"seed-{self.result_seed}", "preds"
        )


def main(args: Arguments):

    # load results
    none_result_paths = glob.glob(op.join(args.none_result_dir, "*"))
    focal_result_paths = glob.glob(op.join(args.focal_result_dir, "*"))
    ts_result_paths = glob.glob(op.join(args.ts_result_dir, "*"))

    none_preds, _, lbs, masks = load_results(none_result_paths)
    focal_preds, _, lbs, masks = load_results(focal_result_paths)
    ts_preds, _, lbs, masks = load_results(ts_result_paths)

    bool_masks = masks.astype(bool)

    none_preds = none_preds[bool_masks]
    focal_preds = focal_preds[bool_masks]
    ts_preds = ts_preds[bool_masks]
    lbs = lbs[bool_masks]

    # calculate calibration curve
    y_none, x_none = calibration_curve(lbs, none_preds, n_bins=15)
    y_focal, x_focal = calibration_curve(lbs, focal_preds, n_bins=15)
    y_ts, x_ts = calibration_curve(lbs, ts_preds, n_bins=15)

    # plot figure
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    ref_x = np.linspace(0, 1, 100)
    ax.plot(ref_x, ref_x, color='black', alpha=1, linewidth=1, zorder=10, label="y=x")

    ax.plot(x_none, y_none, color='goldenrod', alpha=0.6, linewidth=2, zorder=10)
    ax.scatter(x_none, y_none, color='goldenrod', marker='^', s=40, alpha=0.8, zorder=100, label="Deterministic")

    ax.plot(x_ts, y_ts, color='darkgray', alpha=0.6, linewidth=2, zorder=10)
    ax.scatter(x_ts, y_ts, color='darkgray', marker='v', s=40, alpha=0.8, zorder=100, label="Temperature Scaling")

    ax.plot(x_focal, y_focal, color='royalblue', alpha=0.6, linewidth=2, zorder=10)
    ax.scatter(x_focal, y_focal, color='royalblue', s=40, alpha=0.8, zorder=100, label="Focal Loss")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Probability', fontsize=12)
    plt.legend()

    init_dir(args.output_dir, clear_original_content=False)
    f_name = f'classification.cal.{args.model_name}.{args.dataset_name}.pdf'
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
