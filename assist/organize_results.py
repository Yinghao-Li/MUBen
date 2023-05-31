"""
Run the basic model and training process
"""

import sys
import glob
import logging
import os.path as op

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from torchmetrics.functional.classification import binary_calibration_error
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    auc,
)

from mubench.utils.io import set_logging
from mubench.utils.macro import (
    DATASET_NAMES,
    MODEL_NAMES,
    UncertaintyMethods,
    FINGERPRINT_FEATURE_TYPES
)


logger = logging.getLogger(__name__)


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
    result_folder: Optional[str] = field(
        default=".", metadata={"help": "The folder which holds the results."}
    )
    log_path: Optional[str] = field(
        default=None, metadata={"help": "Path to save the log file."}
    )
    overwrite_output: Optional[bool] = field(
        default=False, metadata={'help': 'Whether overwrite existing outputs.'}
    )
    result_seeds: Optional[int] = field(
        default=None, metadata={
            "nargs": "*",
            "help": "the seeds the models are trained with."
        }
    )

    def __post_init__(self):
        if self.model_name == "DNN":
            assert self.feature_type != 'none', ValueError("Invalid feature type for DNN!")
            self.model_name = f"{self.model_name}-{self.feature_type}"
        if self.result_seeds is None:
            self.result_seeds: list[int] = [0, 1, 2]
        elif isinstance(self.result_seeds, int):
            self.result_seeds: list[int] = [self.result_seeds]


def main(args: Arguments):

    uncertainty_results = dict()

    for uncertainty_method in UncertaintyMethods.options():

        result_dir = op.join(args.result_folder, args.dataset_name, args.model_name, uncertainty_method)
        results_for_seeds = list()
        skip_uncertainty = False

        for seed in args.result_seeds:

            seeded_result_dir = op.join(result_dir, f"seed-{seed}")
            test_result_paths = glob.glob(op.join(seeded_result_dir, "preds", "*.pt"))

            if not test_result_paths:
                logger.warning(f"Directory {seeded_result_dir} does not contain any model prediction! "
                               f"Will skip metric logging for {uncertainty_method}")
                skip_uncertainty = True
                break

            preds, variances, lbs, masks = load_results(test_result_paths)

            if variances:  # regression
                metrics = regression_metrics(preds, variances, lbs, masks)
            else:  # classification
                metrics = classification_metrics(preds, lbs, masks)

            result_dict = {k: v['macro-avg'] for k, v in metrics.items()}
            results_for_seeds.append(result_dict)

        if skip_uncertainty:
            continue

        results_aggr = aggregate_seeded_results(results_for_seeds)
        uncertainty_results[uncertainty_method] = results_aggr

    save_results(uncertainty_results, args.result_folder, args.model_name, args.dataset_name)

    return None


def load_results(result_paths):
    lbs = masks = None
    preds_list = list()
    variances_list = list()

    for test_result_path in result_paths:
        results = torch.load(test_result_path)

        if lbs is not None:
            assert (lbs == results['lbs']).all()
        else:
            lbs: np.ndarray = results['lbs']

        if masks is not None:
            assert (masks == results['masks']).all()
        else:
            masks: np.ndarray = results['masks']

        preds_list.append(results['preds']['preds'])
        try:
            variances_list.append(results['preds']['vars'])
        except KeyError:
            pass

    # aggregate mean and variance
    preds = np.stack(preds_list).mean(axis=0)
    if variances_list:  # regression
        # variances = np.mean(np.stack(preds_list) ** 2 + np.stack(variances_list), axis=0) - preds ** 2
        variances = np.stack(variances_list).mean(axis=0)
    else:
        variances = None

    return preds, variances, lbs, masks


def aggregate_seeded_results(results_for_seeds: list[dict[str, float]]):
    assert results_for_seeds

    results_aggr = {metric: [r[metric] for r in results_for_seeds] for metric in list(results_for_seeds[0].keys())}
    for k in results_aggr:
        mean = np.mean(results_aggr)
        std = np.std(results_aggr)
        results_aggr[k] = {'mean': mean, 'std': std}

    return results_aggr


def save_results(results, result_dir, model_name, dataset_name):
    if not results:
        logger.warning("Result dict is empty. No results is saved!")
        return None

    uncertainty_names = list(results.keys())
    metrics = list(list(results.values())[0].keys())
    columns_headers = ["method"] + [f"{metric}-mean" for metric in metrics] + [f"{metric}-std" for metric in metrics]
    columns = {k: list() for k in columns_headers}
    columns["method"] = uncertainty_names
    for uncertainty in uncertainty_names:
        metric_dict = results[uncertainty]
        for metric_name, value in metric_dict.items():
            columns[f"{metric_name}-mean"].append(value["mean"])
            columns[f"{metric_name}-std"].append(value["std"])

    df = pd.DataFrame(columns)
    df.to_csv(op.join(result_dir, f"results-{model_name}-{dataset_name}.csv"))
    return None


def classification_metrics(preds, lbs, masks):

    result_metrics_dict = dict()

    # --- roc-auc ---
    roc_auc_list = list()
    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        roc_auc = roc_auc_score(lbs_, preds_)
        roc_auc_list.append(roc_auc)
    roc_auc_avg = np.mean(roc_auc_list)

    result_metrics_dict['roc-auc'] = {
        'all': roc_auc_list,
        'macro-avg': roc_auc_avg
    }

    # --- prc-auc ---
    prc_auc_list = list()
    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        p, r, _ = precision_recall_curve(lbs_, preds_)
        prc_auc = auc(r, p)
        prc_auc_list.append(prc_auc)
    prc_auc_avg = np.mean(prc_auc_list)

    result_metrics_dict['prc-auc'] = {
        'all': prc_auc_list,
        'macro-avg': prc_auc_avg
    }

    # --- ece ---
    ece_list = list()
    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        ece = binary_calibration_error(torch.from_numpy(preds_), torch.from_numpy(lbs_)).item()
        ece_list.append(ece)
    ece_avg = np.mean(ece_list)

    result_metrics_dict['ece'] = {
        'all': ece_list,
        'macro-avg': ece_avg
    }

    # --- mce ---
    mce_list = list()
    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        mce = binary_calibration_error(torch.from_numpy(preds_), torch.from_numpy(lbs_), norm='max').item()
        mce_list.append(mce)
    mce_avg = np.mean(mce_list)

    result_metrics_dict['mce'] = {
        'all': mce_list,
        'macro-avg': mce_avg
    }

    # --- nll ---
    nll_list = list()
    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        nll = F.binary_cross_entropy(
            input=torch.from_numpy(preds_),
            target=torch.from_numpy(lbs_).to(torch.float),
            reduction='mean'
        ).item()
        nll_list.append(nll)
    nll_avg = np.mean(nll_list)

    result_metrics_dict['nll'] = {
        'all': nll_list,
        'macro-avg': nll_avg
    }

    return result_metrics_dict


def regression_metrics(preds, variances, lbs, masks):

    if len(preds.shape) == 1:
        preds = preds[:, np.newaxis]

    if len(variances.shape) == 1:
        variances = variances[:, np.newaxis]

    # --- rmse ---
    result_metrics_dict = dict()

    rmse_list = list()
    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        rmse = mean_squared_error(lbs_, preds_, squared=False)
        rmse_list.append(rmse)
    rmse_avg = np.mean(rmse_list)

    result_metrics_dict['rmse'] = {
        'all': rmse_list,
        'macro-avg': rmse_avg
    }

    # --- mae ---
    mae_list = list()
    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        mae = mean_absolute_error(lbs_, preds_)
        mae_list.append(mae)
    mae_avg = np.mean(mae_list)

    result_metrics_dict['mae'] = {
        'all': mae_list,
        'macro-avg': mae_avg
    }

    # --- Gaussian NLL ---
    nll_list = list()
    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        vars_ = variances[:, i][masks[:, i].astype(bool)]
        nll = F.gaussian_nll_loss(torch.from_numpy(preds_), torch.from_numpy(lbs_), torch.from_numpy(vars_)).item()
        nll_list.append(nll)
    nll_avg = np.mean(nll_list)

    result_metrics_dict['nll'] = {
        'all': nll_list,
        'macro-avg': nll_avg
    }

    return result_metrics_dict


if __name__ == '__main__':

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        arguments, = parser.parse_args_into_dataclasses()

    set_logging()
    main(args=arguments)
