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
    brier_score_loss
)
from scipy.stats import norm as gaussian

from muben.utils.io import set_logging, init_dir, load_results
from muben.utils.macro import (
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
    dataset_names: Optional[str] = field(
        default=None,
        metadata={
            "nargs": "*",
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

        if self.dataset_names is None:
            self.dataset_names: list[str] = DATASET_NAMES
        elif isinstance(self.dataset_names, str):
            self.dataset_names: list[str] = [self.dataset_names]


def main(args: Arguments):

    for dataset_name in args.dataset_names:
        logger.info(f"Processing {dataset_name} dataset...")

        uncertainty_results = dict()
        for uncertainty_method in UncertaintyMethods.options():

            result_dir = op.join(args.result_folder, dataset_name, args.model_name, uncertainty_method)

            if uncertainty_method != UncertaintyMethods.ensembles:

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

                    if variances is not None:  # regression
                        metrics = regression_metrics(preds, variances, lbs, masks)
                    else:  # classification
                        metrics = classification_metrics(preds, lbs, masks)

                    result_dict = {k: v['macro-avg'] for k, v in metrics.items()}
                    results_for_seeds.append(result_dict)

                if skip_uncertainty:
                    continue

                results_aggr = aggregate_seeded_results(results_for_seeds)
                uncertainty_results[uncertainty_method] = results_aggr

            else:

                seeded_result_dirs = glob.glob(op.join(result_dir, '*'))
                test_result_paths = [op.join(sid, "preds", "0.pt") for sid in seeded_result_dirs]

                if not test_result_paths:
                    logger.warning(f"Directory {result_dir} does not contain any model prediction! "
                                   f"Will skip metric logging for {uncertainty_method}")
                    break

                preds, variances, lbs, masks = load_results(test_result_paths)

                if variances is not None:  # regression
                    metrics = regression_metrics(preds, variances, lbs, masks)
                else:  # classification
                    metrics = classification_metrics(preds, lbs, masks)

                result_dict = {k: v['macro-avg'] for k, v in metrics.items()}
                results_aggr = {k: {"mean": v, "std": np.NaN} for k, v in result_dict.items()}

                uncertainty_results[uncertainty_method] = results_aggr

        save_results(uncertainty_results, args.result_folder, args.model_name, dataset_name)

    return None


def aggregate_seeded_results(results_for_seeds: list[dict[str, float]]):
    assert results_for_seeds

    results_aggr = {metric: [r.get(metric, np.NaN) for r in results_for_seeds] for metric in list(results_for_seeds[0].keys())}
    for k in results_aggr:
        mean = np.nanmean(results_aggr[k])
        std = np.nanstd(results_aggr[k])
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
    init_dir(op.join(result_dir, "RESULTS", "scores"), clear_original_content=False)
    df.to_csv(op.join(result_dir, "RESULTS", "scores", f"{model_name}-{dataset_name}.csv"))
    return None


def classification_metrics(preds, lbs, masks):

    result_metrics_dict = dict()

    roc_auc_list = list()
    prc_auc_list = list()
    ece_list = list()
    mce_list = list()
    nll_list = list()
    brier_list = list()

    roc_auc_valid_flag = True
    prc_auc_valid_flag = True
    ece_valid_flag = True
    mce_valid_flag = True
    nll_valid_flag = True
    brier_valid_flag = True

    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]

        if len(lbs_) < 1:
            continue
        if (lbs_ < 0).any():
            raise ValueError("Invalid label value encountered!")
        if (lbs_ == 0).all() or (lbs_ == 1).all():  # skip tasks with only one label type, as Uni-Mol did.
            continue

        # --- roc-auc ---
        try:
            roc_auc = roc_auc_score(lbs_, preds_)
            roc_auc_list.append(roc_auc)
        except:
            roc_auc_valid_flag = False

        # --- prc-auc ---
        try:
            p, r, _ = precision_recall_curve(lbs_, preds_)
            prc_auc = auc(r, p)
            prc_auc_list.append(prc_auc)
        except:
            prc_auc_valid_flag = False

        # --- ece ---
        try:
            ece = binary_calibration_error(torch.from_numpy(preds_), torch.from_numpy(lbs_)).item()
            ece_list.append(ece)
        except:
            ece_valid_flag = False

        # --- mce ---
        try:
            mce = binary_calibration_error(torch.from_numpy(preds_), torch.from_numpy(lbs_), norm='max').item()
            mce_list.append(mce)
        except:
            mce_valid_flag = False

        # --- nll ---
        try:
            nll = F.binary_cross_entropy(
                input=torch.from_numpy(preds_),
                target=torch.from_numpy(lbs_).to(torch.float),
                reduction='mean'
            ).item()
            nll_list.append(nll)
        except:
            nll_valid_flag = False

        # --- brier ---
        try:
            brier = brier_score_loss(lbs_, preds_)
            brier_list.append(brier)
        except:
            brier_valid_flag = False

    if roc_auc_valid_flag:
        roc_auc_avg = np.mean(roc_auc_list)
        result_metrics_dict['roc-auc'] = {'all': roc_auc_list, 'macro-avg': roc_auc_avg}

    if prc_auc_valid_flag:
        prc_auc_avg = np.mean(prc_auc_list)
        result_metrics_dict['prc-auc'] = {'all': prc_auc_list, 'macro-avg': prc_auc_avg}

    if ece_valid_flag:
        ece_avg = np.mean(ece_list)
        result_metrics_dict['ece'] = {'all': ece_list, 'macro-avg': ece_avg}

    if mce_valid_flag:
        mce_avg = np.mean(mce_list)
        result_metrics_dict['mce'] = {'all': mce_list, 'macro-avg': mce_avg}

    if nll_valid_flag:
        nll_avg = np.mean(nll_list)
        result_metrics_dict['nll'] = {'all': nll_list, 'macro-avg': nll_avg}

    if brier_valid_flag:
        brier_avg = np.mean(brier_list)
        result_metrics_dict['brier'] = {'brier': brier_list, 'macro-avg': brier_avg}

    return result_metrics_dict


def regression_metrics(preds, variances, lbs, masks):

    if len(preds.shape) == 1:
        preds = preds[:, np.newaxis]

    if len(variances.shape) == 1:
        variances = variances[:, np.newaxis]

    # --- rmse ---
    result_metrics_dict = dict()

    rmse_list = list()
    mae_list = list()
    nll_list = list()
    ce_list = list()

    for i in range(lbs.shape[-1]):
        lbs_ = lbs[:, i][masks[:, i].astype(bool)]
        preds_ = preds[:, i][masks[:, i].astype(bool)]
        vars_ = variances[:, i][masks[:, i].astype(bool)]

        # --- rmse ---
        rmse = mean_squared_error(lbs_, preds_, squared=False)
        rmse_list.append(rmse)

        # --- mae ---
        mae = mean_absolute_error(lbs_, preds_)
        mae_list.append(mae)

        # --- Gaussian NLL ---
        nll = F.gaussian_nll_loss(torch.from_numpy(preds_), torch.from_numpy(lbs_), torch.from_numpy(vars_)).item()
        nll_list.append(nll)

        # --- calibration error ---
        ce = regression_calibration_error(lbs_, preds_, vars_)
        ce_list.append(ce)

    rmse_avg = np.mean(rmse_list)
    result_metrics_dict['rmse'] = {'all': rmse_list, 'macro-avg': rmse_avg}

    mae_avg = np.mean(mae_list)
    result_metrics_dict['mae'] = {'all': mae_list, 'macro-avg': mae_avg}

    nll_avg = np.mean(nll_list)
    result_metrics_dict['nll'] = {'all': nll_list, 'macro-avg': nll_avg}

    ce_avg = np.mean(ce_list)
    result_metrics_dict['ce'] = {'all': ce_list, 'macro-avg': ce_avg}

    return result_metrics_dict


def regression_calibration_error(lbs, preds, variances, n_bins=20):
    sigma = np.sqrt(variances)
    phi_lbs = gaussian.cdf(lbs, loc=preds.reshape(-1, 1), scale=sigma.reshape(-1, 1))

    expected_confidence = np.linspace(0, 1, n_bins+1)[1:-1]
    observed_confidence = np.zeros_like(expected_confidence)

    for i in range(0, len(expected_confidence)):
        observed_confidence[i] = np.mean(phi_lbs <= expected_confidence[i])

    calibration_error = np.mean((expected_confidence.ravel() - observed_confidence.ravel()) ** 2)

    return calibration_error


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
