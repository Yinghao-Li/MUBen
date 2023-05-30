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
from datetime import datetime
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

from mubench.utils.io import set_logging, set_log_path
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
    uncertainty_method: Optional[str] = field(
        default="none",
        metadata={
            "choices": UncertaintyMethods.options(),
            "help": "Uncertainty method"
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

    def __post_init__(self):
        if self.model_name == "DNN":
            assert self.feature_type != 'none', ValueError("Invalid feature type for DNN!")
            self.model_name = f"{self.model_name}-{self.feature_type}"

        self.result_dir = op.join(self.result_folder, self.dataset_name, self.model_name, self.uncertainty_method)


def main(args: Arguments):

    seeded_result_dirs = glob.glob(op.join(args.result_dir, '*'))
    for seeded_result_dir in seeded_result_dirs:

        test_result_paths = glob.glob(op.join(seeded_result_dir, "preds", "*.pt"))

        lbs = masks = None
        preds_list = list()
        variances_list = list()
        for test_result_path in test_result_paths:
            results = torch.load(test_result_path)

            lbs = results['lbs']
            masks = results['masks']

            preds_list.append(results['preds']['preds'])
            try:
                variances_list.append(results['preds']['vars'])
            except KeyError:
                pass

        # aggregate mean and variance
        preds = np.stack(preds_list).mean(axis=0)
        if variances_list:  # regression
            variances = np.mean(np.stack(preds_list) ** 2 + np.stack(variances_list), axis=0) - preds ** 2
            metrics = regression_metrics(preds, variances, lbs, masks)
        else:  # classification
            metrics = classification_metrics(preds, lbs, masks)

        result_dict = {k: v['macro-avg'] for k, v in metrics.items()}
        result_df = pd.DataFrame([result_dict])
        result_df.to_csv(op.join(seeded_result_dir, 'metrics.csv'))

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

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(json_file=op.abspath(sys.argv[1]))
    else:
        arguments, = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_path", None):
        arguments.log_path = set_log_path(arguments, _time)

    set_logging(log_path=arguments.log_path)

    main(args=arguments)
