"""
# Author: Yinghao Li
# Modified: February 29th, 2024
# ---------------------------------------
# Description:

These Python functions are designed to calculate various metrics for classification and regression tasks,
particularly focusing on evaluating the performance of models on tasks involving predictions with associated uncertainties.
"""

import torch
import torch.nn.functional as F
import numpy as np

from torchmetrics.functional.classification import binary_calibration_error
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    brier_score_loss,
)
from scipy.stats import norm as gaussian


def classification_metrics(preds, lbs, masks, exclude: list = None):
    """Calculates various metrics for classification tasks.

    This function computes ROC-AUC, PRC-AUC, ECE, MCE, NLL, and Brier score for classification predictions.

    Args:
        preds (numpy.ndarray): Predicted probabilities for each class.
        lbs (numpy.ndarray): Ground truth labels.
        masks (numpy.ndarray): Masks indicating valid data points for evaluation.
        exclude (list, optional): List of metrics to exclude from the calculation.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    if not exclude:
        exclude = list()

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

        if not "roc-auc" in exclude:
            # --- roc-auc ---
            try:
                roc_auc = roc_auc_score(lbs_, preds_)
                roc_auc_list.append(roc_auc)
            except:
                roc_auc_valid_flag = False

        if not "prc-auc" in exclude:
            # --- prc-auc ---
            try:
                p, r, _ = precision_recall_curve(lbs_, preds_)
                prc_auc = auc(r, p)
                prc_auc_list.append(prc_auc)
            except:
                prc_auc_valid_flag = False

        if not "ece" in exclude:
            # --- ece ---
            try:
                ece = binary_calibration_error(torch.from_numpy(preds_), torch.from_numpy(lbs_)).item()
                ece_list.append(ece)
            except:
                ece_valid_flag = False

        if not "mce" in exclude:
            # --- mce ---
            try:
                mce = binary_calibration_error(torch.from_numpy(preds_), torch.from_numpy(lbs_), norm="max").item()
                mce_list.append(mce)
            except:
                mce_valid_flag = False

        if not "nll" in exclude:
            # --- nll ---
            try:
                nll = F.binary_cross_entropy(
                    input=torch.from_numpy(preds_),
                    target=torch.from_numpy(lbs_).to(torch.float),
                    reduction="mean",
                ).item()
                nll_list.append(nll)
            except:
                nll_valid_flag = False

        if not "brier" in exclude:
            # --- brier ---
            try:
                brier = brier_score_loss(lbs_, preds_)
                brier_list.append(brier)
            except:
                brier_valid_flag = False

    if roc_auc_valid_flag and not "roc-auc" in exclude:
        roc_auc_avg = np.mean(roc_auc_list)
        result_metrics_dict["roc-auc"] = {
            "all": roc_auc_list,
            "macro-avg": roc_auc_avg,
        }

    if prc_auc_valid_flag and not "prc-auc" in exclude:
        prc_auc_avg = np.mean(prc_auc_list)
        result_metrics_dict["prc-auc"] = {
            "all": prc_auc_list,
            "macro-avg": prc_auc_avg,
        }

    if ece_valid_flag and not "ece" in exclude:
        ece_avg = np.mean(ece_list)
        result_metrics_dict["ece"] = {"all": ece_list, "macro-avg": ece_avg}

    if mce_valid_flag and not "mce" in exclude:
        mce_avg = np.mean(mce_list)
        result_metrics_dict["mce"] = {"all": mce_list, "macro-avg": mce_avg}

    if nll_valid_flag and not "nll" in exclude:
        nll_avg = np.mean(nll_list)
        result_metrics_dict["nll"] = {"all": nll_list, "macro-avg": nll_avg}

    if brier_valid_flag and not "brier" in exclude:
        brier_avg = np.mean(brier_list)
        result_metrics_dict["brier"] = {
            "all": brier_list,
            "macro-avg": brier_avg,
        }

    return result_metrics_dict


def regression_metrics(preds, variances, lbs, masks, exclude: list = None):
    """Calculates various metrics for regression tasks.

    Computes RMSE, MAE, NLL, and calibration error for regression predictions and their associated uncertainties.

    Args:
        preds (numpy.ndarray): Predicted values (means).
        variances (numpy.ndarray): Predicted variances.
        lbs (numpy.ndarray): Ground truth values.
        masks (numpy.ndarray): Masks indicating valid data points for evaluation.
        exclude (list, optional): List of metrics to exclude from the calculation.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    if not exclude:
        exclude = list()

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

        if not "rmse" in exclude:
            # --- rmse ---
            rmse = mean_squared_error(lbs_, preds_, squared=False)
            rmse_list.append(rmse)

        if not "mae" in exclude:
            # --- mae ---
            mae = mean_absolute_error(lbs_, preds_)
            mae_list.append(mae)

        if not "nll" in exclude:
            # --- Gaussian NLL ---
            nll = F.gaussian_nll_loss(
                torch.from_numpy(preds_),
                torch.from_numpy(lbs_),
                torch.from_numpy(vars_),
            ).item()
            nll_list.append(nll)

        if not "ce" in exclude:
            # --- calibration error ---
            ce = regression_calibration_error(lbs_, preds_, vars_)
            ce_list.append(ce)

    if not "rmse" in exclude:
        rmse_avg = np.mean(rmse_list)
        result_metrics_dict["rmse"] = {"all": rmse_list, "macro-avg": rmse_avg}

    if not "mae" in exclude:
        mae_avg = np.mean(mae_list)
        result_metrics_dict["mae"] = {"all": mae_list, "macro-avg": mae_avg}

    if not "nll" in exclude:
        nll_avg = np.mean(nll_list)
        result_metrics_dict["nll"] = {"all": nll_list, "macro-avg": nll_avg}

    if not "ce" in exclude:
        ce_avg = np.mean(ce_list)
        result_metrics_dict["ce"] = {"all": ce_list, "macro-avg": ce_avg}

    return result_metrics_dict


def regression_calibration_error(lbs, preds, variances, n_bins=20):
    """Calculates the calibration error for regression tasks.

    Uses the predicted means and variances to estimate the calibration error across a specified number of bins.

    Args:
        lbs (numpy.ndarray): Ground truth values.
        preds (numpy.ndarray): Predicted values (means).
        variances (numpy.ndarray): Predicted variances.
        n_bins (int, optional): Number of bins to use for calculating calibration error. Defaults to 20.

    Returns:
        float: The calculated calibration error.
    """
    sigma = np.sqrt(variances)
    phi_lbs = gaussian.cdf(lbs, loc=preds.reshape(-1, 1), scale=sigma.reshape(-1, 1))

    expected_confidence = np.linspace(0, 1, n_bins + 1)[1:-1]
    observed_confidence = np.zeros_like(expected_confidence)

    for i in range(0, len(expected_confidence)):
        observed_confidence[i] = np.mean(phi_lbs <= expected_confidence[i])

    calibration_error = np.mean((expected_confidence.ravel() - observed_confidence.ravel()) ** 2)

    return calibration_error
