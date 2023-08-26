"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: Validation and prediction test metrics
"""


import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    auc,
)
from typing import Union, List

__all__ = ["calculate_binary_classification_metrics", "calculate_regression_metrics"]


def calculate_binary_classification_metrics(
    lbs: np.ndarray,
    probs: np.ndarray,
    masks: np.ndarray,
    metrics: Union[str, List[str]],
) -> dict:
    """
    Calculate the classification metrics

    Parameters
    ----------
    lbs: true labels, with shape (dataset_size, n_tasks)
    probs: predicted logits, of shape (dataset_size, n_tasks, [n_lbs])
    masks: label masks, of shape (dataset_size, n_tasks), should be bool values
    metrics: which metric to calculate

    Returns
    -------
    classification metrics
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    results = dict()
    for metric in metrics:
        if len(lbs.shape) == 1:  # only one task
            lbs = lbs[masks]
            probs = probs[masks]

            if metric == "roc_auc":
                if probs.shape[-1] == 2:
                    probs = probs[..., 1]
                val = roc_auc_score(lbs, probs)
            elif metric == "prc_auc":
                if probs.shape[-1] == 2:
                    probs = probs[..., 1]
                p, r, _ = precision_recall_curve(lbs, probs)
                val = auc(r, p)
            else:
                raise NotImplementedError("Metric is not implemented")

        else:  # multiple classification tasks
            # swap n_tasks and dataset size
            lbs = lbs.swapaxes(0, 1)
            probs = probs.swapaxes(0, 1)
            masks = masks.swapaxes(0, 1)

            vals = list()
            for lbs_, probs_, masks_ in zip(lbs, probs, masks):
                lbs_ = lbs_[masks_]
                probs_ = probs_[masks_]

                if len(lbs_) < 1:
                    continue
                if (lbs_ < 0).any():
                    raise ValueError("Invalid label value encountered!")

                # skip tasks with no positive label, as Uni-Mol did.
                if (lbs_ == 0).all() or (lbs_ == 1).all():
                    continue

                if metric == "roc_auc":
                    vals.append(roc_auc_score(lbs_, probs_))
                elif metric == "prc_auc":
                    p, r, _ = precision_recall_curve(lbs_, probs_)
                    vals.append(auc(r, p))
                else:
                    raise NotImplementedError("Metric is not implemented")
            val = np.mean(vals)

        results[metric] = val

    return results


def calculate_regression_metrics(
    lbs: np.ndarray,
    preds: np.ndarray,
    masks: np.ndarray,
    metrics: Union[str, List[str]],
) -> dict:
    """
    Calculate the regression metrics

    Parameters
    ----------
    lbs: true labels, with shape (dataset_size, n_tasks)
    preds: predicted values, of shape (dataset_size, n_tasks)
    masks: label masks, of shape (dataset_size, n_tasks), should be bool values
    metrics: which metric to calculate

    Returns
    -------
    classification metrics
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    # micro and macro average are the save for regression tasks
    lbs = lbs[masks]
    preds = preds[masks]

    results = dict()
    for metric in metrics:
        if metric == "rmse":
            val = mean_squared_error(lbs, preds, squared=False)
        elif metric == "mae":
            val = mean_absolute_error(lbs, preds)
        else:
            raise NotImplementedError("Metric is not implemented")

        results[metric] = val

    return results
