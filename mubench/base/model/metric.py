
import numpy as np
from scipy.special import softmax, expit
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    auc,
)
from typing import Optional, Union, List


def calculate_classification_metrics(lbs: np.ndarray,
                                     logits: np.ndarray,
                                     masks: np.ndarray,
                                     metrics: Union[str, List[str]],
                                     normalized: Optional[bool] = False) -> dict:
    """
    Calculate the classification metrics

    Parameters
    ----------
    lbs: true labels, with shape (dataset_size, n_tasks)
    logits: predicted logits, of shape (dataset_size, n_tasks, n_lbs)
    masks: label masks, of shape (dataset_size, n_tasks), should be bool values
    metrics: which metric to calculate
    normalized: whether the logits are already normalized and became probabilities

    Returns
    -------
    classification metrics
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    if not normalized:
        if len(logits.shape) > 1 and logits.shape[-1] >= 2:
            probs = softmax(logits, axis=-1)
        else:
            probs = expit(logits)  # sigmoid function
    else:
        probs = logits

    assert not (len(logits.shape) > 1 and probs.shape[-1] > 2), \
        ValueError('Currently only support binary classification metrics!')

    results = dict()
    for metric in metrics:
        if len(lbs.shape) == 1:  # only one task

            lbs = lbs[masks]
            probs = probs[masks]

            if metric == 'roc_auc':
                if probs.shape[-1] == 2:
                    probs = probs[..., 1]
                val = roc_auc_score(lbs, probs)
            elif metric == 'prc_auc':
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

                if (lbs_ < 0).any():
                    raise ValueError("Invalid label value encountered!")
                if lbs_.sum() < 1:  # skip tasks with no positive label, as Uni-Mol did.
                    continue

                if metric == 'roc_auc':
                    if probs_.shape[-1] == 2:
                        probs_ = probs_[..., 1]
                    vals.append(roc_auc_score(lbs_, probs_))
                elif metric == 'prc_auc':
                    if probs_.shape[-1] == 2:
                        probs_ = probs_[..., 1]
                    p, r, _ = precision_recall_curve(lbs_, probs_)
                    vals.append(auc(r, p))
                else:
                    raise NotImplementedError("Metric is not implemented")
            val = np.mean(vals)

        results[metric] = val

    return results


def calculate_regression_metrics(lbs: np.ndarray,
                                 preds: np.ndarray,
                                 masks: np.ndarray,
                                 metrics: Union[str, List[str]]) -> dict:
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
        if metric == 'rmse':
            val = mean_squared_error(lbs, preds, squared=False)
        elif metric == 'mae':
            val = mean_absolute_error(lbs, preds)
        else:
            raise NotImplementedError("Metric is not implemented")

        results[metric] = val

    return results
