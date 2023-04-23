
import numpy as np
from torch import Tensor
from torch.nn import GaussianNLLLoss
from torch.nn import functional as F
from scipy.special import softmax, expit
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    precision_recall_curve,
    auc,
)
from typing import Optional, Union, List


# noinspection PyShadowingBuiltins
class GaussianNLL(GaussianNLLLoss):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super().__init__(full=full, eps=eps, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs):
        mean = input[..., 0]
        var = F.softplus(input[..., 1]) + 1e-6
        return super().forward(input=mean, target=target, var=var)


def calculate_classification_metrics(lbs: np.ndarray,
                                     logits: np.ndarray,
                                     metrics: Union[str, List[str]],
                                     normalized: Optional[bool] = False) -> dict:
    """
    Calculate the classification metrics

    Parameters
    ----------
    lbs: true labels
    logits: predicted logits, of shape (dataset_size, n_tasks, n_lbs)
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

        results[metric] = val

    return results


def calculate_regression_metrics(lbs: np.ndarray, preds: np.ndarray, metrics: Union[str, List[str]]) -> dict:
    """
    Calculate the regression metrics

    Parameters
    ----------
    lbs: true labels
    preds: predicted values, of shape (dataset_size, n_tasks)
    metrics: which metric to calculate

    Returns
    -------
    classification metrics
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    results = dict()
    for metric in metrics:
        if metric == 'rmse':
            val = mean_squared_error(lbs, preds, squared=False)
        else:
            raise NotImplementedError("Metric is not implemented")

        results[metric] = val

    return results
