
import numpy as np
from torch import Tensor
from torch.nn import GaussianNLLLoss
from scipy.special import softmax, expit
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    precision_recall_curve,
    auc,
)
from seqlbtoolkit.training.eval import Metric
from typing import Optional, Union, List
from ..utils.math import logit_to_var


# noinspection PyShadowingBuiltins
class GaussianNLL(GaussianNLLLoss):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super().__init__(full=full, eps=eps, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs):
        mean = input[..., 0]
        var = logit_to_var(input[..., 1])
        return super().forward(input=mean, target=target, var=var)


#  We might just replace this by a dict
class ValidMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__()

        self.remove_attrs()

        for k, v in kwargs.items():
            setattr(self, k, v)


def calculate_classification_metrics(lbs: np.ndarray, logits: np.ndarray, metrics: Union[str, List[str]]):
    """
    Calculate the classification metrics

    Parameters
    ----------
    lbs: true labels
    logits: predicted logits, of shape (dataset_size, n_tasks, n_lbs)
    metrics: which metric to calculate

    Returns
    -------
    classification metrics
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if len(logits.shape) > 1 and logits.shape[-1] >= 2:
        preds = softmax(logits, axis=-1)
    else:
        preds = expit(logits)  # sigmoid function

    assert not (len(logits.shape) > 1 and preds.shape[-1] > 2), \
        ValueError('Currently only support binary classification metrics!')

    results = dict()
    for metric in metrics:
        if metric == 'roc_auc':
            if preds.shape[-1] == 2:
                preds = preds[..., 1]
            val = roc_auc_score(lbs, preds)

        elif metric == 'prc_auc':
            if preds.shape[-1] == 2:
                preds = preds[..., 1]
            p, r, _ = precision_recall_curve(lbs, preds)
            val = auc(r, p)

        else:
            raise NotImplementedError("Metric is not implemented")

        results[metric] = val

    return ValidMetric(**results)


def calculate_regression_metrics(lbs: np.ndarray, logits: np.ndarray, metrics: Union[str, List[str]]):
    """
    Calculate the regression metrics

    Parameters
    ----------
    lbs: true labels
    logits: predicted logits, of shape (dataset_size, n_tasks, n_lbs)
    metrics: which metric to calculate

    Returns
    -------
    classification metrics
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    preds = logits if logits.shape[-1] == 1 or len(logits.shape) == 1 else logits[..., 0]

    results = dict()
    for metric in metrics:
        if metric == 'rmse':
            val = mean_squared_error(lbs, preds, squared=False)
        else:
            raise NotImplementedError("Metric is not implemented")

        results[metric] = val

    return ValidMetric(**results)
