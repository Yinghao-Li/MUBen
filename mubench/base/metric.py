
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


def calculate_classification_metrics(lbs: np.ndarray,
                                     logits: np.ndarray,
                                     metrics: Union[str, List[str]],
                                     normalized: Optional[bool] = False):
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

    return ValidMetric(**results)


def calculate_regression_metrics(lbs: np.ndarray, preds: np.ndarray, metrics: Union[str, List[str]]):
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

    return ValidMetric(**results)
