
import numpy as np
from torch import Tensor
from torch.nn import GaussianNLLLoss
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error
)
from ..utils.math import logit_to_var
from seqlbtoolkit.training.eval import Metric


# noinspection PyShadowingBuiltins
class GaussianNLL(GaussianNLLLoss):
    def __init__(self, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super().__init__(full=full, eps=eps, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor, *args, **kwargs):
        mean = input[..., 0]
        var = logit_to_var(input[..., 1])
        return super().forward(input=mean, target=target, var=var)


class ClassificationMetric(Metric):
    def __init__(self, accuracy=None, roc_auc=None):
        super().__init__()

        self.remove_attrs()

        self.accuracy = accuracy
        self.roc_auc = roc_auc


def get_classification_metrics(lbs: np.ndarray, probs: np.ndarray):
    """
    Calculate the classification metrics

    Parameters
    ----------
    lbs: true labels
    probs: predicted probabilities

    Returns
    -------
    classification metrics
    """
    preds = np.argmax(probs, axis=-1)
    accuracy = np.sum(preds == lbs) / len(lbs)

    if len(probs.shape) == 1 or probs.shape[-1] == 1:
        roc_auc = roc_auc_score(lbs, probs)
    elif probs.shape[-1] == 2:
        roc_auc = roc_auc_score(lbs, probs[:, -1])
    else:
        roc_auc = None

    return ClassificationMetric(accuracy=accuracy, roc_auc=roc_auc)


class RegressionMetric(Metric):

    def __init__(self, rmse=None, mae=None):
        super().__init__()

        self.remove_attrs()

        self.rmse = rmse
        self.mae = mae


def get_regression_metrics(lbs: np.ndarray, preds: np.ndarray):
    """
    Calculate the regression metrics

    Parameters
    ----------
    lbs: true labels
    preds: predicted probabilities

    Returns
    -------
    regression metrics
    """
    rmse = mean_squared_error(lbs, preds, squared=False)
    mae = mean_absolute_error(lbs, preds)

    return RegressionMetric(rmse=rmse, mae=mae)
