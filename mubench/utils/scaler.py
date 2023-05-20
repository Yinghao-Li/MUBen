"""
The scaler for the regression task.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaler.py
"""
import numpy as np
from typing import Any, Optional

__all__ = ['StandardScaler']


class StandardScaler:
    """
    A StandardScaler normalizes a dataset.

    When fit on a dataset, the StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the StandardScaler subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        Initialize StandardScaler, optionally with means and standard deviations precomputed.

        Parameters
        ----------
        means: An optional 1D numpy array of precomputed means.
        stds: An optional 1D numpy array of precomputed standard deviations.
        replace_nan_token: The token to use in place of nans.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, x: np.ndarray) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis.

        Parameters
        ----------
        x: A list of lists of floats.

        Returns
        -------
        The fitted StandardScaler.
        """
        x = np.array(x).astype(float)
        self.means = np.nanmean(x, axis=0)
        self.stds = np.nanstd(x, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, x: np.ndarray):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        Parameters
        ----------
        x: A list of lists of floats.

        Returns
        -------
        The transformed data.
        """
        x = np.array(x).astype(float)
        transformed_with_nan = (x - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, x:  np.ndarray, var: Optional[np.ndarray] = None):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        Parameters
        ----------
        x: model-predicted labels
        var: optional, model-predicted label Gaussian variances

        Returns
        -------
        The inverse transformed data.
        """
        assert isinstance(x, np.ndarray), TypeError("x is required to be numpy array!")
        if var is not None:
            assert isinstance(x, np.ndarray), TypeError("x is required to be numpy array!")

        transformed_x = x * self.stds + self.means
        transformed_x = np.where(
            np.isnan(transformed_x), self.replace_nan_token, transformed_x
        )
        if var is None:
            return transformed_x

        # TODO: should double check if this is correct
        transformed_var = var * self.stds ** 2
        transformed_var = np.where(
            np.isnan(transformed_var), self.replace_nan_token, transformed_var
        )

        return transformed_x, transformed_var
