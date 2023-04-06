"""
The scaler for the regression task.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaler.py
"""
from typing import Any
import numpy as np

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

    def inverse_transform(self, x:  np.ndarray):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        Parameters
        ----------
        x: A list of lists of floats.

        Returns
        -------
        The inverse transformed data.
        """
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = np.array(x).astype(float)
            transformed_with_nan = x * self.stds + self.means
            transformed_with_none = np.where(
                np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
            )
            return transformed_with_none
        else:
            return None
