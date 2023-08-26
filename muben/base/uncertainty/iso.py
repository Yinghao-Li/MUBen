"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description:

Implements regression calibration.  This calibration method 
attempts to improve the calibration of regression models using isotonic regression.

# Reference: https://github.com/kage08/DistCal/tree/master
"""

import torch
import logging
import numpy as np
import scipy.stats
import scipy.integrate
import scipy.optimize
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

__all__ = ["IsotonicCalibration"]

eps = np.finfo(np.random.randn(1).dtype).eps


class IsotonicCalibration:
    def __init__(self, n_task):
        """
        Initialize the Isotonic Calibration.

        Parameters
        ----------
        n_task : int
            Number of tasks for which isotonic calibration is performed.
        """
        super().__init__()
        self._n_task = n_task
        self._isotonic_regressors = [
            IsotonicRegression(out_of_bounds="clip") for _ in range(n_task)
        ]

    def fit(self, means, variances, lbs, masks) -> "IsotonicCalibration":
        """
        Fit isotonic regressors to the calibration (validation) dataset.

        Parameters
        ----------
        means : np.ndarray or torch.Tensor
            Predicted means with shape (batch_size, n_tasks).
        variances : np.ndarray or torch.Tensor
            Predicted variances with shape (batch_size, n_tasks).
        lbs : np.ndarray or torch.Tensor
            True labels.
        masks : np.ndarray or torch.Tensor
            Masks with shape (batch_size, n_tasks), indicating valid entries.

        Returns
        -------
        self
        """
        # Convert tensors to numpy arrays if needed.
        means, variances, lbs, masks = [
            x.cpu().numpy() if isinstance(x, torch.Tensor) else x
            for x in [means, variances, lbs, masks]
        ]
        bool_masks = masks.astype(bool)

        # Ensure the data is 2-dimensional.
        for data in [means, variances, lbs, bool_masks]:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

        # Fit the isotonic regressors.
        for task_means, task_vars, task_lbs, task_masks, regressor in zip(
            means.T,
            variances.T,
            lbs.T,
            bool_masks.T,
            self._isotonic_regressors,
        ):
            task_means = task_means[task_masks]
            task_vars = task_vars[task_masks]
            task_lbs = task_lbs[task_masks]

            q, q_hat = get_iso_cal_table(
                task_lbs, task_means, np.sqrt(task_vars)
            )

            regressor.fit(q, q_hat)

        return self

    def calibrate(self, means, variances, lbs, masks):
        """
        Apply isotonic calibration to the test dataset and compute metrics.

        Parameters
        ----------
        means : np.ndarray or torch.Tensor
            Predicted means with shape (batch_size, n_tasks).
        variances : np.ndarray or torch.Tensor
            Predicted variances with shape (batch_size, n_tasks).
        lbs : np.ndarray or torch.Tensor
            True labels.
        masks : np.ndarray or torch.Tensor
            Masks with shape (batch_size, n_tasks), indicating valid entries.
        """
        n_t_test = 1024

        # Convert tensors to numpy arrays if needed.
        means, variances, lbs, masks = [
            x.cpu().numpy() if isinstance(x, torch.Tensor) else x
            for x in [means, variances, lbs, masks]
        ]

        bool_masks = masks.astype(bool)

        # Ensure the data is 2-dimensional.
        for data in [means, variances, lbs, bool_masks]:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

        nll_list = list()
        rmse_list = list()
        for task_means, task_vars, task_lbs, task_masks, regressor in zip(
            means.T,
            variances.T,
            lbs.T,
            bool_masks.T,
            self._isotonic_regressors,
        ):
            task_means = task_means[task_masks]
            task_vars = task_vars[task_masks]
            task_stds = np.sqrt(task_vars)
            task_lbs = task_lbs[task_masks]

            t_list_test = np.linspace(
                np.min(task_means) - 16.0 * np.max(task_stds),
                np.max(task_means) + 16.0 * np.max(task_stds),
                n_t_test,
            ).reshape(1, -1)

            q_base, s_base = get_norm_q(
                task_means.ravel(), task_stds.ravel(), t_list_test.ravel()
            )
            q_iso = regressor.predict(q_base.ravel()).reshape(np.shape(q_base))

            s_iso = np.diff(q_iso, axis=1) / (
                t_list_test[0, 1:] - t_list_test[0, :-1]
            ).ravel().reshape(1, -1).repeat(len(task_lbs), axis=0)

            y_iso = get_y_hat(t_list_test.ravel(), s_iso)

            # NLL
            nll_iso = get_log_loss(task_lbs, t_list_test.ravel(), s_iso)
            # MSE
            se_iso = get_se(y_iso, task_lbs)

            nll_list.append(np.mean(nll_iso))
            rmse_list.append(np.sqrt(np.mean(se_iso)))

        logger.info(f"[ISO calibration] RMSE: {np.mean(rmse_list):.4f}")
        logger.info(f"[ISO calibration] NLL: {np.mean(nll_list):.4f}")


def get_iso_cal_table(y, mu, sigma):
    """
    Generate the calibration table for isotonic regression.

    Parameters
    ----------
    lbs : np.ndarray
        True labels.
    means : np.ndarray
        Predicted means.
    stds : np.ndarray
        Predicted standard deviations.

    Returns
    -------
    q : np.ndarray
        The sorted percentile values.
    q_hat : np.ndarray
        The observed percentile values.
    """
    q_raw = scipy.stats.norm.cdf(
        y, loc=mu.reshape(-1, 1), scale=sigma.reshape(-1, 1)
    )
    q_list, idx = np.unique(q_raw, return_inverse=True)

    q_hat_list = np.zeros_like(q_list)
    for i in range(0, len(q_list)):
        q_hat_list[i] = np.mean(q_raw <= q_list[i])
    q_hat = q_hat_list[idx]

    return q_raw.ravel(), q_hat.ravel()


def get_norm_q(mu, sigma, t_list):
    """
    Compute the expected percentile values for a normal distribution.

    Parameters
    ----------
    means : np.ndarray
        Predicted means.
    stds : np.ndarray
        Predicted standard deviations.
    t_list : np.ndarray
        Points at which percentiles are calculated.

    Returns
    -------
    q : np.ndarray
        The expected percentile values.
    s : np.ndarray
        The expected standard deviations for each percentile.
    """
    q = np.zeros([len(mu), len(t_list)])

    s = np.zeros([len(mu), len(t_list)])

    for j in range(0, len(t_list)):
        q[:, j] = np.squeeze(
            scipy.stats.norm.cdf(t_list[j], loc=mu, scale=sigma)
        )
        s[:, j] = np.squeeze(
            scipy.stats.norm.pdf(t_list[j], loc=mu, scale=sigma)
        )

    return q, s


def get_log_loss(y, t_list, density_hat):
    """
    Calculate log loss for given y values and densities.

    Parameters
    ----------
    y : np.ndarray
        Array of data points.
    t_list : np.ndarray
        Array of t-values.
    density_hat : np.ndarray
        Array of density values.

    Returns
    -------
    np.ndarray
        Log loss for each data point in y.
    """
    t_list_hat = (t_list[0:-1] + t_list[1:]) / 2

    ll = np.zeros(len(y))

    for i in range(0, len(y)):
        t_loc = np.argmin(np.abs(y[i] - t_list_hat))
        if density_hat[i, t_loc] <= 0:
            ll[i] = -np.log(eps)
        else:
            ll[i] = -np.log(density_hat[i, t_loc])

    return ll


def get_y_hat(t_list, density_hat):
    """
    Compute y_hat based on t_list and density values.

    Parameters
    ----------
    t_list : np.ndarray
        Array of t-values.
    density_hat : np.ndarray
        Array of density values.

    Returns
    -------
    np.ndarray
        y_hat for each entry in density_hat.
    """
    n_y, n_t = np.shape(density_hat)

    t_list_hat = (t_list[0:-1] + t_list[1:]) / 2

    y_hat = np.zeros(n_y)

    if len(t_list_hat) == n_t:
        for i in range(0, n_y):
            y_py = t_list_hat * density_hat[i, :]

            y_hat[i] = scipy.integrate.trapz(y_py, t_list_hat)

    else:
        for i in range(0, n_y):
            y_py = t_list * density_hat[i, :]

            y_hat[i] = scipy.integrate.trapz(y_py, t_list)

    return y_hat


def get_y_var(t_list, density_hat):
    """
    Compute y variance based on t_list and density values.

    Parameters
    ----------
    t_list : np.ndarray
        Array of t-values.
    density_hat : np.ndarray
        Array of density values.

    Returns
    -------
    np.ndarray
        Variance for each entry in density_hat.
    """
    n_y, n_t = np.shape(density_hat)

    t_list_hat = (t_list[0:-1] + t_list[1:]) / 2

    y_hat = get_y_hat(t_list, density_hat)
    y_var = np.zeros(n_y)

    if len(t_list_hat) == n_t:
        for i in range(0, n_y):
            y_py = ((t_list_hat - y_hat[i]) ** 2) * density_hat[i, :]

            y_var[i] = scipy.integrate.trapz(y_py, t_list_hat)

    else:
        for i in range(0, n_y):
            y_py = ((t_list - y_hat[i]) ** 2) * density_hat[i, :]

            y_var[i] = scipy.integrate.trapz(y_py, t_list)

    return y_var


def get_q_y(y, q, t_list):
    """
    Get q values for given y and t_list.

    Parameters
    ----------
    y : np.ndarray
        Array of data points.
    q : np.ndarray
        Array of q-values.
    t_list : np.ndarray
        Array of t-values.

    Returns
    -------
    np.ndarray
        q-values for each data point in y.
    """
    q_y = np.zeros(len(y))

    for i in range(0, len(y)):
        t_loc = np.argmax(t_list > y[i])
        q_y[i] = q[i, t_loc]

    return q_y
