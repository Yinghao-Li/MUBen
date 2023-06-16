"""
Modified from https://github.com/kage08/DistCal/tree/master
"""
import torch
import numpy as np
import scipy.stats
import scipy.integrate
import scipy.optimize
from sklearn.isotonic import IsotonicRegression

__all__ = ["IsotonicCalibration"]

eps = np.finfo(np.random.randn(1).dtype).eps


class IsotonicCalibration:
    def __init__(self, n_task):
        super().__init__()
        self._n_task = n_task
        self._isotonic_regressors = [IsotonicRegression(out_of_bounds='clip') for _ in n_task]

    def fit(self, means, variances, lbs, masks) -> "IsotonicCalibration":
        """
        Fit isotonic regressors to the calibration (validation) dataset

        Parameters
        ----------
        means: predicted mean, (batch_size, n_tasks)
        variances: predicted variances, (batch_size, n_tasks)
        lbs: true labels
        masks: masks, (batch_size, n_tasks)

        Returns
        -------
        self
        """
        if isinstance(means, torch.Tensor):
            means = means.cpu().numpy()
        if isinstance(variances, torch.Tensor):
            variances = variances.cpu().numpy()
        if isinstance(lbs, torch.Tensor):
            lbs = lbs.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        bool_masks = masks.to(torch.bool)

        if len(means.shape) == 1:
            means = means.reshape(-1, 1)
        if len(variances.shape) == 1:
            variances = variances.reshape(-1, 1)
        if len(lbs.shape) == 1:
            lbs = lbs.reshape(-1, 1)
        if len(bool_masks.shape) == 1:
            bool_masks = bool_masks.reshape(-1, 1)

        for task_means, task_vars, task_lbs, task_masks, regressor in \
                zip(means.T, variances.T, lbs.T, bool_masks.T, self._isotonic_regressors):
            task_means = task_means[task_masks]
            task_vars = task_vars[task_masks]
            task_lbs = task_lbs[task_masks]

            q, q_hat = get_iso_cal_table(task_lbs, task_means, task_vars.sqrt())

            regressor.fit(q, q_hat)

        return self

    def calibrate(self, means, variances, lbs, masks):
        n_t_test = 1024

        if isinstance(means, torch.Tensor):
            means = means.cpu().numpy()
        if isinstance(variances, torch.Tensor):
            variances = variances.cpu().numpy()
        if isinstance(lbs, torch.Tensor):
            lbs = lbs.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        bool_masks = masks.to(torch.bool)

        if len(means.shape) == 1:
            means = means.reshape(-1, 1)
        if len(variances.shape) == 1:
            variances = variances.reshape(-1, 1)
        if len(lbs.shape) == 1:
            lbs = lbs.reshape(-1, 1)
        if len(bool_masks.shape) == 1:
            bool_masks = bool_masks.reshape(-1, 1)

        for task_means, task_vars, task_lbs, task_masks, regressor in \
                zip(means.T, variances.T, lbs.T, bool_masks.T, self._isotonic_regressors):

            task_means = task_means[task_masks]
            task_vars = task_vars[task_masks]
            task_stds = task_vars.sqrt()
            task_lbs = task_lbs[task_masks]

            t_list_test = np.linspace(np.min(task_means) - 16.0 * np.max(task_stds),
                                      np.max(task_means) + 16.0 * np.max(task_stds),
                                      n_t_test).reshape(1, -1)

            y_base = task_means.ravel()

            q_base, s_base = get_norm_q(task_means.ravel(), task_stds.ravel(), t_list_test.ravel())
            q_iso = regressor.predict(q_base.ravel()).reshape(np.shape(q_base))

            s_iso = np.diff(q_iso, axis=1) / \
                (t_list_test[0, 1:] - t_list_test[0, :-1]).ravel().reshape(1, -1).repeat(len(task_lbs), axis=0)

            y_iso = get_y_hat(t_list_test.ravel(), s_iso)

            # NLL
            ll_base = - scipy.stats.norm.logpdf(task_lbs.reshape(-1, 1),
                                                loc=task_means.reshape(-1, 1),
                                                scale=task_stds.reshape(-1, 1)).ravel()
            ll_iso = get_log_loss(task_lbs, t_list_test.ravel(), s_iso)
            print([np.mean(ll_base), np.mean(ll_iso)])

            # MSE
            se_base = get_se(y_base, task_lbs)
            se_iso = get_se(y_iso, task_lbs)
            print([np.mean(se_base), np.mean(se_iso)])


def get_iso_cal_table(y, mu, sigma):

    q_raw = scipy.stats.norm.cdf(y, loc=mu.reshape(-1, 1), scale=sigma.reshape(-1, 1))
    q_list, idx = np.unique(q_raw, return_inverse=True)

    q_hat_list = np.zeros_like(q_list)
    for i in range(0, len(q_list)):
        q_hat_list[i] = np.mean(q_raw <= q_list[i])
    q_hat = q_hat_list[idx]

    return q_raw.ravel(), q_hat.ravel()


def get_cal_table_test(mu, sigma, t_list_test):

    n_t = np.shape(t_list_test)[1]

    n_y = np.shape(mu)[0]

    t = t_list_test.repeat(n_y, axis=1).reshape(-1, 1)

    mu_cal = mu.reshape(1, -1).repeat(n_t, axis=0).reshape(-1, 1)

    sigma_cal = sigma.reshape(1, -1).repeat(n_t, axis=0).reshape(-1, 1)

    ln_s = scipy.stats.norm.logcdf(t, loc=mu_cal, scale=sigma_cal)

    ln_ns = scipy.stats.norm.logsf(t, loc=mu_cal, scale=sigma_cal)

    n = np.shape(ln_s)[0]

    s = np.hstack([ln_s, ln_ns, np.ones([n, 1])])

    return s


def get_norm_q(mu, sigma, t_list):

    q = np.zeros([len(mu), len(t_list)])

    s = np.zeros([len(mu), len(t_list)])

    for j in range(0, len(t_list)):
        q[:, j] = np.squeeze(scipy.stats.norm.cdf(t_list[j], loc=mu, scale=sigma))
        s[:, j] = np.squeeze(scipy.stats.norm.pdf(t_list[j], loc=mu, scale=sigma))

    return q, s


def get_log_loss(y, t_list, density_hat):

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

    n_y, n_t = np.shape(density_hat)

    t_list_hat = (t_list[0:-1] + t_list[1:]) / 2

    y_hat = get_y_hat(t_list, density_hat)
    y_var = np.zeros(n_y)

    if len(t_list_hat) == n_t:

        for i in range(0, n_y):

            y_py = ((t_list_hat-y_hat[i])**2) * density_hat[i, :]

            y_var[i] = scipy.integrate.trapz(y_py, t_list_hat)

    else:
        for i in range(0, n_y):

            y_py = ((t_list - y_hat[i])**2) * density_hat[i, :]

            y_var[i] = scipy.integrate.trapz(y_py, t_list)

    return y_var


def get_se(y, y_hat):

    se = (np.squeeze(y) - np.squeeze(y_hat))**2

    return se


def get_q_y(y, q, t_list):

    q_y = np.zeros(len(y))

    for i in range(0, len(y)):
        t_loc = np.argmax(t_list > y[i])
        q_y[i] = q[i, t_loc]

    return q_y


def get_cal_error(q_y):

    ce = np.zeros(20)

    q_list = np.linspace(0, 1, 21)[1:-1]

    q_hat = np.zeros_like(q_list)

    for i in range(0, len(q_list)):
        q_hat[i] = np.mean(q_y <= q_list[i])

    ce[1:20] = (q_list.ravel() - q_hat.ravel())**2

    ce[0] = np.mean(ce[1:20])

    return ce
