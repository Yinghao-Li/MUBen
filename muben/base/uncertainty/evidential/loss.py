"""
# Author: Yinghao Li
# Created: July 19th, 2023
# Modified: July 19th, 2023
# ---------------------------------------
# Description: Implementation of evidential loss functions.
               Modified from https://github.com/aamini/evidential-deep-learning
"""

import torch
import numpy as np

import torch.nn.functional as F


# --- continuous (regression) ---
class EvidentialRegressionLoss:
    """
    Evidential Regression Loss
    """
    def __init__(self, coeff=1.0, **kwargs):
        self._coeff = coeff

    @staticmethod
    def nig_nll(y: torch.Tensor,
                gamma: torch.Tensor,
                nu: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor):
        inter = 2 * beta * (1 + nu)

        nll = 0.5 * (np.pi / nu).log() \
            - alpha * inter.log() \
            + (alpha + 0.5) * (nu * (y - gamma) ** 2 + inter).log() \
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        return nll

    @staticmethod
    def nig_reg(y, gamma, nu, alpha):

        error = (y - gamma).abs()
        evidence = 2. * nu + alpha

        return error * evidence

    def __call__(self, logits, lbs):

        gamma, nu, alpha, beta = torch.split(logits, 1, dim=-1)

        loss_nll = self.nig_nll(lbs, gamma.view(-1), nu.view(-1), alpha.view(-1), beta.view(-1))
        loss_reg = self.nig_reg(lbs, gamma.view(-1), nu.view(-1), alpha.view(-1))

        return loss_nll + self._coeff * loss_reg


class EvidentialClassificationLoss:
    def __init__(self, n_classes, n_steps_per_epoch, annealing_epochs=10, device='cpu'):
        self._n_classes = n_classes
        self._n_steps_per_epoch = n_steps_per_epoch
        self._annealing_epochs = annealing_epochs
        self._device = device
        self._i_epoch = 0
        self._i_step = 0

    def kl_divergence(self, alpha):
        ones = torch.ones([1, self._n_classes], device=self._device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    @staticmethod
    def loglikelihood_loss(y, alpha):
        s = torch.sum(alpha, dim=1, keepdim=True)

        loglikelihood_err = torch.sum((y - (alpha / s)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(alpha * (s - alpha) / (s * s * (s + 1)), dim=1, keepdim=True)

        loglikelihood = loglikelihood_err + loglikelihood_var

        return loglikelihood

    def mse_loss(self, y, alpha):
        loglikelihood = self.loglikelihood_loss(y, alpha)

        kl_alpha = (alpha - 1) * (1 - y) + 1

        annealing_coef = min(1.0, self._i_epoch / self._annealing_epochs)
        kl_div = annealing_coef * self.kl_divergence(kl_alpha)

        return loglikelihood + kl_div

    def update_idx(self):
        self._i_step += 1
        if self._i_step % self._n_steps_per_epoch == 0:
            self._i_epoch += 1

    def __call__(self, inputs, targets):
        evidence = F.relu(inputs)
        alpha = evidence + 1
        targets = targets.unsqueeze(-1)
        loss = self.mse_loss(targets, alpha)
        self.update_idx()
        return loss
