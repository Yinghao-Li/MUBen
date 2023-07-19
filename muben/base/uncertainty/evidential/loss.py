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

import torch.nn as nn
import torch.nn.functional as F

mse = nn.MSELoss(reduction='mean')
bce_loss = torch.nn.BCEWithLogitsLoss()


# --- continuous (regression) ---

def reduce(val, reduction):
    if reduction == 'mean':
        val = val.mean()
    elif reduction == 'sum':
        val = val.sum()
    elif reduction == 'none':
        pass
    else:
        raise ValueError(f"Invalid reduction argument: {reduction}")
    return val


class EvidentialRegressionLoss:
    """
    Evidential Regression Loss
    """
    def __init__(self, coeff=1.0, reduction='mean'):
        self._coeff = coeff
        self._reduction = reduction

    def nig_nll(self,
                y: torch.Tensor,
                gamma: torch.Tensor,
                nu: torch.Tensor,
                alpha: torch.Tensor,
                beta: torch.Tensor):
        inter = 2 * beta * (1 + nu)

        nll = 0.5 * (np.pi / nu).log() \
            - alpha * inter.log() \
            + (alpha + 0.5) * (nu * (y - gamma) ** 2 + inter).log() \
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        return reduce(nll, reduction=self._reduction)

    def nig_reg(self, y, gamma, nu, alpha):

        error = (y - gamma).abs()
        evidence = 2. * nu + alpha

        return reduce(error * evidence, reduction=self._reduction)

    def __call__(self, logits, lbs):

        gamma, nu, alpha, beta = torch.split(logits, 1, dim=-1)

        loss_nll = self.nig_nll(lbs, gamma.view(-1), nu.view(-1), alpha.view(-1), beta.view(-1))
        loss_reg = self.nig_reg(lbs, gamma.view(-1), nu.view(-1), alpha.view(-1))

        return loss_nll + self._coeff * loss_reg


# --- discrete (classification) ---

def dirichlet_sos(y, outputs, device=None):
    return edl_log_loss(outputs, y, device=device if device else outputs.device)


def dirichlet_evidence(outputs):
    """Calculate ReLU evidence"""
    return relu_evidence(outputs)


def dirichlet_matches(predictions, labels):
    """Calculate the number of matches from index predictions"""
    assert predictions.shape == labels.shape, f"Dimension mismatch between predictions " \
                                              f"({predictions.shape}) and labels ({labels.shape})"
    return torch.reshape(torch.eq(predictions, labels).float(), (-1, 1))


def dirichlet_predictions(outputs):
    """Calculate predictions from logits"""
    return torch.argmax(outputs, dim=1)


def dirichlet_uncertainty(outputs):
    """Calculate uncertainty from logits"""
    alpha = relu_evidence(outputs) + 1
    return alpha.size(1) / torch.sum(alpha, dim=1, keepdim=True)


def sigmoid_ce(y, y_logits, device=None):
    return bce_loss(y_logits, y, device=device if device else y_logits.device)


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def edl_loss(func, y, alpha, device=None):
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = kl_divergence(kl_alpha, y.shape[1], device=device)
    return A + kl_div


def edl_log_loss(output, target, device=None):
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha, device=device))
    assert loss is not None
    return loss
