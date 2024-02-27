"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description:

SGLD (Stochastic Gradient Langevin Dynamics) optimizers

# Reference:
https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics/optimizers.py

"""

import torch
import numpy as np
from torch.optim.optimizer import Optimizer

__all__ = ["SGLDOptimizer", "PSGLDOptimizer"]


class SGLDOptimizer(Optimizer):
    def __init__(self, params, lr, norm_sigma=0.1, addnoise=True):
        """
        SGLD (Stochastic Gradient Langevin Dynamics) optimiser based on pytorch's SGD.

        Parameters
        ----------
        params: iterable
            Parameters to be optimized.
        lr: float
            Learning rate.
        norm_sigma: float, optional
            Sigma for gaussian prior (default is 0.1).
        addnoise: bool, optional
            Whether to add Langevin noise (default is True).

        Note: The weight decay is specified in terms of the gaussian prior sigma.
        """
        weight_decay = 1 / (norm_sigma**2)

        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(lr=lr, weight_decay=weight_decay, addnoise=addnoise)

        super(SGLDOptimizer, self).__init__(params, defaults)

    def step(self, *args, **kwargs):
        """
        Performs a single optimization step.

        Returns
        -------
        torch.Tensor:
            The loss after the optimization step.
        """
        loss = None

        for group in self.param_groups:
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if group["addnoise"]:
                    langevin_noise = p.data.new(p.data.size()).normal_(
                        mean=0, std=1
                    ) / np.sqrt(group["lr"])
                    p.data.add_(-group["lr"], 0.5 * d_p + langevin_noise)
                else:
                    p.data.add_(-group["lr"], 0.5 * d_p)

        return loss


class PSGLDOptimizer(Optimizer):
    def __init__(
        self,
        params,
        lr,
        norm_sigma=0.1,
        alpha=0.99,
        eps=1e-8,
        centered=False,
        addnoise=True,
    ):
        """
        RMSprop preconditioned SGLD (Stochastic Gradient Langevin Dynamics) using pytorch rmsprop implementation.

        Parameters
        ----------
        params: iterable
            Parameters to be optimized.
        lr: float
            Learning rate.
        norm_sigma: float, optional
            Sigma for gaussian prior (default is 0.1).
        alpha: float, optional
            Factor for running average (default is 0.99).
        eps: float, optional
            Term added to denominator for numerical stability (default is 1e-8).
        centered: bool, optional
            Whether to compute a centered RMSProp (default is False).
        addnoise: bool, optional
            Whether to add Langevin noise (default is True).
        """
        weight_decay = 1 / (norm_sigma**2)

        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            alpha=alpha,
            eps=eps,
            centered=centered,
            addnoise=addnoise,
        )
        super(PSGLDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PSGLDOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("centered", False)

    def step(self, *args, **kwargs):
        """
        Performs a single optimization step.

        Returns
        -------
        torch.Tensor:
            The loss after the optimization step.
        """
        loss = None

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.zeros_like(p.data)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p.data)

                square_avg = state["square_avg"]
                alpha = group["alpha"]
                state["step"] += 1

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(1 - alpha, d_p)
                    avg = (
                        square_avg.cmul(-1, grad_avg, grad_avg)
                        .sqrt()
                        .add_(group["eps"])
                    )
                else:
                    avg = square_avg.sqrt().add_(group["eps"])

                if group["addnoise"]:
                    langevin_noise = p.data.new(p.data.size()).normal_(
                        mean=0, std=1
                    ) / np.sqrt(group["lr"])
                    p.data.add_(
                        -group["lr"],
                        0.5 * d_p.div_(avg) + langevin_noise / torch.sqrt(avg),
                    )

                else:
                    p.data.addcdiv_(-group["lr"], 0.5 * d_p, avg)

        return loss
