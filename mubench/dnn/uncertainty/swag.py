"""
Implements SWA and SWAG algorithms: https://arxiv.org/abs/1902.02476

Modified from the PyTorch implementation at torch.optim.swa_utils.py
and the original SWAG repo: https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
"""

import torch
import regex
import logging
from torch.nn import Module
from copy import deepcopy

logger = logging.getLogger(__name__)


class SWAModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA).

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`

    .. note::
        When using SWA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        You can do so by using :meth:`torch.optim.swa_utils.update_bn` utility.

    .. note::
        :attr:`avg_fn` is not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    """

    def __init__(self, model, k_models, var_clamp=1e-30, device='cpu'):
        super(SWAModel, self).__init__()

        self.module = deepcopy(model)
        self.k_models = k_models
        self.var_clamp = var_clamp

        self.module.to(device)

        for name, param in self.module.named_parameters():
            name = regex.sub(r"\.", "_", name)
            self.register_buffer(f"{name}_Mean", torch.zeros_like(param.data))
            self.register_buffer(f"{name}_SqMean", torch.zeros_like(param.data))
            self.register_buffer(f"{name}_CovMatSqrt", torch.zeros((1, param.data.numel())))

        self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long, device=device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):

        for name, params in model.named_parameters():

            name_ = regex.sub(r"\.", "_", name)

            mean = self.__getattr__(f"{name_}_Mean")
            sq_mean = self.__getattr__(f"{name_}_SqMean")
            cov_mat_sqrt = self.__getattr__(f"{name_}_CovMatSqrt")

            if self.n_averaged == 0:
                self.__setattr__(f"{name_}_Mean", params.data.cpu())
                self.__setattr__(f"{name_}_SqMean", params.data.square().cpu())

            else:
                # first moment
                mean = mean * self.n_averaged / (self.n_averaged + 1.0) + params.data.cpu() / (self.n_averaged + 1.0)
                # second moment
                sq_mean = sq_mean * self.n_averaged / (self.n_averaged + 1.0) + \
                    params.data.square().cpu() / (self.n_averaged + 1.0)
                # block covariance matrices, store deviation from current mean
                dev = (params.data.cpu() - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).T), dim=0)
                # remove first column if we have stored too many models
                if cov_mat_sqrt.shape[0] > self.k_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]

                self.__setattr__(f"{name_}_Mean", mean)
                self.__setattr__(f"{name_}_SqMean", sq_mean)
                self.__setattr__(f"{name_}_CovMatSqrt", cov_mat_sqrt)

        self.n_averaged += 1

        return self

    def sample_parameters(self):

        scale = torch.sqrt(torch.tensor(0.5))

        for name, _ in self.module.named_parameters():

            name_ = regex.sub(r"\.", "_", name)

            mean = self.__getattr__(f"{name_}_Mean").cpu()
            sq_mean = self.__getattr__(f"{name_}_SqMean").cpu()
            cov_mat_sqrt = self.__getattr__(f"{name_}_CovMatSqrt").cpu()

            eps = torch.randn_like(mean)
            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
            scaled_diag_sample = scale * torch.sqrt(var) * eps

            eps = torch.randn((cov_mat_sqrt.shape[0], 1))
            cov_sample = (scale / (self.k_models - 1) ** 0.5) * cov_mat_sqrt.T.matmul(eps).view_as(mean)

            w = mean + scaled_diag_sample + cov_sample

            self.module.get_parameter(name).data = w

        return self


@torch.no_grad()
def update_bn(model, training_loader, device: str = 'cpu', hf_training=False):
    r"""
    Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        training_loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for batch in training_loader:
        batch.to(device)

        if hf_training:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                model(batch)
        else:
            model(batch)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
