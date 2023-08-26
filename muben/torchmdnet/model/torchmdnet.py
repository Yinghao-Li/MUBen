"""
# Author: Yinghao Li
# Modified: August 5th, 2023
# ---------------------------------------
# Description: Implementation of (adjusted) TorchMD-NET.
               Modified from https://github.com/shehzaidi/pre-training-via-denoising.
"""


import re
import torch
import logging

from torch import nn
from torch_scatter import scatter

from .et import TorchMDET
from .layers import EquivariantScalar, EquivariantVectorOutput
from .modules import act_class_mapping

from muben.base.model import OutputLayer

logger = logging.getLogger(__name__)


class TorchMDNET(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.representation_model = TorchMDET(
            attn_activation=config["attn_activation"],
            num_heads=config["num_heads"],
            distance_influence=config["distance_influence"],
            layernorm_on_vec=config["layernorm_on_vec"],
            hidden_channels=config["embedding_dimension"],
            num_layers=config["num_layers"],
            num_rbf=config["num_rbf"],
            rbf_type=config["rbf_type"],
            trainable_rbf=config["trainable_rbf"],
            activation=config["activation"],
            neighbor_embedding=config["neighbor_embedding"],
            cutoff_lower=config["cutoff_lower"],
            cutoff_upper=config["cutoff_upper"],
            max_z=config["max_z"],
            max_num_neighbors=config["max_num_neighbors"],
        )
        self.output_model = EquivariantScalar(
            config["embedding_dimension"],
            config["activation"],
            external_output_layer=True,
        )
        self.output_model_noise = EquivariantVectorOutput(
            config["embedding_dimension"], config["activation"]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.activation = act_class_mapping[config.activation]()
        self.linear = nn.Linear(
            config.embedding_dimension // 2, config.embedding_dimension // 2
        )

        self.output_layer = OutputLayer(
            config.embedding_dimension // 2,
            config.n_lbs * config.n_tasks,
            config.uncertainty_method,
            task_type=config.task_type,
            bbp_prior_sigma=config.bbp_prior_sigma,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        self.linear.reset_parameters()
        self.output_layer.initialize()
        return None

    def load_from_checkpoint(self, ckpt):
        state_dict = {
            re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()
        }
        loading_return = self.load_state_dict(state_dict, strict=False)

        if len(loading_return.unexpected_keys) > 0:
            logger.warning(f"Unexpected model layers: {loading_return.unexpected_keys}")
        if len(loading_return.missing_keys) > 0:
            logger.warning(f"Missing model layers: {loading_return.missing_keys}")
        return self

    def forward(self, batch):
        atoms = batch.atoms
        coords = batch.coords
        mol_ids = batch.mol_ids
        assert atoms.dim() == 1 and atoms.dtype == torch.long

        # run the representation model
        hidden, v, atoms, coords, batch = self.representation_model(
            atoms, coords, batch=mol_ids
        )

        # apply the output network
        hidden = self.output_model.pre_reduce(hidden, v, atoms, coords, mol_ids)
        # aggregate atoms
        hidden = scatter(hidden, mol_ids, dim=0, reduce="add")

        out = self.output_layer(
            self.dropout(self.activation(self.linear(self.dropout(hidden))))
        )

        return out


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""

    def __init__(self, accumulator_shape: tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return (batch - self.mean) / self.std
