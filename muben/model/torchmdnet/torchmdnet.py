"""
# Author: Yinghao Li
# Modified: February 27th, 2024
# ---------------------------------------
# Description: Implementation of (adjusted) TorchMD-NET.
# Reference: Modified from https://github.com/shehzaidi/pre-training-via-denoising.
"""

import re
import torch
import logging

from torch import nn
from torch_scatter import scatter

from .et import TorchMDET
from .layers import EquivariantScalar, EquivariantVectorOutput
from .modules import act_class_mapping

from muben.layers import OutputLayer

logger = logging.getLogger(__name__)


class TorchMDNET(nn.Module):
    """
    Modified TorchMD-NET model implementation for molecular property prediction.

    Attributes
    ----------
    representation_model : nn.Module
        The primary representation model.
    output_model : EquivariantScalar
        Scalar model for output.
    output_model_noise : EquivariantVectorOutput
        Vector model for noise output.
    dropout : nn.Dropout
        Dropout layer.
    activation : nn.Module
        Activation function module.
    linear : nn.Linear
        Linear transformation layer.
    output_layer : OutputLayer
        Layer for producing the final output.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : object
            Configuration object with model parameters and hyperparameters.
        """
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
        self.output_model_noise = EquivariantVectorOutput(config["embedding_dimension"], config["activation"])

        self.dropout = nn.Dropout(config.dropout)
        self.activation = act_class_mapping[config.activation]()
        self.linear = nn.Linear(config.embedding_dimension // 2, config.embedding_dimension // 2)

        self.output_layer = OutputLayer(
            config.embedding_dimension // 2,
            config.n_lbs * config.n_tasks,
            config.uncertainty_method,
            task_type=config.task_type,
            bbp_prior_sigma=config.bbp_prior_sigma,
        )

        self.reset_parameters()

        ckpt = torch.load(config.checkpoint_path, map_location="cpu")
        self.load_from_checkpoint(ckpt)

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        self.linear.reset_parameters()
        self.output_layer.initialize()
        return None

    def load_from_checkpoint(self, ckpt):
        state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
        loading_return = self.load_state_dict(state_dict, strict=False)

        if len(loading_return.unexpected_keys) > 0:
            logger.warning(f"Unexpected model layers: {loading_return.unexpected_keys}")
        if len(loading_return.missing_keys) > 0:
            logger.warning(f"Missing model layers: {loading_return.missing_keys}")
        return self

    def forward(self, batch):
        """
        Compute the forward pass of the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input feature including atom ids, coordinates and batch ids.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """
        atoms = batch.atoms
        coords = batch.coords
        mol_ids = batch.mol_ids
        assert atoms.dim() == 1 and atoms.dtype == torch.long

        # run the representation model
        hidden, v, atoms, coords, batch = self.representation_model(atoms, coords, batch=mol_ids)

        # apply the output network
        hidden = self.output_model.pre_reduce(hidden, v, atoms, coords, mol_ids)
        # aggregate atoms
        hidden = scatter(hidden, mol_ids, dim=0, reduce="add")

        out = self.output_layer(self.dropout(self.activation(self.linear(self.dropout(hidden)))))

        return out
