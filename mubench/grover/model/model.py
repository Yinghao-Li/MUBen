"""
The GROVER models for pretraining, finetuning and fingerprint generating.
"""

import logging
import torch
from torch import nn as nn
from typing import List, Dict

from .layers import Readout, GTransEncoder
from .utils import get_activation_function, get_model_args
from ..dataset.molgraph import get_atom_fdim, get_bond_fdim
from mubench.base.model import OutputLayer
from mubench.utils.macro import UncertaintyMethods


logger = logging.getLogger(__name__)


class GROVEREmbedding(nn.Module):
    """
    The GROVER Embedding class. It contains the GTransEncoder.
    This GTransEncoder can be replaced by any validate encoders.
    """

    def __init__(self, args):
        """
        Initialize the GROVEREmbedding class.
        :param args:
        """
        super(GROVEREmbedding, self).__init__()
        self.embedding_output_type = args.embedding_output_type
        edge_dim = get_bond_fdim() + get_atom_fdim()
        node_dim = get_atom_fdim()
        self.encoders = GTransEncoder(args,
                                      hidden_size=args.hidden_size,
                                      edge_fdim=edge_dim,
                                      node_fdim=node_dim,
                                      dropout=args.dropout,
                                      activation=args.activation,
                                      num_mt_block=args.num_mt_block,
                                      num_attn_head=args.num_attn_head,
                                      atom_emb_output=self.embedding_output_type,
                                      bias=args.bias)

    def forward(self, graph_batch: List) -> Dict:
        """
        The forward function takes graph_batch as input and output a dict. The content of the dict is decided by
        self.embedding_output_type.

        :param graph_batch: the input graph batch generated by MolCollator.
        :return: a dict containing the embedding results.
        """
        output = self.encoders(graph_batch)
        if self.embedding_output_type == 'atom':
            return {"atom_from_atom": output[0], "atom_from_bond": output[1],
                    "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        elif self.embedding_output_type == 'bond':
            return {"atom_from_atom": None, "atom_from_bond": None,
                    "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        elif self.embedding_output_type == "both":
            return {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                    "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}


def create_ffn(config):
    """
    Creates the feed-forward network for the model.

    :param config: Arguments.
    """
    # Note: args.features_dim is set according the real loaded features data
    first_linear_dim = config.hidden_size

    activation = get_activation_function(config.activation)
    # Create FFN layers
    if config.ffn_num_layers == 1:
        ffn = [
            nn.Dropout(config.dropout),
        ]
    else:
        ffn = [
            nn.Dropout(config.dropout),
            nn.Linear(first_linear_dim, config.ffn_hidden_size)
        ]
        for _ in range(config.ffn_num_layers - 2):
            ffn.extend([
                activation,
                nn.Dropout(config.dropout),
                nn.Linear(config.ffn_hidden_size, config.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            nn.Dropout(config.dropout),
        ])

    # Create FFN model
    return nn.Sequential(*ffn)


class GROVERFinetuneModel(nn.Module):
    """
    The finetune
    """
    def __init__(self, config):
        super(GROVERFinetuneModel, self).__init__()

        self.hidden_size = config.hidden_size

        self.grover = GROVEREmbedding(config)
        self.readout = Readout(hidden_size=self.hidden_size)

        self.mol_atom_from_atom_ffn = create_ffn(config)
        self.mol_atom_from_bond_ffn = create_ffn(config)

        self.atom_output_layer = OutputLayer(
            config.ffn_hidden_size,
            config.n_lbs*config.n_tasks,
            config.uncertainty_method == UncertaintyMethods.bbp
        ).initialize()
        self.bond_output_layer = OutputLayer(
            config.ffn_hidden_size,
            config.n_lbs*config.n_tasks,
            config.uncertainty_method == UncertaintyMethods.bbp
        ).initialize()

    def forward(self, batch, **kwargs):
        molecule_components = batch.molecule_graphs.components
        _, _, _, _, _, a_scope, _, _ = molecule_components

        output = self.grover(molecule_components)
        # Share readout
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
        bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)

        atom_ffn_output = self.atom_output_layer(atom_ffn_output)
        bond_ffn_output = self.bond_output_layer(bond_ffn_output)

        return atom_ffn_output, bond_ffn_output


def build_model(config):
    """
    Builds a MPNN, which is a message passing neural network + feed-forward layers.

    Parameters
    ----------
    config: Arguments.

    Returns
    -------
    A MPNN containing the MPN encoder along with final linear layers with parameters initialized.
    """
    model = GROVERFinetuneModel(config)

    # Initializes the weights of a model in place.
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model


def load_checkpoint(config):
    """
    Loads a model checkpoint.

    :param config: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :return: The loaded MPNN.
    """

    # Load model and args
    state = torch.load(config.checkpoint_path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    model_args = get_model_args()

    if config is not None:
        for key, value in vars(args).items():
            if key in model_args:
                setattr(config, key, value)

    # Build model
    model = build_model(config)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        new_param_name = param_name
        if new_param_name not in model_state_dict:
            logger.info(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
            logger.info(f'Pretrained parameter "{param_name}" '
                        f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                        f'model parameter of shape {model_state_dict[new_param_name].shape}.')
        else:
            pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
    logger.info(f'Pretrained parameter loaded.')
    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    return model