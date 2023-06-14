"""
The utility function for model construction.

This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/nn_utils.py
"""
import torch
from torch import nn


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def index_select_nd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def select_neighbor_and_aggregate(feature, index):
    """
    The basic operation in message passing.
    Caution: the index_selec_ND would cause the reproducibility issue when performing the training on CUDA.
    See: https://pytorch.org/docs/stable/notes/randomness.html
    :param feature: the candidate feature for aggregate. (n_nodes, hidden)
    :param index: the selected index (neighbor indexes).
    :return:
    """
    neighbor = index_select_nd(feature, index)
    return neighbor.sum(dim=1)


def get_model_args():
    """
    Get model structure related parameters

    :return: a list containing parameters
    """
    return ['model_type', 'ensemble_size', 'input_layer', 'hidden_size', 'bias', 'depth',
            'dropout', 'activation', 'undirected', 'ffn_hidden_size', 'ffn_num_layers',
            'atom_message', 'weight_decay', 'select_by_loss', 'skip_epoch', 'backbone',
            'embedding_output_type', 'self_attention', 'attn_hidden', 'attn_out', 'dense',
            'distinct_init', 'aug_rate', 'fine_tune_coff', 'nencoders',
            'dist_coff', 'no_attach_fea', 'coord', "num_attn_head", "num_mt_block"]
