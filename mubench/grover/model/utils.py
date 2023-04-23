"""
The utility function for model construction.

This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/nn_utils.py
"""
import logging
import torch
from torch import nn

from .model import GROVERFinetuneModel

logger = logging.getLogger(__name__)


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


def initialize_weights(model: nn.Module, distinct_init=False, model_idx=0):
    """
    Initializes the weights of a model in place.
    """
    init_fns = [nn.init.kaiming_normal_, nn.init.kaiming_uniform_,
                nn.init.xavier_normal_, nn.init.xavier_uniform_]
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            if distinct_init:
                init_fn = init_fns[model_idx % 4]
                if 'kaiming' in init_fn.__name__:
                    init_fn(param, nonlinearity='relu')
                else:
                    init_fn(param)
            else:
                nn.init.xavier_normal_(param)


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
            'bond_drop_rate', 'distinct_init', 'aug_rate', 'fine_tune_coff', 'nencoders',
            'dist_coff', 'no_attach_fea', 'coord', "num_attn_head", "num_mt_block"]


def build_model(config, model_idx=0):
    """
    Builds a MPNN, which is a message passing neural network + feed-forward layers.

    Parameters
    ----------
    config: Arguments.
    model_idx: model index

    Returns
    -------
    A MPNN containing the MPN encoder along with final linear layers with parameters initialized.
    """
    if hasattr(config, 'num_tasks'):
        config.output_size = config.num_tasks
    else:
        config.output_size = 1

    model = GROVERFinetuneModel(config)
    initialize_weights(model=model, model_idx=model_idx)
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
