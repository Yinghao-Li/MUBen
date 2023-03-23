"""
The general utility functions.
"""
import logging

import torch

from mubench.utils.scaler import StandardScaler
from .model import GROVERFinetuneModel
from .utils_nn import initialize_weights

logger = logging.getLogger(__name__)


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


def load_scalars(path: str):
    """
    Loads the scalars a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data scaler and the features scaler.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return scaler, features_scaler


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
