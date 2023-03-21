"""
The general utility functions.
"""
import csv
import logging
import os
import pickle
import random
from collections import defaultdict
from typing import List, Set, Tuple, Union, Dict

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn as nn
from tqdm import tqdm as core_tqdm

from ..data.moldataset import MoleculeDatapoint, MoleculeDataset
from ..data.scaler import StandardScaler
from ..model.models import GroverFinetuneTask
from .nn_utils import initialize_weights
from .scheduler import NoamLR

logger = logging.getLogger(__name__)


def get_model_args():
    """
    Get model11 structure related parameters

    :return: a list containing parameters
    """
    return ['model_type', 'ensemble_size', 'input_layer', 'hidden_size', 'bias', 'depth',
            'dropout', 'activation', 'undirected', 'ffn_hidden_size', 'ffn_num_layers',
            'atom_message', 'weight_decay', 'select_by_loss', 'skip_epoch', 'backbone',
            'embedding_output_type', 'self_attention', 'attn_hidden', 'attn_out', 'dense',
            'bond_drop_rate', 'distinct_init', 'aug_rate', 'fine_tune_coff', 'nencoders',
            'dist_coff', 'no_attach_fea', 'coord', "num_attn_head", "num_mt_block"]


def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    Loads features saved in a variety of formats.

    Supported formats:
    - .npz compressed (assumes features are saved with name "features")

    All formats assume that the SMILES strings loaded elsewhere in the code are in the same
    order as the features loaded here.

    :param path: Path to a file containing features.
    :return: A 2D numpy array of size (num_molecules, features_size) containing the features.
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        npz_file = np.load(path)
        keys = list(npz_file.keys())
        assert len(keys) == 1, f'Expected 1 key in npz file, got {len(keys)}'
        features = npz_file[keys[0]]
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


class tqdm(core_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ascii", True)
        super(tqdm, self).__init__(*args, **kwargs)


def get_task_names(path: str, use_compound_names: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.

    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    datapoint_list = []
    for idx, datapoint in enumerate(data):
        if datapoint.smiles == '':
            print(f'invalid smiles {idx}: {datapoint.smiles}')
            continue
        mol = Chem.MolFromSmiles(datapoint.smiles)
        if mol.GetNumHeavyAtoms() == 0:
            print(f'invalid heavy {idx}')
            continue
        datapoint_list.append(datapoint)
    return MoleculeDataset(datapoint_list)


def get_data(path: str,
             skip_invalid_smiles: bool = True,
             config=None,
             max_data_size: int = None,
             use_compound_names: bool = None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    Parameters
    ----------
    path: Path to a CSV file.
    skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    config: Arguments.
    max_data_size: The maximum number of data points to load.
    use_compound_names: Whether file has compound names in addition to smiles strings.

    Returns
    -------
    A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """

    if config is not None:
        # Prefer explicit function arguments but default to args if not provided
        max_data_size = max_data_size if max_data_size is not None else config.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else config.use_compound_names
    else:
        use_compound_names = False

    max_data_size = max_data_size or np.inf

    # Load features
    features_data = None
    if config is not None:
        config.features_dim = 0

    skip_smiles = set()

    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        lines = []
        for line in reader:
            smiles = line[0]

            if smiles in skip_smiles:
                continue

            lines.append(line)

            if len(lines) >= max_data_size:
                break

        data = MoleculeDataset([
            MoleculeDatapoint(
                line=line,
                config=config,
                features=features_data[i] if features_data is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(lines), total=len(lines), disable=True)
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            logger.info(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool = True, config=None) -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.

    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param config: arguments.
    :return: A MoleculeDataset with all of the provided SMILES.
    """

    data = MoleculeDataset([MoleculeDatapoint(line=[smile], config=config) for smile in smiles])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            logger.warning(f'{original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               config=None) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """
    Splits data into training, validation, and test splits.

    Parameters
    ----------
    data: A MoleculeDataset.
    split_type: Split type.
    sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    seed: The random seed to use before shuffling data.
    config: Namespace of arguments.

    Returns
    -------
    A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if config is not None:
        folds_file, val_fold_index, test_fold_index = \
            config.folds_file, config.val_fold_index, config.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'crossval':
        index_set = config.crossval_index_sets[config.seed]
        data_split = []
        for split in range(3):
            split_indices = []
            for index in index_set[split]:
                with open(os.path.join(config.crossval_index_dir, f'{index}.pkl'), 'rb') as rf:
                    split_indices.extend(pickle.load(rf))
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'index_predetermined':
        split_indices = config.crossval_index_sets[config.seed]
        assert len(split_indices) == 3
        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index:
            assert sizes[2] == 0  # test set is created separately so use all of the other data for train and val
        assert folds_file is not None
        assert test_fold_index is not None

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2
        # assert len(data) == sum([len(fold_indices) for fold_indices in all_fold_indices])

        log_scaffold_stats(data, all_fold_indices)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.seed(seed)
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed)

    elif split_type == 'random':
        data.shuffle(seed=seed)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


def get_class_sizes(data: MoleculeDataset) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.

    :param data: A classification dataset
    :return: A list of lists of class proportions. Each inner list contains the class proportions
    for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        # Make sure we're dealing with a binary classification task
        assert set(np.unique(task_targets)) <= {0, 1}

        try:
            ones = np.count_nonzero(task_targets) / len(task_targets)
        except ZeroDivisionError:
            ones = float('nan')
            print('Warning: class has no targets')
        class_sizes.append([1 - ones, ones])

    return class_sizes


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(data: MoleculeDataset,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   seed: int = 0) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    Parameters
    ----------
    data: A MoleculeDataset.
    sizes: A length-3 tuple with the proportions of data in the train, validation, and test sets.
    balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    seed: Seed for shuffling when doing balanced splitting.

    Returns
    -------
    A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.smiles(), use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    # Sort from largest to smallest scaffold sets
    else:
        index_sets = sorted(list(scaffold_to_indices.values()), key=lambda idx_set: len(idx_set), reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    logger.info(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                f'train scaffolds = {train_scaffold_count:,} | '
                f'val scaffolds = {val_scaffold_count:,} | '
                f'test scaffolds = {test_scaffold_count:,}')

    log_scaffold_stats(data, index_sets)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def log_scaffold_stats(data: MoleculeDataset,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular scaffolds.

    :param data: A MoleculeDataset.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_scaffolds: The number of scaffolds about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :return: A list of tuples where each tuple contains a list of average target values
    across the first num_labels labels and a list of the number of non-zero values for
    the first num_scaffolds scaffolds, sorted in decreasing order of scaffold frequency.
    """
    # print some statistics about scaffolds
    target_avgs = []
    counts = []
    for index_set in index_sets:
        data_set = [data[i] for i in index_set]
        targets = [d.targets for d in data_set]
        targets = np.array(targets, dtype=float)
        target_avgs.append(np.nanmean(targets, axis=0))
        counts.append(np.count_nonzero(~np.isnan(targets), axis=0))
    stats = [(target_avgs[i][:num_labels], counts[i][:num_labels]) for i in range(min(num_scaffolds, len(target_avgs)))]

    logger.info('Label averages per scaffold, in decreasing order of scaffold frequency,'
                f'capped at {num_scaffolds} scaffolds and {num_labels} labels: {stats}')

    return stats


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def load_args(path: str):
    """
    Loads the arguments a model11 was trained with.

    :param path: Path where model11 checkpoint is saved.
    :return: The arguments Namespace that the model11 was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage)['args']


def get_ffn_layer_id(model: GroverFinetuneTask):
    """
    Get the ffn layer id for GroverFinetune Task. (Adhoc!)
    :param model:
    :return:
    """
    return [id(x) for x in model.state_dict() if "grover" not in x and "ffn" in x]


def build_optimizer(model: nn.Module, config):
    """
    Builds an Optimizer.

    :param model: The model11 to optimize.
    :param config: Arguments.
    :return: An initialized Optimizer.
    """

    # Only adjust the learning rate for the GroverFinetuneTask.
    if type(model) == GroverFinetuneTask:
        ffn_params = get_ffn_layer_id(model)
    else:
        # if not, init adam optimizer normally.
        return torch.optim.Adam(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)
    base_params = filter(lambda p: id(p) not in ffn_params, model.parameters())
    ffn_params = filter(lambda p: id(p) in ffn_params, model.parameters())
    if config.fine_tune_coff == 0:
        for param in base_params:
            param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': config.init_lr * config.fine_tune_coff},
        {'params': ffn_params, 'lr': config.init_lr}
    ], lr=config.init_lr, weight_decay=config.weight_decay)

    return optimizer


def build_lr_scheduler(optimizer, config):
    """
    Builds a learning rate scheduler.

    param optimizer: The Optimizer whose learning rate will be scheduled.
    param args: Arguments.
    param total_epochs: The total number of epochs for which the model11 will be task.
    return: An initialized learning rate scheduler.
    """

    # Learning rate scheduler
    # Divide the parameter into two groups for the finetune.
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        steps_per_epoch=config.train_data_size // config.batch_size,
        init_lr=config.init_lr,
        max_lr=config.max_lr,
        final_lr=config.final_lr,
        fine_tune_coff=config.fine_tune_coff
    )


def load_checkpoint(path: str,
                    current_args=None,
                    cuda: bool = None):
    """
    Loads a model11 checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model11 to cuda.
    :return: The loaded MPNN.
    """

    # Load model11 and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    model_ralated_args = get_model_args()

    if current_args is not None:
        for key, value in vars(args).items():
            if key in model_ralated_args:
                setattr(current_args, key, value)
    else:
        current_args = args

    # args.cuda = cuda if cuda is not None else args.cuda

    # Build model11
    model = build_model(current_args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        new_param_name = param_name
        if new_param_name not in model_state_dict:
            logger.info(f'Pretrained parameter "{param_name}" cannot be found in model11 parameters.')
        elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
            logger.info(f'Pretrained parameter "{param_name}" '
                        f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                        f'model11 parameter of shape {model_state_dict[new_param_name].shape}.')
        else:
            pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
    logger.info(f'Pretrained parameter loaded.')
    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        logger.info('Moving model11 to cuda')
        model = model.cuda()

    return model


def get_loss_func(config, model=None):
    """
    Gets the loss function corresponding to a given dataset type.

    :param config: Namespace containing the dataset type ("classification" or "regression").
    :param model: model11
    :return: A PyTorch loss function.
    """
    if hasattr(model, "get_loss_func"):
        return model.get_loss_func(config)
    if config.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')
    if config.dataset_type == 'regression':
        return nn.MSELoss(reduction='none')

    raise ValueError(f'Dataset type "{config.dataset_type}" not supported.')


def load_scalars(path: str):
    """
    Loads the scalars a model11 was trained with.

    :param path: Path where model11 checkpoint is saved.
    :return: A tuple with the data scaler and the features scaler.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return scaler, features_scaler


def save_checkpoint(path: str,
                    model,
                    scaler,
                    features_scaler,
                    config=None):
    """
    Saves a model11 checkpoint.

    param model11: A MPNN.
    param scaler: A StandardScaler fitted on the data.
    param features_scaler: A StandardScaler fitted on the features.
    param args: Arguments namespace.
    param path: Path where checkpoint will be saved.
    """
    state = {
        'args': config,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    torch.save(state, path)


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

    model = GroverFinetuneTask(config)
    initialize_weights(model=model, model_idx=model_idx)
    return model
