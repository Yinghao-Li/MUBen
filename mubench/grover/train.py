"""
The training function used in the finetuning task.
"""
import logging
import os
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from mubench.grover.data.molgraph import MolCollator
from mubench.grover.data.moldataset import MoleculeDataset
from mubench.grover.data.scaler import StandardScaler
from mubench.grover.util.metrics import get_metric_func
from mubench.grover.util.nn_utils import initialize_weights, param_count
from mubench.grover.util.scheduler import NoamLR
from mubench.grover.util.config import GroverConfig
from mubench.grover.util.utils import (
    build_optimizer,
    build_lr_scheduler,
    makedirs,
    load_checkpoint,
    get_loss_func,
    save_checkpoint,
    build_model,
    get_class_sizes,
    get_data,
    split_data,
    get_task_names
)

from mubench.grover.task.predict import predict, evaluate, evaluate_predictions

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
def train(model, data, loss_func, optimizer, scheduler, shared_dict, config: GroverConfig, n_iter: int = 0):
    """
    Trains a model11 for an epoch.

    Parameters
    ----------
    model: Model.
    data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    loss_func: Loss function.
    optimizer: An Optimizer.
    scheduler: A learning rate scheduler.
    shared_dict: N/A
    config: Arguments.
    n_iter: The number of iterations (training examples) trained on so far.

    Returns
    -------
    The total number of iterations (training examples) trained on so far.
    """
    # debug = logger.debug if logger is not None else print

    model.train()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0

    mol_collator = MolCollator(shared_dict=shared_dict, args=config)

    num_workers = 4
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True,
                                num_workers=num_workers, collate_fn=mol_collator)

    for _, item in enumerate(mol_loader):
        _, batch, _, mask, targets = item
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if config.cuda:
            class_weights = class_weights.cuda()

        # Run model11
        model.zero_grad()
        preds = model(batch)
        loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += config.batch_size

        cum_loss_sum += loss.item()
        cum_iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += config.batch_size

    return n_iter, cum_loss_sum / cum_iter_count


# noinspection PyUnresolvedReferences
def run_training(config: GroverConfig) -> List[float]:
    """
    Trains a model11 and returns test scores on the model11 checkpoint with the highest validation score.

    Parameters
    ----------
    config: Arguments.

    Returns
    -------
    A list of ensemble scores for each task.
    """

    # pin GPU to local rank.
    idx = config.gpu
    if config.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(idx)

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(config)

    metric_func = get_metric_func(metric=config.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), config.num_tasks))

    ensemble_scores = 0.0
    # Train ensemble of models
    for model_idx in range(config.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(config.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model11
        if config.checkpoint_paths is not None:
            if len(config.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            logger.info(f'Loading model {cur_model} from {config.checkpoint_paths[cur_model]}')
            model = load_checkpoint(config.checkpoint_paths[cur_model], current_args=config)
        else:
            logger.info(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, config=config)

        if config.fine_tune_coff != 1 and config.checkpoint_paths is not None:
            logger.info("Fine tune fc layer with different lr")
            initialize_weights(model_idx=model_idx, model=model.ffn, distinct_init=config.distinct_init)

        # Get loss and metric functions
        loss_func = get_loss_func(config, model)

        optimizer = build_optimizer(model, config)

        # debug(model11)
        logger.info(f'Number of parameters = {param_count(model):,}')
        if config.cuda:
            logger.info('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, config)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, config)

        # Bulid data_loader
        shuffle = True
        mol_collator = MolCollator(shared_dict={}, args=config)
        train_data = DataLoader(train_data,
                                batch_size=config.batch_size,
                                shuffle=shuffle,
                                num_workers=0,
                                collate_fn=mol_collator)

        # Run training
        best_score = np.inf if config.minimize_score else -np.inf
        best_epoch, n_iter = 0, 0
        min_val_loss = np.inf
        for epoch in range(config.epochs):
            s_time = time.time()
            n_iter, train_loss = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                n_iter=n_iter,
                shared_dict=shared_dict
            )
            t_time = time.time() - s_time
            s_time = time.time()
            val_scores, val_loss = evaluate(
                model=model,
                data=val_data,
                loss_func=loss_func,
                num_tasks=config.num_tasks,
                metric_func=metric_func,
                batch_size=config.batch_size,
                dataset_type=config.dataset_type,
                scaler=scaler,
                shared_dict=shared_dict,
                args=config
            )
            v_time = time.time() - s_time
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            if config.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(config.task_names, val_scores):
                    logger.info(f'Validation {task_name} {config.metric} = {val_score:.6f}')
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.6f}'.format(train_loss),
                  'loss_val: {:.6f}'.format(val_loss),
                  f'{config.metric}_val: {avg_val_score:.4f}',
                  # 'auc_val: {:.4f}'.format(avg_val_score),
                  'cur_lr: {:.5f}'.format(scheduler.get_lr()[-1]),
                  't_time: {:.4f}s'.format(t_time),
                  'v_time: {:.4f}s'.format(v_time))

            # Save model11 checkpoint if improved validation score
            if config.select_by_loss:
                if val_loss < min_val_loss:
                    min_val_loss, best_epoch = val_loss, epoch
                    save_checkpoint(os.path.join(save_dir, 'model11.pt'), model, scaler, features_scaler, config)
            else:
                if config.minimize_score and avg_val_score < best_score or \
                        not config.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model11.pt'), model, scaler, features_scaler, config)

            if epoch - best_epoch > config.early_stop_epoch:
                break

        # Evaluate on test set using model11 with best validation score
        if config.select_by_loss:
            logger.info(f'Model {model_idx} best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
        else:
            logger.info(f'Model {model_idx} best validation {config.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model11.pt'), cuda=config.cuda)

        test_preds, _ = predict(
            model=model,
            data=test_data,
            loss_func=loss_func,
            batch_size=config.batch_size,
            shared_dict=shared_dict,
            scaler=scaler,
            config=config
        )

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=config.num_tasks,
            metric_func=metric_func,
            dataset_type=config.dataset_type
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds, dtype=float)

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        logger.info(f'Model {model_idx} test {config.metric} = {avg_test_score:.6f}')

        if config.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(config.task_names, test_scores):
                logger.info(f'Model {model_idx} test {task_name} {config.metric} = {test_score:.6f}')

        # Evaluate ensemble on test set
        avg_test_preds = (sum_test_preds / config.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=config.num_tasks,
            metric_func=metric_func,
            dataset_type=config.dataset_type
        )

        ind = [['preds'] * config.num_tasks + ['targets'] * config.num_tasks, config.task_names * 2]
        ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
        data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
        test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
        test_result.to_csv(os.path.join(config.save_dir, 'test_result.csv'))

        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(ensemble_scores)
        logger.info(f'Ensemble test {config.metric} = {avg_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if config.show_individual_scores:
            for task_name, ensemble_score in zip(config.task_names, ensemble_scores):
                logger.info(f'Ensemble test {task_name} {config.metric} = {ensemble_score:.6f}')

    return ensemble_scores


def load_data(config: GroverConfig):
    """
    load the training data.
    """
    # Get data
    logger.info('Loading data')
    config.task_names = get_task_names(config.data_path)
    data = get_data(path=config.data_path, config=config)
    if data.data[0].features is not None:
        config.features_dim = len(data.data[0].features)
    else:
        config.features_dim = 0
    shared_dict = {}
    config.num_tasks = data.num_tasks()
    config.features_size = data.features_size()
    logger.info(f'Number of tasks = {config.num_tasks}')

    # Split data
    logger.info(f'Splitting data with seed {config.seed}')

    if config.separate_test_path:
        test_data = get_data(
            path=config.separate_test_path,
            config=config,
            features_path=config.separate_test_features_path
        )
    if config.separate_val_path:
        val_data = get_data(
            path=config.separate_val_path,
            config=config,
            features_path=config.separate_val_features_path
        )

    if config.separate_val_path and config.separate_test_path:
        train_data = data
    elif config.separate_val_path:
        train_data, _, test_data = split_data(
            data=data, split_type=config.split_type, sizes=(0.8, 0.2, 0.0), seed=config.seed, config=config
        )
    elif config.separate_test_path:
        train_data, val_data, _ = split_data(
            data=data, split_type=config.split_type, sizes=(0.8, 0.2, 0.0), seed=config.seed, config=config
        )
    else:
        train_data, val_data, test_data = split_data(
            data=data, split_type=config.split_type, sizes=config.split_sizes, seed=config.seed, config=config
        )
    if config.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        logger.info('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            logger.info(
                f'{config.task_names[i]} '
                f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}'
            )

    if config.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    config.train_data_size = len(train_data)
    logger.info(
        f'Total size = {len(data):,} | '
        f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}'
    )

    # Initialize scaler and scale training targets
    # by subtracting mean and dividing standard deviation (regression only)
    if config.dataset_type == 'regression' and config.regression_scaling:
        logger.info('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)
    else:
        scaler = None
    return features_scaler, scaler, shared_dict, test_data, train_data, val_data


def save_datasets(save_dir: str,
                  train_data: MoleculeDataset,
                  valid_data: MoleculeDataset,
                  test_data: MoleculeDataset):
    
    train_features = [dp.features for dp in train_data.data]
    train_smiles = [dp.smiles for dp in train_data.data]
    train_targets = [dp.targets for dp in train_data.data]
    train_dict = {'features': train_features,
                  'smiles': train_smiles,
                  'targets': train_targets}

    valid_features = [dp.features for dp in valid_data.data]
    valid_smiles = [dp.smiles for dp in valid_data.data]
    valid_targets = [dp.targets for dp in valid_data.data]
    valid_dict = {'features': valid_features,
                  'smiles': valid_smiles,
                  'targets': valid_targets}

    test_features = [dp.features for dp in test_data.data]
    test_smiles = [dp.smiles for dp in test_data.data]
    test_targets = [dp.targets for dp in test_data.data]
    test_dict = {'features': test_features,
                 'smiles': test_smiles,
                 'targets': test_targets}

    torch.save(train_dict, os.path.join(save_dir, 'train.pt'))
    torch.save(valid_dict, os.path.join(save_dir, 'valid.pt'))
    torch.save(test_dict, os.path.join(save_dir, 'test.pt'))

    return None
