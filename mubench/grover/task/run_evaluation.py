"""
The evaluation function.
"""
import logging
from typing import List

import numpy as np
import torch
import torch.utils.data.distributed

from ..data.scaler import StandardScaler
from ..util.utils import get_class_sizes, get_data, split_data, get_task_names, get_loss_func, load_checkpoint
from ..util.config import GroverConfig
from ..util.metrics import get_metric_func
from ..util.nn_utils import param_count
from .predict import evaluate_predictions
from .predict import predict

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
def run_evaluation(config: GroverConfig) -> List[float]:
    """
    Trains a model11 and returns test scores on the model11 checkpoint with the highest validation score.

    :param config: Arguments.
    :return: A list of ensemble scores for each task.
    """

    torch.cuda.set_device(0)

    # Get data
    logger.info('Loading data')
    config.task_names = get_task_names(config.data_path)
    data = get_data(path=config.data_path, config=config)
    config.num_tasks = data.num_tasks()
    config.features_size = data.features_size()
    logger.info(f'Number of tasks = {config.num_tasks}')

    # Split data
    logger.info(f'Splitting data with seed {config.seed}')

    train_data, val_data, test_data = split_data(
        data=data, split_type=config.split_type, sizes=(0.8, 0.1, 0.1), seed=config.seed, config=config
    )

    if config.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        logger.info('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            logger.info(f'{config.task_names[i]} '
                        f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if config.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)

    config.train_data_size = len(train_data)

    logger.info(f'Total size = {len(data):,} | '
                f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler  (regression only)
    scaler = None
    if config.dataset_type == 'regression':
        logger.info('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)

    metric_func = get_metric_func(metric=config.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), config.num_tasks))

    # Load/build model11
    cur_model = config.seed
    target_path = []
    for path in config.checkpoint_paths:
        if "fold_%d" % cur_model in path:
            target_path = path
    logger.info(f'Loading model11 {config.seed} from {target_path}')
    model = load_checkpoint(target_path, current_args=config, cuda=config.cuda)
    # Get loss and metric functions
    loss_func = get_loss_func(config, model)

    logger.info(f'Number of parameters = {param_count(model):,}')

    test_preds, _ = predict(
        model=model,
        data=test_data,
        batch_size=config.batch_size,
        loss_func=loss_func,
        shared_dict={},
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
    logger.info(f'Model test {config.metric} = {avg_test_score:.6f}')

    if config.show_individual_scores:
        # Individual test scores
        for task_name, test_score in zip(config.task_names, test_scores):
            logger.info(f'Model test {task_name} {config.metric} = {test_score:.6f}')

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / config.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=config.num_tasks,
        metric_func=metric_func,
        dataset_type=config.dataset_type
    )

    return ensemble_scores
