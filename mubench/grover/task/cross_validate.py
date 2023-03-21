"""
The cross validation function for finetuning.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/train/cross_validate.py
"""
import os
import logging
from typing import Tuple

import numpy as np

from ..util.utils import get_task_names, makedirs
from ..util.config import GroverConfig
from .run_evaluation import run_evaluation
from mubench.grover.train import run_training

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
def cross_validate(config: GroverConfig) -> Tuple[float, float]:
    """
    k-fold cross validation.

    :return: A tuple of mean_score and std_score.
    """

    # Initialize relevant variables
    init_seed = config.seed
    save_dir = config.save_dir
    task_names = get_task_names(config.data_path)

    # Run training with different random seeds for each fold
    all_scores = []
    for fold_num in range(config.num_folds):
        logger.info(f'Fold {fold_num}')
        config.seed = init_seed + fold_num
        config.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(config.save_dir)
        if config.task == "finetune":
            model_scores = run_training(config)
        else:
            model_scores = run_evaluation(config)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    logger.info(f'{config.num_folds}-fold cross validation')

    for fold_num, scores in enumerate(all_scores):
        logger.info(f'Seed {init_seed + fold_num} ==> test {config.metric} = {np.nanmean(scores):.6f}')

        if config.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                logger.info(f'Seed {init_seed + fold_num} ==> test {task_name} {config.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model11 across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    logger.info(f'overall_{config.split_type}_test_{config.metric}={mean_score:.6f}')
    logger.info(f'std={std_score:.6f}')

    if config.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            logger.info(
                f'Overall test {task_name} {config.metric} = '
                f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}'
            )

    return float(mean_score), float(std_score)
