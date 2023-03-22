import os
import sys
import logging
import numpy as np
from rdkit import RDLogger
from datetime import datetime
from typing import Tuple

from transformers import (
    HfArgumentParser,
    set_seed,
)
from mubench.utils.io import set_logging, logging_args
from mubench.grover.args import GroverArguments, Config
from mubench.grover.util.utils import (
    get_task_names,
    makedirs
)
from mubench.grover.grover_train import run_training

logger = logging.getLogger(__name__)


def cross_validate(config: Config) -> Tuple[float, float]:
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
        model_scores = run_training(config)
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


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # setup random seed
    set_seed(seed=42)
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # --- set up arguments ---
    parser = HfArgumentParser(GroverArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        arguments, = parser.parse_args_into_dataclasses()

    # Setup logging
    if not getattr(arguments, "log_path", None):
        arguments.log_path = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_path=arguments.log_path)
    logging_args(arguments)

    # Package descriptastorus damages the logger. It must be imported after the logging is set up.

    if arguments.task in ['finetune', 'eval']:
        configuration = Config().from_train_args(arguments)
        cross_validate(configuration)

    else:
        raise ValueError(f"Task {arguments.task} is undefined!")
