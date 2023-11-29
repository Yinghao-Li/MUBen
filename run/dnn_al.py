"""
# Author: Yinghao Li
# Modified: November 29th, 2023
# ---------------------------------------
# Description: Run the uncertainty quantification experiments
               with DNN backbone model.
"""

import os
import os.path as osp
import sys
import logging
import numpy as np
from datetime import datetime
from transformers import set_seed

from muben.utils.io import set_logging, set_log_path
from muben.utils.argparser import ArgumentParser
from muben.dnn.dataset import Dataset
from muben.dnn.args import Arguments, Config
from muben.dnn.train import Trainer


logger = logging.getLogger(__name__)


def main(args: Arguments):
    # --- construct and validate configuration ---
    config = Config().from_args(args).get_meta().validate().log()

    # --- prepare dataset ---
    training_dataset = Dataset().prepare(config=config, partition="train").downsample_by(config.init_inst_path)
    valid_dataset = Dataset().prepare(config=config, partition="valid")
    test_dataset = Dataset().prepare(config=config, partition="test")

    # --- initialize trainer ---
    trainer = Trainer(
        config=config,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        result_dir=osp.join(config.result_dir, "al-0"),
    )

    # --- run training and testing ---
    trainer.run()

    # --- active learning ---
    for idx_l in range(config.n_al_loops):
        logger.info(f"Active learning loop {idx_l + 1} / {config.n_al_loops}")
        if config.al_random_sampling:
            candidate_ids = list(
                filter(lambda x: x not in training_dataset.selected_ids, list(range(len(training_dataset.smiles))))
            )
            new_ids = np.random.choice(candidate_ids, size=config.n_al_select, replace=False)
        else:
            _, preds = trainer.test_on_training_data(return_preds=True)

            if config.task_type == "classification":
                if preds.shape[0] > 1:
                    preds = preds.mean(axis=0)

                preds = preds.squeeze()
                diff = np.abs(preds - 0.5)

                if len(preds.shape) > 1:
                    masks = training_dataset.masks
                    diff[~training_dataset.masks.astype(bool)] = 0
                    diff = diff.sum(axis=-1)
                    diff /= masks.sum(axis=-1)

                sorted_preds_ids = np.argsort(diff)
                sorted_preds_ids = np.array(
                    list(filter(lambda x: x not in training_dataset.selected_ids, sorted_preds_ids))
                )
                new_ids = sorted_preds_ids[: config.n_al_select]

            elif config.task_type == "regression":
                _, variances = preds
                if variances.shape[0] > 1:
                    variances = variances.mean(axis=0)
                variances = variances.squeeze()

                if len(variances.shape) > 1:
                    masks = training_dataset.masks
                    variances[~training_dataset.masks.astype(bool)] = 0
                    variances = variances.sum(axis=-1)
                    variances /= masks.sum(axis=-1)

                sorted_preds_ids = np.argsort(variances)
                sorted_preds_ids = np.array(
                    list(filter(lambda x: x not in training_dataset.selected_ids, sorted_preds_ids))
                )
                new_ids = sorted_preds_ids[-config.n_al_select :]

        training_dataset.add_sample_by_ids(new_ids)

        trainer = Trainer(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            result_dir=osp.join(config.result_dir, f"al-{idx_l + 1}"),
        )
        trainer.run()

    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")

    # --- set up arguments ---
    parser = ArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_path", None):
        arguments.log_path = set_log_path(arguments, _time)

    set_logging(log_path=arguments.log_path)
    set_seed(arguments.seed)

    if arguments.deploy:
        try:
            main(args=arguments)
        except Exception as e:
            logger.exception(e)
    else:
        main(args=arguments)
