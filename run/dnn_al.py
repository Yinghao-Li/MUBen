"""
# Author: Yinghao Li
# Modified: November 18th, 2023
# ---------------------------------------
# Description: Run the uncertainty quantification experiments
               with DNN backbone model.
"""

import os
import sys
import wandb
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
    )

    # --- run training and testing ---
    trainer.run()

    for idx_l in range(config.n_al_loops):
        _, preds = trainer.test_on_training_data(return_preds=True)
        if isinstance(preds, np.ndarray):
            preds = preds.squeeze()

        sorted_preds_ids = np.argsort(np.abs(preds - 0.5))
        sorted_preds_ids = np.array(list(filter(lambda x: x not in training_dataset.selected_ids, sorted_preds_ids)))
        sorted_preds_ids = sorted_preds_ids[: config.n_al_select]

        training_dataset.add_sample_by_ids(sorted_preds_ids)

        trainer = Trainer(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
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
