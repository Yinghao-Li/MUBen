"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: Run the uncertainty quantification experiments
               with TorchMD-NET backbone model.
"""

import os
import sys
import wandb
import logging
from datetime import datetime
from transformers import set_seed

from muben.utils.io import set_logging, set_log_path
from muben.utils.argparser import ArgumentParser
from muben.dataset import Dataset3D, Collator3D
from muben.args.args_3d import Arguments3D, Config3D
from muben.train import Trainer
from muben.model import TorchMDNET


logger = logging.getLogger(__name__)


def main(args: Arguments3D):
    # --- construct and validate configuration ---
    config = Config3D().from_args(args).get_meta().validate().log()

    # --- prepare dataset ---
    training_dataset = Dataset3D().prepare(config=config, partition="train")
    valid_dataset = Dataset3D().prepare(config=config, partition="valid")
    test_dataset = Dataset3D().prepare(config=config, partition="test")

    # --- initialize trainer ---
    trainer = Trainer(
        config=config,
        model_class=TorchMDNET,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        collate_fn=Collator3D(config),
    ).initialize(config=config)

    # --- run training and testing ---
    trainer.run()

    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")

    # --- set up arguments ---
    parser = ArgumentParser(Arguments3D)
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
