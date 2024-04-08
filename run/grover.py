"""
# Author: Yinghao Li
# Modified: April 8th, 2024
# ---------------------------------------
# Description: Run the uncertainty quantification experiments
               with GROVER backbone model.
"""

import os
import sys
import logging
from rdkit import RDLogger
from datetime import datetime

from transformers import set_seed, HfArgumentParser

from muben.utils.io import set_logging, set_log_path
from muben.model import GROVER
from muben.dataset import DatasetGrover, CollatorGrover
from muben.args import ArgumentsGrover as ArgumentsGrover, ConfigGrover as ConfigGrover
from muben.train import TrainerGrover


logger = logging.getLogger(__name__)


def main(args: ArgumentsGrover):
    # --- construct and validate configuration ---
    config = ConfigGrover().from_args(args).get_meta().validate().log()

    # --- prepare dataset ---
    training_dataset = DatasetGrover().prepare(config=config, partition="train")
    valid_dataset = DatasetGrover().prepare(config=config, partition="valid")
    test_dataset = DatasetGrover().prepare(config=config, partition="test")

    # --- initialize trainer ---
    trainer = TrainerGrover(
        config=config,
        model_class=GROVER,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        collate_fn=CollatorGrover(config),
    ).initialize(config=config)

    # --- run training and testing ---
    trainer.run()

    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")

    # --- set up arguments ---
    parser = HfArgumentParser(ArgumentsGrover)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (arguments,) = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (arguments,) = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_path", None):
        arguments.log_path = set_log_path(arguments, _time)

    set_logging(log_path=arguments.log_path)

    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    set_seed(arguments.seed)

    if arguments.deploy:
        try:
            main(args=arguments)
        except Exception as e:
            logger.exception(e)
    else:
        main(args=arguments)
