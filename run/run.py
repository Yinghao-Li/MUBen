"""
# Author: Yinghao Li
# Modified: April 9th, 2024
# ---------------------------------------
# Description: Run the uncertainty quantification experiments
               with various backbone models.
"""

import os
import sys
import logging
from datetime import datetime
from transformers import set_seed, HfArgumentParser

from muben.utils.io import set_logging, set_log_path
from muben.utils.selectors import argument_selector, configure_selector, dataset_selector, model_selector
from muben.args import DescriptorArguments
from muben.train import Trainer


logger = logging.getLogger(__name__)


def main(args):
    # --- construct and validate configuration ---
    config_class = configure_selector(args.descriptor_type)
    config = config_class().from_args(args).get_meta().validate().log()

    # --- prepare dataset ---
    dataset_class, collator_class = dataset_selector(args.descriptor_type)
    training_dataset = dataset_class().prepare(config=config, partition="train")
    valid_dataset = dataset_class().prepare(config=config, partition="valid")
    test_dataset = dataset_class().prepare(config=config, partition="test")

    # --- initialize trainer ---
    trainer = Trainer(
        config=config,
        model_class=model_selector(args.descriptor_type),
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        collate_fn=collator_class(config),
    ).initialize(config=config)

    # --- run training and testing ---
    trainer.run()

    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")

    # --- set up arguments ---
    parser = HfArgumentParser(DescriptorArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (arguments,) = parser.parse_json_file(os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (arguments,) = parser.parse_yaml_file(os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        (arguments, _) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    argument_class = argument_selector(arguments.descriptor_type)
    parser = HfArgumentParser(argument_class)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (arguments,) = parser.parse_json_file(os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (arguments,) = parser.parse_yaml_file(os.path.abspath(sys.argv[1]), allow_extra_keys=True)
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
