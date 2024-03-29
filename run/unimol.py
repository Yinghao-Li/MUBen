"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: Run the uncertainty quantification experiments
               with Uni-Mol backbone model.
"""

import os
import sys
import logging
from datetime import datetime
from transformers import set_seed

from muben.utils.io import set_logging, set_log_path
from muben.utils.argparser import ArgumentParser
from muben.dataset import DatasetUniMol, DictionaryUniMol, CollatorUniMol
from muben.model import UniMol
from muben.args import ArgumentsUniMol as ArgumentsUniMol, ConfigUniMol as ConfigUniMol
from muben.train import TrainerUnimol


logger = logging.getLogger(__name__)


def main(args: ArgumentsUniMol):
    # --- construct and validate configuration ---
    config = ConfigUniMol().from_args(args).get_meta().validate().log()

    # --- initialize wandb ---
    dictionary = DictionaryUniMol.load()
    dictionary.add_symbol("[MASK]", is_special=True)

    # --- prepare dataset ---
    training_dataset = DatasetUniMol().prepare(config=config, partition="train", dictionary=dictionary)
    valid_dataset = DatasetUniMol().prepare(config=config, partition="valid", dictionary=dictionary)
    test_dataset = DatasetUniMol().prepare(config=config, partition="test", dictionary=dictionary)

    # --- initialize trainer ---
    trainer = TrainerUnimol(
        config=config,
        model_class=UniMol,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        collate_fn=CollatorUniMol(config, atom_pad_idx=dictionary.pad()),
    ).initialize(config=config, dictionary=dictionary)

    # --- run training and testing ---
    trainer.run()

    return None


if __name__ == "__main__":
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")

    # --- set up arguments ---
    parser = ArgumentParser(ArgumentsUniMol)
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
