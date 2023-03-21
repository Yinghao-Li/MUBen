"""
Run the basic model11 and training process
"""

import os
import sys
import wandb
import logging
from datetime import datetime

from transformers import (
    HfArgumentParser,
    set_seed,
)

from seqlbtoolkit.io import set_logging, logging_args

from mubench.base.dataset import Dataset
from mubench.base.args import Arguments, Config
from mubench.base.train import Trainer


logger = logging.getLogger(__name__)


def main(args: Arguments):
    config = Config().from_args(args).get_meta()

    if args.apply_wandb and args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config.__dict__,
        mode='online' if args.apply_wandb else 'disabled'
    )

    training_dataset = Dataset().prepare(
        config=config,
        partition="train"
    )
    valid_dataset = Dataset().prepare(
        config=config,
        partition="valid"
    )
    test_dataset = Dataset().prepare(
        config=config,
        partition="test"
    )

    trainer = Trainer(
        config=config,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset
    )

    trainer.run()

    return None


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        arguments, = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_dir", None):
        arguments.log_dir = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=arguments.log_dir)
    logging_args(arguments)

    set_seed(arguments.seed)

    main(args=arguments)
