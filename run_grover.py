import os
import sys
import logging
from rdkit import RDLogger
from datetime import datetime

from mubench.utils.io import set_logging, logging_args
from mubench.grover.args import GroverArguments, GroverConfig

from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


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
        from mubench.grover.task.cross_validate import cross_validate
        config = GroverConfig().from_train_args(arguments)
        cross_validate(config)

    else:
        raise ValueError(f"Task {arguments.task} is undefined!")
