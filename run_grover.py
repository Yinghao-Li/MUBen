import os
import sys
import logging
from rdkit import RDLogger
from datetime import datetime

from mubench.grover.args import GroverArguments, GroverConfig
from seqlbtoolkit.io import set_logging, logging_args

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
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        args, = parser.parse_args_into_dataclasses()

    # Setup logging
    if getattr(args, "log_dir", None) is None:
        args.log_dir = os.path.join('logs', f'{_current_file_name}-{args.task}', f'{_time}.log')
    set_logging(log_dir=args.log_dir)
    logging_args(args)

    # Package descriptastorus damages the logger. It must be imported after the logging is set up.

    if args.task in ['finetune', 'eval']:
        from mubench.grover.task.cross_validate import cross_validate
        config = GroverConfig().from_train_args(args)
        cross_validate(config)

    else:
        raise ValueError(f"Task {args.task} is undefined!")
