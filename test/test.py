"""
Construct dataset from raw data files
"""
import os
import sys
import logging
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from mubench.utils.io import set_logging, logging_args

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    x: Optional[str] = field(
        default=None,
        metadata={
            # "nargs": "*",
            "help": "The name of the dataset to construct."
        }
    )
    y: Optional[bool] = field(
        default=False,
        metadata={
            # "nargs": "*",
            "help": "The name of the dataset to construct."
        }
    )


def main(args: Arguments):
    logger.info(args.x)


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

    set_logging()
    logging_args(arguments)

    main(args=arguments)
