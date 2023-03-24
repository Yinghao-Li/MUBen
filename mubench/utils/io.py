import os
import sys
import re
import json
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def set_logging(log_path: Optional[str] = None):
    """
    setup logging
    Last modified: 07/20/21

    Parameters
    ----------
    log_path: where to save logging file. Leave None to save no log files

    Returns
    -------
    None
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    if log_path:
        log_path = os.path.abspath(log_path)
        if not os.path.isdir(os.path.split(log_path)[0]):
            os.makedirs(os.path.abspath(os.path.normpath(os.path.split(log_path)[0])))
        if os.path.isfile(log_path):
            os.remove(log_path)

        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setLevel(logging.DEBUG)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=0,
            handlers=[
                stream_handler,
                file_handler
            ]
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[stream_handler]
        )

    return None


def logging_args(args):
    """
    Logging model arguments into logs
    Last modified: 08/19/21

    Parameters
    ----------
    args: arguments

    Returns
    -------
    None
    """
    arg_elements = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr))
                    and not attr.startswith("__") and not attr.startswith("_")}
    logger.info(f"Parameters: ({type(args)})")
    for arg_element, value in arg_elements.items():
        logger.info(f"  {arg_element}: {value}")

    return None


def remove_dir(directory: str):
    """
    Remove a directory and its subtree folders/files
    """
    dirpath = Path(directory)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    return None


def init_dir(directory: str, clear_original_content: Optional[bool] = True):
    """
    Create the target directory. If the directory exists, remove all subtree folders/files in it.
    """

    if clear_original_content:
        remove_dir(directory)
    os.makedirs(os.path.normpath(directory), exist_ok=True)
    return None


def save_json(obj, path: str, collapse_level: Optional[int] = None):
    """
    Save objective to a json file.
    Create this function so that we don't need to worry about creating parent folders every time

    Parameters
    ----------
    obj: the objective to save
    path: the path to save
    collapse_level: set to any collapse value to prettify output json accordingly

    Returns
    -------
    None
    """
    file_dir = os.path.dirname(os.path.normpath(path))
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    json_obj = json.dumps(obj, indent=2, ensure_ascii=False)
    if collapse_level:
        json_obj = prettify_json(json_obj, collapse_level=collapse_level)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(json_obj)

    return None


def prettify_json(text, indent=2, collapse_level=4):
    """
    Make json file more readable by collapsing indent levels higher than `collapse_level`.

    Parameters
    ----------
    text: input json text obj
    indent: the indent value of your json text. Notice that this value needs to be larger than 0
    collapse_level: the level from which the program stops adding new lines

    Usage
    -----
    ```
    my_instance = list()  # user-defined serializable data structure
    json_obj = json.dumps(my_instance, indent=2, ensure_ascii=False)
    json_obj = prettify_json(json_text, indent=2, collapse_level=4)
    with open(path_to_file, 'w', encoding='utf=8') as f:
        f.write(json_text)
    ```
    """
    pattern = r"[\r\n]+ {%d,}" % (indent * collapse_level)
    text = re.sub(pattern, ' ', text)
    text = re.sub(r'([\[({])+ +', r'\g<1>', text)
    text = re.sub(r'[\r\n]+ {%d}([])}])' % (indent * (collapse_level - 1)), r'\g<1>', text)
    text = re.sub(r'(\S) +([])}])', r'\g<1>\g<2>', text)
    return text


def convert_arguments_from_argparse(args):
    """
    Convert argparse Namespace to transformers style arguments

    Parameters
    ----------
    args: Namespace

    Returns
    -------
    str
    """
    args_string = ''
    for k, v in args.__dict__.items():
        default_value = f"'{v}'" if isinstance(v, str) else v
        arg_str = f"{k}: Optional[{type(v)}] = field(\n"
        arg_str += f"    default={default_value}, metadata={{'help': ''}}\n"
        arg_str += f")\n\n"
        args_string += arg_str
    return args_string
