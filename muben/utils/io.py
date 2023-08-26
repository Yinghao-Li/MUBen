"""
# Author: Yinghao Li
# Modified: August 26th, 2023
# ---------------------------------------
# Description: IO functions
"""


import os
import os.path as op
import regex
import pickle
import json
import shutil
import lmdb
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "set_log_path",
    "set_logging",
    "logging_args",
    "init_dir",
    "save_json",
    "save_results",
    "load_results",
    "load_lmdb",
    "load_unimol_preprocessed",
]


def set_log_path(args, time):
    """
    Setup log path.
    Notice that the log path is specified for muben and may not be
    directly compatible with other projects.
    """
    log_path = op.join(
        "logs",
        args.dataset_name,
        args.model_name
        if args.feature_type == "none"
        else f"{args.model_name}-{args.feature_type}",
        args.uncertainty_method,
        f"{time}.log",
    )
    return log_path


def set_logging(log_path: Optional[str] = None):
    """
    setup logging format

    Parameters
    ----------
    log_path: where to save logging file. Leave None to save no log files

    Returns
    -------
    None
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    if log_path and log_path != "disabled":
        log_path = op.abspath(log_path)
        if not op.isdir(op.split(log_path)[0]):
            os.makedirs(op.abspath(op.normpath(op.split(log_path)[0])))
        if op.isfile(log_path):
            os.remove(log_path)

        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setLevel(logging.DEBUG)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=0,
            handlers=[stream_handler, file_handler],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[stream_handler],
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
    arg_elements = {
        attr: getattr(args, attr)
        for attr in dir(args)
        if not callable(getattr(args, attr))
        and not attr.startswith("__")
        and not attr.startswith("_")
    }
    logger.info(f"Configurations: ({type(args)})")
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
    os.makedirs(op.normpath(directory), exist_ok=True)
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
    file_dir = op.dirname(op.normpath(path))
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    json_obj = json.dumps(obj, indent=2, ensure_ascii=False)
    if collapse_level:
        json_obj = prettify_json(json_obj, collapse_level=collapse_level)

    with open(path, "w", encoding="utf-8") as f:
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
    text = regex.sub(pattern, " ", text)
    text = regex.sub(r"([\[({])+ +", r"\g<1>", text)
    text = regex.sub(
        r"[\r\n]+ {%d}([])}])" % (indent * (collapse_level - 1)),
        r"\g<1>",
        text,
    )
    text = regex.sub(r"(\S) +([])}])", r"\g<1>\g<2>", text)
    return text


def convert_arguments_from_argparse(args):
    """
    Convert argparse Namespace to transformers style arguments parser strings.
    Used to transfer code style.

    Parameters
    ----------
    args: Namespace

    Returns
    -------
    str
    """
    args_string = ""
    for k, v in args.__dict__.items():
        default_value = f"'{v}'" if isinstance(v, str) else v
        arg_str = f"{k}: Optional[{type(v).__name__}] = field(\n"
        arg_str += f"    default={default_value}, metadata={{'help': ''}}\n"
        arg_str += f")\n\n"
        args_string += arg_str
    return args_string


def save_results(path, preds, variances, lbs, masks):
    """
    Save muben prediction results.

    Parameters
    ----------
    path: path to save
    preds: model predictions
    variances: variances for regression tasks
    lbs: ground truth labels
    masks: label masks

    Returns
    -------
    None
    """
    if not path.endswith(".pt"):
        path = f"{path}.pt"

    data_dict = {
        "version": 2,
        "preds": preds,
        "vars": variances,
        "lbs": lbs,
        "masks": masks,
    }

    os.makedirs(op.dirname(op.normpath(path)), exist_ok=True)
    torch.save(data_dict, path)
    return None


def load_results(result_paths: list[str]):
    """
    Load muben prediction results.

    Parameters
    ----------
    result_paths : list[str]
        paths to the result files
    """
    lbs = masks = np.nan
    preds_list = list()
    variances_list = list()

    for test_result_path in result_paths:
        results = torch.load(test_result_path)

        if lbs is not np.nan:
            assert (lbs == results["lbs"]).all()
        else:
            lbs: np.ndarray = results["lbs"]

        if masks is not np.nan:
            assert (masks == results["masks"]).all()
        else:
            masks: np.ndarray = results["masks"]

        if results.get("version", 1) == 1:
            preds_list.append(results["preds"]["preds"])
            try:
                variances_list.append(results["preds"]["vars"])
            except KeyError:
                pass
        elif results.get("version", 1) == 2:
            preds_list.append(results["preds"])
            try:
                variances_list.append(results["vars"])
            except KeyError:
                pass
        else:
            raise ValueError(
                f"Undefined result version: {results.get('version', 1)}"
            )

    # aggregate mean and variance
    preds = np.stack(preds_list).mean(axis=0)
    if (
        variances_list and not (np.asarray(variances_list) == None).any()
    ):  # regression
        # variances = np.mean(np.stack(preds_list) ** 2 + np.stack(variances_list), axis=0) - preds ** 2
        variances = np.stack(variances_list).mean(axis=0)
    else:
        variances = None

    return preds, variances, lbs, masks


def load_lmdb(
    data_path, keys_to_load: list[str] = None, return_dict: bool = False
):
    """
    Load the lmdb-formatted dataset splits used in Uni-Mol

    Parameters
    ----------
    data_path: path to data split (*.lmdb)
    keys_to_load: dict keys to load. The argument `return_dict` is forced to be True if `keys_to_load` is left None
    return_dict: whether to return a dictionary, will be forced to be True if `keys_to_load` is left None

    Returns
    -------
    tuple if return_dict is False else dict of loaded values corresponding to specified keys
    """

    if keys_to_load is None:
        result_dict = None
        return_dict = True
    elif isinstance(keys_to_load, str):
        result_dict = {keys_to_load: list()}
    else:
        result_dict = {k: list() for k in keys_to_load}

    if not os.path.exists(data_path):
        logger.error(f"Path {data_path} does not exist!")
        return -1

    env = lmdb.open(
        data_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))

    for idx in keys:
        datapoint_pickled = txn.get(idx)
        data = pickle.loads(datapoint_pickled)
        if result_dict is None:
            result_dict = {k: list() for k in data.keys()}
        for k in result_dict:
            result_dict[k].append(data[k])

    if return_dict:
        return result_dict
    else:
        return (v for v in result_dict.values())


def load_unimol_preprocessed(data_dir: str):
    """
    Load all lmdb-formatted dataset training & valid& test splits used in Uni-Mol

    Parameters
    ----------
    data_dir: dir to data splits

    Returns
    -------
    tuple if return_dict is False else dict of loaded values corresponding to specified keys
    """

    result_dict = None
    for partition in ("train", "valid", "test"):
        # read data points
        lmdb_path = os.path.join(data_dir, f"{partition}.lmdb")
        results = load_lmdb(lmdb_path)

        if results == -1:
            raise FileNotFoundError(f"Invalid lmdb path: {lmdb_path}")

        if result_dict is None:
            result_dict = results

        else:
            for k in result_dict:
                result_dict[k] += results[k]

    return result_dict
