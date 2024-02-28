"""
# Author: Yinghao Li
# Modified: February 28th, 2024
# ---------------------------------------
# Description: Base classes for dataset creation and batch processing.
"""

import json
import os
import regex
import torch
import logging
import pandas as pd
import numpy as np

from ast import literal_eval
from typing import Union, Optional
from functools import cached_property
from torch.utils.data import Dataset as TorchDataset


logger = logging.getLogger(__name__)

__all__ = ["Dataset", "Batch", "pack_instances", "unpack_instances"]


class Dataset(TorchDataset):
    """
    Custom Dataset class to handle data storage, manipulation, and preprocessing operations.

    Attributes:
        _smiles (Union[list[str], None]): Chemical structures represented as strings.
        _lbs (Union[np.ndarray, None]): Label data.
        _masks (Union[np.ndarray, None]): Data masks.
        _ori_ids (Union[np.ndarray, None]): Original IDs, specifically useful for randomly split datasets.
        data_instances: Packed instances of data.
    """

    def __init__(self):
        super().__init__()

        self._smiles: Union[list[str], None] = None
        self._lbs: Union[np.ndarray, None] = None
        self._masks: Union[np.ndarray, None] = None
        self._ori_ids: Union[np.ndarray, None] = None

        self.data_instances_all = None
        self.data_instances_selected = None
        self.selected_ids = None
        self.use_full_dataset = False

        self._lbs_standardized: Union[np.ndarray, None] = None
        self._use_standardized_lbs = False
        self.has_standardized_lbs = False

    @property
    def data_instances(self):
        """
        Be careful with this property and the setter below
        """
        if self.use_full_dataset:
            return self.data_instances_all
        return self.data_instances_selected if self.data_instances_selected is not None else self.data_instances_all

    @data_instances.setter
    def data_instances(self, x):
        """
        Be careful with this property and the setter below
        """
        self.data_instances_all = x

    @property
    def smiles(self) -> list[str]:
        return self._smiles

    @property
    def lbs(self) -> np.ndarray:
        if self.has_standardized_lbs and self._use_standardized_lbs:
            return self._lbs_standardized
        return self._lbs

    def toggle_standardized_lbs(self, use_standardized_lbs: bool = None):
        """
        Toggle between standardized and unstandardized labels

        Parameters
        ----------
        use_standardized_lbs: bool, optional
            whether to use standardized labels

        Returns
        -------
        self
        """
        if use_standardized_lbs is None:
            self._use_standardized_lbs = not self._use_standardized_lbs
            self.data_instances = self.get_instances()
        else:
            unchanged = use_standardized_lbs == self._use_standardized_lbs
            self._use_standardized_lbs = use_standardized_lbs
            if not unchanged:
                self.data_instances = self.get_instances()
        return self

    @cached_property
    def masks(self) -> np.ndarray:
        """
        Return masks, and if not present, generate masks with ones.
        """
        return self._masks if self._masks is not None else np.ones_like(self.lbs).astype(int)

    def __len__(self):
        if self.data_instances is not None:
            return len(self.data_instances)
        return len(self._smiles)

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def update_lbs(self, lbs):
        """
        Update dataset labels and instance list accordingly
        """
        self._lbs = lbs
        self.data_instances = self.get_instances()
        return self

    def set_standardized_lbs(self, lbs):
        """
        Update dataset labels and instance list accordingly
        """
        self._lbs_standardized = lbs
        self.has_standardized_lbs = True
        return self

    def prepare(self, config, partition, **kwargs):
        """
        Prepare dataset for training and test

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test]

        Returns
        -------
        self
        """

        assert partition in ["train", "valid", "test"], ValueError(
            f"Argument `partition` should be one of 'train', 'valid' or 'test'!"
        )

        method_identifier = (
            f"{config.model_name}-{config.feature_type}" if config.feature_type != "none" else config.model_name
        )
        preprocessed_path = os.path.normpath(
            os.path.join(
                config.data_dir,
                "processed",
                method_identifier,
                f"{partition}.pt",
            )
        )
        # Load Pre-processed dataset if exist
        if os.path.exists(preprocessed_path) and not config.ignore_preprocessed_dataset:
            logger.info(f"Loading pre-processed dataset {preprocessed_path}")
            self.load(preprocessed_path)
        # else, load dataset from csv and generate features
        else:
            self.read_csv(config.data_dir, partition)

            logger.info("Creating features")
            self.create_features(config)

            # Always save pre-processed dataset to disk
            if not config.disable_dataset_saving:
                logger.info("Saving pre-processed dataset")
                self.save(preprocessed_path)

        self.data_instances = self.get_instances()
        return self

    def downsample_by(self, file_path: str = None, ids: list[int] = None):
        """
        Down-sample the instances to a subset with the specified indices

        Parameters
        ----------
        file_path: str, optional
            path to the file containing the indices of the selected instances
        ids: list[int], optional
            indices of the selected instances

        Returns
        -------
        self
        """
        assert ids is not None or file_path is not None, ValueError("Either `ids` or `file_path` should be specified!")

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                ids = json.load(f)

        self.selected_ids = ids
        self.data_instances_selected = [self.data_instances_all[idx] for idx in self.selected_ids]
        return self

    def add_sample_by_ids(self, ids: list[int] = None):
        """
        Append a subset of data instances to the data_instances_selected


        Parameters
        ----------
        ids: list[int], optional
            indices of the selected instances

        Returns
        -------
        self
        """
        assert ids is not None, ValueError("`ids` should be specified!")

        intersection = set(ids).intersection(set(self.selected_ids))
        if intersection:
            logger.warning(f"IDs {ids} already exist in the selected instances.")
            return self

        self.selected_ids = list(set(self.selected_ids).union(set(ids)))
        self.data_instances_selected = [self.data_instances_all[idx] for idx in self.selected_ids]

        return self

    # noinspection PyTypeChecker
    def create_features(self, config):
        """
        Create data features

        Returns
        -------
        self
        """
        raise NotImplementedError

    def get_instances(self):
        raise NotImplementedError

    def save(self, file_path: str):
        """
        Save the entire dataset for future usage

        Parameters
        ----------
        file_path: path to the saved file

        Returns
        -------
        self
        """
        attr_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                attr_dict[attr] = value

        os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)
        torch.save(attr_dict, file_path)

        return self

    def load(self, file_path: str):
        """
        Load the entire dataset from disk

        Parameters
        ----------
        file_path: path to the saved file

        Returns
        -------
        self
        """
        attr_dict = torch.load(file_path)

        for attr, value in attr_dict.items():
            if attr not in self.__dict__:
                logger.warning(f"Attribute {attr} is not natively defined in dataset!")

            setattr(self, attr, value)

        return self

    def read_csv(self, data_dir: str, partition: str):
        """
        Read data from csv files
        """
        file_path = os.path.normpath(os.path.join(data_dir, f"{partition}.csv"))
        logger.info(f"Loading dataset {file_path}")

        if not (file_path and os.path.exists(file_path)):
            raise FileNotFoundError(f"File {file_path} does not exist!")

        df = pd.read_csv(file_path)
        self._smiles = df.smiles.tolist()
        self._lbs = np.asarray(df.labels.map(literal_eval).to_list())
        self._masks = np.asarray(df.masks.map(literal_eval).to_list()) if not df.masks.isnull().all() else None
        self._ori_ids = df.ori_ids.to_numpy() if "ori_ids" in df.keys() else None  # for randomly split dataset

        return self


class Batch:
    """
    A batch of data instances, each is initialized with a dict with attribute names as keys
    """

    def __init__(self, **kwargs):
        self.size = 0
        self._tensor_members = dict()
        for k, v in kwargs.items():
            if k == "batch_size":
                self.size = v
            setattr(self, k, v)
            self._register_tensor_members(k, v)

    def _register_tensor_members(self, k, v):
        """
        Register tensor members to the batch
        """
        if isinstance(v, torch.Tensor) or callable(getattr(v, "to", None)):
            self._tensor_members[k] = v

    def to(self, device):
        """
        Move all tensor members to the target device
        """
        for k, v in self._tensor_members.items():
            setattr(self, k, v.to(device))
        return self

    def __len__(self):
        return len(tuple(self._tensor_members.values())[0]) if not self.size else self.size


def pack_instances(**kwargs) -> list[dict]:
    """
    Convert attribute lists to a list of data instances, each is a dict with attribute names as keys
    and one datapoint attribute values as values
    """

    instance_list = list()
    keys = tuple(kwargs.keys())

    for inst_attrs in zip(*tuple(kwargs.values())):
        inst = dict(zip(keys, inst_attrs))
        instance_list.append(inst)

    return instance_list


def unpack_instances(instance_list: list[dict], attr_names: Optional[list[str]] = None):
    """
    Convert a list of dict-type instances to a list of value lists,
    each contains all values within a batch of each attribute

    Parameters
    ----------
    instance_list: a list of attributes
    attr_names: the name of the needed attributes. Notice that this variable should be specified
        for Python versions that does not natively support ordered dict
    """
    if not attr_names:
        attr_names = list(instance_list[0].keys())
    attribute_tuple = [[inst[name] for inst in instance_list] for name in attr_names]

    return attribute_tuple