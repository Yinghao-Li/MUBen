"""
# Author: Yinghao Li
# Modified: April 8th, 2024
# ---------------------------------------
# Description: This module includes base classes for dataset creation and batch processing.
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
        smiles (Union[list[str], None]): Chemical structures represented as strings.
        lbs (Union[np.ndarray, None]): data labels.
        masks (Union[np.ndarray, None]): Data masks.
        _ori_ids (Union[np.ndarray, None]): Original IDs of the datapoints,
            specifically used for randomly split datasets.
        data_instances: Packed instances of data.
    """

    def __init__(self):
        """Initialize the Dataset class."""
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
        """Returns the current data instances, considering whether the full dataset or a selection is being used."""
        if self.use_full_dataset:
            return self.data_instances_all
        return self.data_instances_selected if self.data_instances_selected is not None else self.data_instances_all

    @data_instances.setter
    def data_instances(self, x):
        """Sets the data instances."""
        self.data_instances_all = x

    @property
    def smiles(self) -> list[str]:
        """Returns the chemical structures represented as strings."""
        return self._smiles

    @property
    def lbs(self) -> np.ndarray:
        """Returns the label data, considering whether standardized labels are being used."""
        if self.has_standardized_lbs and self._use_standardized_lbs:
            return self._lbs_standardized
        return self._lbs

    @property
    def ori_ids(self) -> np.ndarray:
        """Returns the original IDs of the data points."""
        return self._ori_ids

    def toggle_standardized_lbs(self, use_standardized_lbs: bool = None):
        """Toggle between using standardized and unstandardized labels.

        Args:
            use_standardized_lbs (bool, optional): Whether to use standardized labels. Defaults to None.

        Returns:
            self (Dataset): The dataset with the standardized labels toggled.
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
        """Returns the data masks, generating masks with ones if not present."""
        return self._masks if self._masks is not None else np.ones_like(self.lbs).astype(int)

    def __len__(self):
        """Returns the length of the dataset."""
        if self.data_instances is not None:
            return len(self.data_instances)
        return len(self._smiles)

    def __getitem__(self, idx):
        """Gets the dataset item at the specified index."""
        return self.data_instances[idx]

    def update_lbs(self, lbs):
        """Updates the dataset labels and the instance list accordingly.

        Args:
            lbs: The new labels.

        Returns:
            self (Dataset): The dataset with the updated labels.
        """
        self._lbs = lbs
        self.data_instances = self.get_instances()
        return self

    def set_standardized_lbs(self, lbs):
        """Sets standardized labels and updates the instance list accordingly.

        Args:
            lbs: The standardized label data.

        Returns:
            self (Dataset): The dataset with the standardized labels set.
        """
        self._lbs_standardized = lbs
        self.has_standardized_lbs = True
        return self

    def prepare(self, config, partition, **kwargs):
        """Prepares the dataset for training and testing.

        Args:
            config: Configuration parameters.
            partition (str): The dataset partition; should be one of 'train', 'valid', 'test'.

        Raises:
            ValueError: If `partition` is not one of 'train', 'valid', 'test'.

        Returns:
            self (Dataset): The prepared dataset.
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
        """Downsamples the dataset to a subset with the specified indices.

        Args:
            file_path (str, optional): Path to the file containing the indices of the selected instances.
            ids (list[int], optional): Indices of the selected instances.

        Raises:
            ValueError: If neither `ids` nor `file_path` is specified.

        Returns:
            self (Dataset): The downsampled dataset.
        """
        assert ids is not None or file_path is not None, ValueError("Either `ids` or `file_path` should be specified!")

        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                ids = json.load(f)

        self.selected_ids = ids
        self.data_instances_selected = [self.data_instances_all[idx] for idx in self.selected_ids]
        return self

    def add_sample_by_ids(self, ids: list[int] = None):
        """Appends a subset of data instances to the selected data instances.

        Args:
            ids (list[int], optional): Indices of the selected instances.

        Raises:
            ValueError: If `ids` is not specified.

        Returns:
            self (Dataset): The dataset with the added data instances.
        """
        assert ids is not None, ValueError("`ids` should be specified!")

        intersection = set(ids).intersection(set(self.selected_ids))
        if intersection:
            logger.warning(f"IDs {ids} already exist in the selected instances.")
            return self

        self.selected_ids = list(set(self.selected_ids).union(set(ids)))
        self.data_instances_selected = [self.data_instances_all[idx] for idx in self.selected_ids]

        return self

    def create_features(self, config):
        """Creates data features. This method should be implemented by subclasses
        to generate data features according to different descriptors or fingerprints.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        logger.warning("Method `create_features` is not implemented! Make sure this is intended.")
        return self

    def get_instances(self):
        """Gets the instances of the dataset. This method should be implemented by subclasses
        to pack data, labels, and masks into data instances.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        logger.warning("Method `get_instances` is not implemented! Make sure this is intended.")
        return self

    def save(self, file_path: str):
        """Saves the entire dataset for future use.

        Args:
            file_path (str): Path to the save file.

        Returns:
            self (Dataset)
        """
        attr_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                attr_dict[attr] = value

        os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)
        torch.save(attr_dict, file_path)

        return self

    def load(self, file_path: str):
        """Loads the entire dataset from disk.

        Args:
            file_path (str): Path to the saved file.

        Returns:
            self (Dataset)
        """
        attr_dict = torch.load(file_path)

        for attr, value in attr_dict.items():
            if attr not in self.__dict__:
                logger.warning(f"Attribute {attr} is not natively defined in dataset!")

            setattr(self, attr, value)

        return self

    def read_csv(self, data_dir: str, partition: str):
        """Reads data from CSV files.

        Args:
            data_dir (str): The directory where data files are stored.
            partition (str): The dataset partition ('train', 'valid', 'test').

        Raises:
            FileNotFoundError: If the specified file does not exist.

        Returns:
            self (Dataset)
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
    """Represents a batch of data instances, where each instance is initialized with attributes provided as keyword arguments.

    Each attribute name acts as a key to its corresponding value, allowing for flexible data handling within a batched context.

    Attributes:
        size (int): The size of the batch. Defaults to 0.
        _tensor_members (dict): A dictionary to keep track of tensor attributes for device transfer operations.
    """

    def __init__(self, **kwargs):
        """Initializes a Batch object with dynamic attributes based on the provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments representing attributes of data instances within the batch.
                      A special keyword 'batch_size' can be used to explicitly set the batch size.
        """
        self.size = 0
        self._tensor_members = dict()
        for k, v in kwargs.items():
            if k == "batch_size":
                self.size = v
            setattr(self, k, v)
            self._register_tensor_members(k, v)

    def _register_tensor_members(self, k, v):
        """Registers tensor attributes for later device transfer operations.

        Args:
            k (str): The name of the attribute.
            v: The value of the attribute, expected to be a tensor or an object with a 'to' method.
        """
        if isinstance(v, torch.Tensor) or callable(getattr(v, "to", None)):
            self._tensor_members[k] = v

    def to(self, device):
        """Moves all tensor attributes to the specified device (cpu, cuda).

        Args:
            device: The target device to move the tensor attributes to.

        Returns:
            self: The batch instance with its tensor attributes moved to the specified device.
        """
        for k, v in self._tensor_members.items():
            setattr(self, k, v.to(device))
        return self

    def __len__(self):
        """Determines the length of the batch.

        Returns:
            int: The number of instances in the batch, determined by the size of the first tensor attribute if present, or the explicitly set batch size.
        """
        return len(tuple(self._tensor_members.values())[0]) if not self.size else self.size


def pack_instances(**kwargs) -> list[dict]:
    """Converts lists of attributes into a list of data instances.

    Each data instance is represented as a dictionary with attribute names as keys and the corresponding data point values as values.

    Args:
        **kwargs: Variable length keyword arguments, where each key is an attribute name and its value is a list of data points.

    Returns:
        List[Dict]: A list of dictionaries, each representing a data instance.
    """

    instance_list = list()
    keys = tuple(kwargs.keys())

    for inst_attrs in zip(*tuple(kwargs.values())):
        inst = dict(zip(keys, inst_attrs))
        instance_list.append(inst)

    return instance_list


def unpack_instances(instance_list: list[dict], attr_names: list[str] = None):
    """Converts a list of dictionaries (data instances) back into lists of attribute values.

    This function is essentially the inverse of `pack_instances`.

    Args:
        instance_list (List[Dict]): A list of data instances, where each instance is a dictionary with attribute names as keys.
        attr_names ([List[str]], optional): A list of attribute names to extract. If not provided, all attributes found in the first instance are used.

    Returns:
        List[List]: A list of lists, where each sublist contains all values for a particular attribute across all instances.
    """
    if not attr_names:
        attr_names = list(instance_list[0].keys())
    attribute_tuple = [[inst[name] for inst in instance_list] for name in attr_names]

    return attribute_tuple
