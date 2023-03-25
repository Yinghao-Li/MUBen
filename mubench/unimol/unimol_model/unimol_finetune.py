# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os
import warnings
from argparse import Namespace
from typing import Any, Callable, Dict, List

import torch

import numpy as np
from mubench.unimol.unimol_model.unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from .data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
)

from .data import TTADataset
from .unicore.data import UnicoreDataset, data_utils, iterators


logger = logging.getLogger(__name__)

task_metainfo = {
    "esol": {
        "mean": -3.0501019503546094,
        "std": 2.096441210089345,
        "target_name": "logSolubility",
    },
    "freesolv": {
        "mean": -3.8030062305295944,
        "std": 3.8478201171088138,
        "target_name": "freesolv",
    },
    "lipo": {"mean": 2.186336, "std": 1.203004, "target_name": "lipo"},
    "qm7dft": {
        "mean": -1544.8360893118609,
        "std": 222.8902092792289,
        "target_name": "u0_atom",
    },
    "qm8dft": {
        "mean": [
            0.22008500524052105,
            0.24892658759891675,
            0.02289283121913152,
            0.043164444107224746,
            0.21669716560818883,
            0.24225989336408812,
            0.020287111373358993,
            0.03312609817084387,
            0.21681478862847584,
            0.24463634931699113,
            0.02345177178004201,
            0.03730141834205415,
        ],
        "std": [
            0.043832862248693226,
            0.03452326954549232,
            0.053401140662012285,
            0.0730556474716259,
            0.04788020599385645,
            0.040309670766319,
            0.05117163534626215,
            0.06030064428723054,
            0.04458294838213221,
            0.03597696243350195,
            0.05786865052149905,
            0.06692733477994665,
        ],
        "target_name": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9dft": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
}


class StatefulContainer(object):

    _state: Dict[str, Any] = dict()
    _factories: Dict[str, Callable[[], Any]] = dict()

    def add_factory(self, name, factory: Callable[[], Any]):
        self._factories[name] = factory

    def merge_state_dict(self, state_dict: Dict[str, Any]):
        self._state.update(state_dict)

    @property
    def state_dict(self) -> Dict[str, Any]:
        return self._state

    def __getattr__(self, name):
        if name not in self._state and name in self._factories:
            self._state[name] = self._factories[name]()

        if name in self._state:
            return self._state[name]

        raise AttributeError(f"Task state has no factory for attribute {name}")


class UnicoreTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Loss and calculating the loss.

    Tasks have limited statefulness. In particular, state that needs to be
    saved to/loaded from checkpoints needs to be stored in the `self.state`
    :class:`StatefulContainer` object. For example::

        self.state.add_factory("dictionary", self.load_dictionary)
        print(self.state.dictionary)  # calls self.load_dictionary()

    This is necessary so that when loading checkpoints, we can properly
    recreate the task state after initializing the task instance.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        pass

    @staticmethod
    def logging_outputs_can_be_summed(loss, is_train) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `reduce_metrics`.
        Setting this to True will improves distributed training speed.
        """
        return loss.logging_outputs_can_be_summed(is_train)

    args: Namespace
    datasets: Dict[str, UnicoreDataset]
    dataset_to_epoch_iter: Dict[UnicoreDataset, Any]
    state: StatefulContainer = None

    def __init__(self, args: Namespace, **kwargs):
        self.args = args
        self.datasets = dict()
        self.dataset_to_epoch_iter = dict()
        self.state = StatefulContainer()


    @classmethod
    def setup_task(cls, args: Namespace, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def has_sharded_data(self, split):
        return os.pathsep in getattr(self.args, "data", "")

    def load_dataset(
            self,
            split: str,
            combine: bool = False,
            **kwargs
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            combine (bool): combines a split segmented into pieces into one dataset
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~unicore.data.UnicoreDataset` corresponding to *split*
        """
        from unicore.data import UnicoreDataset

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], UnicoreDataset):
            raise TypeError("Datasets are expected to be of type UnicoreDataset")
        return self.datasets[split]

    def can_reuse_epoch_itr(self, dataset):
        # We can reuse the epoch iterator across epochs as long as the dataset
        # hasn't disabled it. We default to ``False`` here, although in practice
        # this will be ``True`` for most datasets that inherit from
        # ``UnicoreDataset`` due to the base implementation there.
        return getattr(dataset, "can_reuse_epoch_itr_across_epochs", False)

    def get_batch_iterator(
            self,
            dataset,
            batch_size=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~unicore.data.UnicoreDataset): dataset to batch
            batch_size (int, optional): max number of samples in each
                batch (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `UnicoreTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~unicore.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.info("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]
        else:
            logger.info("get EpochBatchIterator for epoch {}".format(epoch))

        assert isinstance(dataset, UnicoreDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            batch_size=batch_size,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            disable_shuffling=self.disable_shuffling(),
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def build_model(self, args: Namespace):
        """
        Build the :class:`~unicore.models.BaseUnicoreModel` instance for this
        task.

        Returns:
            a :class:`~unicore.models.BaseUnicoreModel` instance
        """
        from unicore import models
        return models.build_model(args, self)

    def build_loss(self, args: Namespace):
        """
        Build the :class:`~unicore.losses.UnicoreLoss` instance for
        this task.

        Args:
            args (Namespace): configration object

        Returns:
            a :class:`~unicore.losses.UnicoreLoss` instance
        """
        from unicore import losses

        return losses.build_loss(args, self)

    def train_step(
            self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def build_dataset_for_inference(
            self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass

    def begin_valid_epoch(self, epoch, model):
        """Hook function called before the start of each validation epoch."""
        pass

    def reduce_metrics(self, logging_outputs, loss, split='train'):
        """Aggregate logging outputs from data parallel training."""
        if not any("bsz" in log for log in logging_outputs):
            warnings.warn(
                "bsz not found in Loss logging outputs, cannot log bsz"
            )
        else:
            bsz = sum(log.get("bsz", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", bsz, priority=190, round=1)

        loss.__class__.reduce_metrics(logging_outputs, split)

    def state_dict(self):
        if self.state is not None:
            return self.state.state_dict
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.state is not None:
            self.state.merge_state_dict(state_dict)

    def disable_shuffling(self) -> bool:
        return False

class UniMolFinetuneTask:
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.task_name in task_metainfo:
            # for regression task, pre-compute mean and std
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        if split == "train":
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")
            sample_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(dataset, sample_dataset)
        else:
            dataset = TTADataset(
                dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )
            dataset = AtomTypeDataset(dataset, dataset)
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
