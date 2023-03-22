import os
import torch
import pickle
import logging
from argparse import Namespace
from tempfile import TemporaryDirectory

from dataclasses import dataclass

from ..util.utils import makedirs

from mubench.grover.args import GroverArguments

from seqlbtoolkit.training.config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class GroverConfig(BaseConfig, GroverArguments):
    """
    Grover model11 & trainer configuration
    """

    cuda = None
    features_scaling = True
    minimize_score = False
    use_input_features = False
    num_lrs = 1
    crossval_index_sets = None
    regression_scaling = True

    def from_train_args(self, args: Namespace) -> "GroverConfig":

        self.from_args(args)

        global TEMP_DIR  # Prevents the temporary directory from being deleted upon function return

        assert self.data_path is not None
        assert self.dataset_type is not None

        if self.save_dir is not None:
            makedirs(self.save_dir)
        else:
            TEMP_DIR = TemporaryDirectory()
            self.save_dir = TEMP_DIR.name

        self.cuda = not self.no_cuda and torch.cuda.is_available()
        del self.no_cuda

        self.features_scaling = not self.no_features_scaling
        del self.no_features_scaling

        self.regression_scaling = not self.no_regression_scaling
        del self.no_regression_scaling

        if self.metric is None:
            if self.dataset_type == 'classification':
                self.metric = 'auc'
            else:
                self.metric = 'rmse'

        if not ((self.dataset_type == 'classification' and self.metric in ['auc', 'prc-auc', 'accuracy']) or
                (self.dataset_type == 'regression' and self.metric in ['rmse', 'mae', 'r2'])):
            raise ValueError(f'Metric "{self.metric}" invalid for dataset type "{self.dataset_type}".')

        self.minimize_score = self.metric in ['rmse', 'mae']

        self._update_checkpoint_args()

        self.num_lrs = 1

        assert (self.split_type == 'predetermined') == (self.folds_file is not None) == (
                self.test_fold_index is not None)
        assert (self.split_type == 'crossval') == (self.crossval_index_dir is not None)
        assert (self.split_type in ['crossval', 'index_predetermined']) == (self.crossval_index_file is not None)
        if self.split_type in ['crossval', 'index_predetermined']:
            with open(self.crossval_index_file, 'rb') as rf:
                self.crossval_index_sets = pickle.load(rf)
            self.num_folds = len(self.crossval_index_sets)
            self.seed = 0

        if self.bond_drop_rate > 0:
            self.no_cache = True

        setattr(self, 'fingerprint', False)

        return self

    def from_predict_args(self, args) -> "GroverConfig":

        self.from_args(args)

        assert self.data_path
        assert self.output_path
        assert self.checkpoint_dir is not None or self.checkpoint_path is not None or self.checkpoint_paths is not None

        self._update_checkpoint_args()

        self.cuda = not self.no_cuda and torch.cuda.is_available()
        del self.no_cuda

        # Create directory for preds path
        makedirs(self.output_path, isfile=True)
        setattr(self, 'fingerprint', False)

        return self

    def from_fingerprint_args(self, args) -> "GroverConfig":

        self.from_args(args)

        assert self.data_path
        assert self.output_path
        assert self.checkpoint_path is not None or self.checkpoint_paths is not None

        self._update_checkpoint_args()
        self.cuda = not self.no_cuda and torch.cuda.is_available()
        del self.no_cuda
        makedirs(self.output_path, isfile=True)
        setattr(self, 'fingerprint', True)

        return self

    def _update_checkpoint_args(self) -> None:
        """
        Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

        :param self: Arguments.
        """
        if not hasattr(self, 'checkpoint_path'):
            self.checkpoint_path = None

        if not hasattr(self, 'checkpoint_dir'):
            self.checkpoint_dir = None

        if self.checkpoint_dir is not None and self.checkpoint_path is not None:
            raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

        if self.checkpoint_dir is None:
            self.checkpoint_paths = [self.checkpoint_path] if self.checkpoint_path is not None else None
            return

        self.checkpoint_paths = []

        for root, _, files in os.walk(self.checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    self.checkpoint_paths.append(os.path.join(root, fname))

        if self.task == "eval":
            assert self.ensemble_size * self.num_folds == len(self.checkpoint_paths)

        self.ensemble_size = len(self.checkpoint_paths)

        if self.ensemble_size == 0:
            raise ValueError(f'Failed to find any model11 checkpoints in directory "{self.checkpoint_dir}"')
