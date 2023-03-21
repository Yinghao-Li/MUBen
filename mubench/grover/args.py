from typing import Optional
from dataclasses import field

import os
import torch
import pickle
import logging
from argparse import Namespace
from tempfile import TemporaryDirectory

from dataclasses import dataclass

from mubench.grover.util.utils import makedirs
from mubench.grover.data.molfeaturegenerator import get_available_features_generators

from seqlbtoolkit.training.config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class GroverArguments:
    """
    Grover fine-tuning/prediction arguments
    """
    task: Optional[str] = field(
        default=None,
        metadata={'choices': ['finetune', 'eval', 'predict', 'fingerprint'],
                  'help': 'The name of the task to run'}
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to data CSV file.'}
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to csv/npz file where outputs will be saved. '
                          'This argument is only used for `fingerprint` and `predict` tasks.'}
    )
    use_compound_names: Optional[bool] = field(
        default=False,
        metadata={'help': 'Use when test data file contains compound names in addition to SMILES strings'}
    )
    max_data_size: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum number of data points to load'}
    )
    features_only: Optional[bool] = field(
        default=False,
        metadata={'help': 'Use only the additional features in an FFN, no graph network'}
    )
    features_generator: Optional[str] = field(
        default=None,
        metadata={'nargs': '*',
                  'help': 'Method of generating additional features.'}
    )
    features_path: Optional[str] = field(
        default=None,
        metadata={'nargs': '*',
                  'help': 'Path to features to use in FNN (instead of features_generator).'}
    )
    save_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Directory where model11 checkpoints will be saved'}
    )
    save_smiles_splits: Optional[bool] = field(
        default=False,
        metadata={'help': 'Save smiles for each train/val/test splits for prediction convenience later'}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Directory from which to load model11 checkpoints'
                          '(walks directory and ensembles all models that are found)'}
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to model11 checkpoint (.pt file)'}
    )

    # Data splitting.
    dataset_type: Optional[str] = field(
        default='classification',
        metadata={'choices': ['classification', 'regression'],
                  'help': 'Type of dataset, e.g. classification or regression.'
                          'This determines the loss function used during training.'}
    )
    separate_val_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to separate val set, optional'}
    )
    separate_val_features_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to file with features for separate val set'}
    )
    separate_test_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to separate test set, optional'}
    )
    separate_test_features_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to file with features for separate test set'}
    )
    split_type: Optional[str] = field(
        default='random',
        metadata={'choices': ['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'],
                  'help': 'Method of splitting the data into train/val/test'}
    )
    split_sizes: Optional[tuple] = field(
        default=(0.8, 0.1, 0.1),
        metadata={'help': 'Split proportions for train/validation/test sets'}
    )
    num_folds: Optional[int] = field(
        default=1,
        metadata={'help': 'Number of folds when performing cross validation'}
    )
    folds_file: Optional[str] = field(
        default=None,
        metadata={'help': 'Optional file of fold labels'}
    )
    val_fold_index: Optional[int] = field(
        default=None,
        metadata={'help': 'Which fold to use as val for leave-one-out cross val'}
    )
    test_fold_index: Optional[int] = field(
        default=None,
        metadata={'help': 'Which fold to use as test for leave-one-out cross val'}
    )
    crossval_index_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Directory in which to find cross validation index files'}
    )
    crossval_index_file: Optional[str] = field(
        default=None,
        metadata={'help': 'Indices of files to use as train/val/test. Overrides --num_folds and --seed.'}
    )
    seed: Optional[int] = field(
        default=0,
        metadata={'help': 'Random seed to use when splitting data into train/val/test sets.'
                          'When `num_folds` > 1, the first fold uses this seed and all'
                          'subsequent folds add 1 to the seed.'}
    )

    # Metric
    metric: Optional[str] = field(
        default=None,
        metadata={'choices': ['auc', 'prc-auc', 'rmse', 'mae', 'r2', 'accuracy', 'recall',
                              'sensitivity', 'specificity', 'matthews_corrcoef'],
                  'help': 'Metric to use during evaluation.'
                          'Note: Does NOT affect loss function used during training'
                          '(loss is determined by the `dataset_type` argument).'
                          'Note: Defaults to "auc" for classification and "rmse" for regression.'}
    )
    show_individual_scores: Optional[bool] = field(
        default=False,
        metadata={'help': 'Show all scores for individual targets, not just average, at the end'}
    )

    # Training arguments
    epochs: Optional[int] = field(
        default=30,
        metadata={'help': 'Number of epochs to task'}
    )
    warmup_epochs: Optional[float] = field(
        default=2.0,
        metadata={'help': 'Number of epochs during which learning rate increases linearly from'
                          'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                          'from max_lr to final_lr.'}
    )
    init_lr: Optional[float] = field(
        default=1e-4,
        metadata={'help': 'Initial learning rate'}
    )
    max_lr: Optional[float] = field(
        default=1e-3,
        metadata={'help': 'Maximum learning rate'}
    )
    final_lr: Optional[float] = field(
        default=1e-4,
        metadata={'help': 'Final learning rate'}
    )
    no_features_scaling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Turn off scaling of features'}
    )
    no_regression_scaling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Disable regression scaling.'}
    )
    early_stop_epoch: Optional[int] = field(
        default=1000,
        metadata={'help': 'If val loss did not drop in this epochs, stop running'}
    )

    # Model arguments
    ensemble_size: Optional[int] = field(
        default=1,
        metadata={'help': 'Number of models for ensemble prediction.'}
    )
    dropout: Optional[float] = field(
        default=0.0,
        metadata={'help': 'Dropout probability'}
    )
    activation: Optional[str] = field(
        default='ReLU',
        metadata={'choices': ['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                  'help': 'Activation function'}
    )
    ffn_hidden_size: Optional[int] = field(
        default=None,
        metadata={'help': 'Hidden dim for higher-capacity FFN (defaults to hidden_size)'}
    )
    ffn_num_layers: Optional[int] = field(
        default=2,
        metadata={'help': 'Number of layers in FFN after MPN encoding'}
    )
    weight_decay: Optional[float] = field(
        default=0.0, metadata={'help': 'weight_decay'}
    )
    select_by_loss: Optional[bool] = field(
        default=False,
        metadata={'help': 'Use validation loss as refence standard to select best model11 to predict'}
    )

    embedding_output_type: Optional[str] = field(
        default="atom",
        metadata={'choices': ["atom", "bond", "both"],
                  'help': "This the model parameters for pretrain model. The current finetuning task only "
                          "use the embeddings from atom branch. "}
    )

    # Self-attentive readout.
    self_attention: Optional[bool] = field(
        default=False,
        metadata={'help': 'Use self attention layer. Otherwise use mean aggregation layer.'}
    )
    attn_hidden: Optional[int] = field(
        default=4,
        metadata={'nargs': '?',
                  'help': 'Self attention layer hidden layer size.'}
    )
    attn_out: Optional[int] = field(
        default=128,
        metadata={'nargs': '?',
                  'help': 'Self attention layer output feature size.'}
    )
    dist_coff: Optional[float] = field(
        default=0.1,
        metadata={'help': 'The dist coefficient for output of two branches.'}
    )
    bond_drop_rate: Optional[float] = field(
        default=0,
        metadata={'help': 'Drop out bond in molecular.'}
    )
    distinct_init: Optional[bool] = field(
        default=False,
        metadata={'help': 'Using distinct weight init for model11 ensemble'}
    )
    fine_tune_coff: Optional[float] = field(
        default=1,
        metadata={'help': 'Enable distinct fine tune learning rate for fc and other layer'}
    )

    # For multi-gpu finetune.
    enbl_multi_gpu: Optional[bool] = field(
        default=False,
        metadata={'dest': 'enbl_multi_gpu',
                  'help': 'enable multi-GPU training'}
    )

    # Common arguments
    no_cache: Optional[bool] = field(
        default=True,
        metadata={'help': 'Turn off caching mol2graph computation'}
    )
    gpu: Optional[int] = field(
        default=0,
        metadata={
            'choices': list(range(torch.cuda.device_count())),
            'help': 'Which GPU to use'}
    )
    no_cuda: Optional[bool] = field(
        default=False,
        metadata={'help': 'Turn off cuda'}
    )
    batch_size: Optional[int] = field(
        default=32,
        metadata={'help': 'Batch size'}
    )


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

        if self.features_only:
            assert self.features_generator or self.features_path

        self.use_input_features = self.features_generator or self.features_path

        if self.features_generator is not None and 'rdkit_2d_normalized' in self.features_generator:
            assert not self.features_scaling

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

        assert self.features_generator is None or \
               set(self.features_generator).issubset(set(get_available_features_generators())), \
            ValueError(f"Argument 'features_generator' should be None or in {get_available_features_generators()}.")

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

        assert self.features_generator is None or \
               set(self.features_generator).issubset(get_available_features_generators()), \
            ValueError(f"Argument 'features_generator' should be None or in {get_available_features_generators()}.")

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
