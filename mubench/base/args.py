import os
import torch
import logging
from typing import Optional
from dataclasses import dataclass, field

from functools import cached_property
from seqlbtoolkit.training.config import BaseConfig
from ..utils.macro import (
    DATASET_NAMES,
    MODEL_NAMES,
    UNCERTAINTY_METHODS,
    FINGERPRINT_FEATURE_TYPES
)

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- wandb parameters ---
    wandb_api_key: Optional[str] = field(
        default=None, metadata={
            'help': 'The API key that indicates your wandb account suppose you want to use a user different from '
                    'whom stored in the environment variables. Can be found here: https://wandb.ai/settings'}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={'help': 'name of the wandb project.'}
    )
    wandb_name: Optional[str] = field(
        default=None, metadata={'help': 'wandb model name.'}
    )
    disable_wandb: Optional[bool] = field(
        default=False, metadata={'help': 'Disable WandB even if relevant arguments are filled.'}
    )

    # --- IO arguments ---
    dataset_name: Optional[str] = field(
        default='', metadata={
            "help": "Dataset Name.",
            "choices": DATASET_NAMES
        }
    )
    dataset_splitting_random_seed: Optional[int] = field(
        default=0, metadata={
            "help": "The random seed used during dataset construction. Leave default (0) if not randomly split."
        }
    )
    data_dir: Optional[str] = field(
        default='', metadata={'help': 'Directory to datasets'}
    )
    result_dir: Optional[str] = field(
        default='./output', metadata={'help': "where to save model outputs."}
    )
    ignore_preprocessed_dataset: Optional[bool] = field(
        default=False, metadata={"help": "Ignore pre-processed datasets and re-generate features if necessary."}
    )
    overwrite_results: Optional[bool] = field(
        default=False, metadata={'help': 'Whether overwrite existing outputs.'}
    )

    # --- Model Arguments ---
    model_name: Optional[str] = field(
        default='DNN', metadata={
            'help': "Name of the model",
            "choices": MODEL_NAMES
        }
    )
    dropout: Optional[float] = field(
        default=0.1, metadata={'help': "Dropout ratio."}
    )
    binary_classification_with_softmax: Optional[bool] = field(
        default=False, metadata={'help': "Use softmax output instead of sigmoid for binary classification."}
    )
    regression_with_variance: Optional[bool] = field(
        default=False, metadata={'help': "Use two regression output heads, one for mean and the other for variance."}
    )

    # --- Uncertainty Arguments ---
    uncertainty_method: Optional[str] = field(
        default='none', metadata={
            "help": "Uncertainty estimation method",
            "choices": UNCERTAINTY_METHODS
        }
    )

    # -- Feature Arguments ---
    feature_type: Optional[str] = field(
        default='none', metadata={
            "help": "Fingerprint generation function",
            "choices": FINGERPRINT_FEATURE_TYPES
        }
    )

    # --- DNN Arguments ---
    n_dnn_hidden_layers: Optional[int] = field(
        default=8, metadata={'help': "The number of DNN hidden layers."}
    )
    d_dnn_hidden: Optional[int] = field(
        default=128, metadata={'help': "The dimensionality of DNN hidden layers."}
    )

    # --- Training Arguments ---
    retrain_model: Optional[bool] = field(
        default=False, metadata={"help": "Train the model from scratch even if there are models saved in result dir"}
    )
    batch_size: Optional[int] = field(
        default=32, metadata={'help': "Batch size."}
    )
    n_epochs: Optional[int] = field(
        default=50, metadata={'help': "How many epochs to train the model."}
    )
    lr: Optional[float] = field(
        default=1e-4, metadata={'help': "Learning Rate."}
    )
    grad_norm: Optional[float] = field(
        default=0, metadata={"help": "Gradient norm. Default is 0 (do not clip gradient)"}
    )
    lr_scheduler_type: Optional[str] = field(
        default='constant', metadata={
            'help': "Learning rate scheduler with warm ups defined in `transformers`, Please refer to "
                    "https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules for details",
            'choices': ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
        }
    )
    warmup_ratio: Optional[float] = field(
        default=0.1, metadata={"help": "Learning rate scheduler warm-up ratio"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    debug: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )

    # --- Evaluation Arguments ---
    valid_epoch_interval: Optional[int] = field(
        default=1, metadata={'help': 'How many training epochs within each validation step. '
                                     'Set to 0 to disable validation.'}
    )
    n_test: Optional[int] = field(
        default=1, metadata={'help': "How many test loops to run in one training process"}
    )

    # --- Device Arguments ---
    no_cuda: Optional[bool] = field(
        default=False, metadata={"help": "Disable CUDA even when it is available."}
    )
    num_workers: Optional[int] = field(
        default=0, metadata={"help": 'The number of threads to process dataset.'}
    )

    def __post_init__(self):
        self.data_dir = os.path.join(self.data_dir, self.dataset_name, f"split-{self.dataset_splitting_random_seed}")
        self.apply_wandb = self.wandb_project and self.wandb_name and not self.disable_wandb

        # if self.model_name == 'DNN' and self.feature_type == 'none':
        #     raise ValueError(f"Must assign a value to `feature_type` while using DNN model. "
        #                      f"Options are {[t for t in FINGERPRINT_FEATURE_TYPES if t != 'none']}")

    # The following three functions are copied from transformers.training_args
    @cached_property
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda or not torch.cuda.is_available():
            device = torch.device("cpu")
            self._n_gpu = 0
        else:
            device = torch.device("cuda")
            self._n_gpu = 1

        return device

    @cached_property
    def device_str(self) -> str:
        if self.no_cuda or not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda"

        return device

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    def n_gpu(self) -> int:
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu


@dataclass
class Config(Arguments, BaseConfig):

    d_feature = None
    classes = None
    task_type = "classification"
    n_tasks = None

    @cached_property
    def n_lbs(self):
        if self.task_type == 'classification':
            if len(self.classes) == 2 and not self.binary_classification_with_softmax:
                return 1
            else:
                return len(self.classes)
        elif self.task_type == 'regression':
            return 2 if self.regression_with_variance else 1
        else:
            ValueError(f"Unrecognized task type: {self.task_type}")

    @cached_property
    def d_feature(self):
        if self.feature_type == 'rdkit':
            return 200
        elif self.feature_type == 'rdkit':
            return 1024
        else:
            return 0
