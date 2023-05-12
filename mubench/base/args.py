import os
import json
import torch
import logging
from typing import Optional
from dataclasses import dataclass, field, asdict
from functools import cached_property

from ..utils.macro import (
    DATASET_NAMES,
    MODEL_NAMES,
    UncertaintyMethods,
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
        default=None, metadata={
            "help": "The random seed used during dataset construction. Leave default (0) if not randomly split."
        }
    )
    data_folder: Optional[str] = field(
        default='', metadata={'help': 'The folder containing all datasets.'}
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
        default=UncertaintyMethods.none, metadata={
            "help": "Uncertainty estimation method",
            "choices": UncertaintyMethods.options()
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

    # --- Ensemble Arguments ---
    n_ensembles: Optional[int] = field(
        default=5, metadata={"help": "The number of ensemble models in the deep ensembles method."}
    )

    # --- SWAG Arguments ---
    swa_lr_decay: Optional[float] = field(
        default=0.1, metadata={"help": "The learning rate decay coefficient during SWA training."}
    )
    n_swa_epochs: Optional[int] = field(
        default=30, metadata={"help": "The number of SWA training epochs."}
    )
    k_swa_checkpoints: Optional[int] = field(
        default=30, metadata={"help": "The number of SWA checkpoints for Gaussian covariance matrix. "
                                      "This number should not exceed `n_swa_epochs`."}
    )

    # --- Temperature Scaling Arguments ---
    ts_lr: Optional[float] = field(
        default=0.01, metadata={"help": "The learning rate to train temperature scaling parameters."}
    )
    n_ts_epochs: Optional[int] = field(
        default=10, metadata={"help": "The number of Temperature Scaling training epochs."}
    )

    # --- Focal Loss Arguments ---
    apply_temperature_scaling_after_focal_loss: Optional[bool] = field(
        default=False, metadata={"help": "Whether to apply temperature scaling after training model with focal loss."}
    )

    # --- SGLD Arguments ---
    apply_preconditioned_sgld: Optional[bool] = field(
        default=False, metadata={"help": "Whether to apply pre-conditioned SGLD instead of the vanilla one."}
    )
    sgld_prior_sigma: Optional[float] = field(
        default=0.1, metadata={"help": "Variance of the SGLD Gaussian prior."}
    )
    n_langevin_samples: Optional[int] = field(
        default=30, metadata={"help": "The number of model checkpoints sampled from the Langevin Dynamics."}
    )
    sgld_sampling_interval: Optional[int] = field(
        default=2, metadata={"help": "The number of epochs per sampling operation."}
    )

    # --- Evaluation Arguments ---
    valid_epoch_interval: Optional[int] = field(
        default=1, metadata={'help': 'How many training epochs within each validation step. '
                                     'Set to 0 to disable validation.'}
    )
    n_test: Optional[int] = field(
        default=1, metadata={'help': "How many test loops to run in one training process. "
                                     "The default value for some Bayesian methods such as MC Dropout is 20."}
    )

    # --- Device Arguments ---
    no_cuda: Optional[bool] = field(
        default=False, metadata={"help": "Disable CUDA even when it is available."}
    )
    no_mps: Optional[bool] = field(
        default=False, metadata={"help": "Disable MPS even when it is available."}
    )
    num_workers: Optional[int] = field(
        default=0, metadata={"help": 'The number of threads to process the dataset.'}
    )
    n_feature_generating_threads: Optional[int] = field(
        default=8, metadata={'help': "Number of feature generation threads"}
    )

    def __post_init__(self):
        if self.dataset_splitting_random_seed is not None:  # random splitting
            self.data_dir = os.path.join(
                self.data_folder, self.dataset_name, f"split-{self.dataset_splitting_random_seed}"
            )
        else:  # scaffold splitting
            self.data_dir = os.path.join(self.data_folder, self.dataset_name, f"scaffold")
        self.apply_wandb = self.wandb_project and self.wandb_name and not self.disable_wandb

        try:
            bf16_supported = True if torch.cuda.is_bf16_supported() else False
        except AssertionError:  # case where gpu is not accessible
            bf16_supported = True
        self.hf_training = True \
            if not self.model_name == "GROVER" and bf16_supported and not self.device == 'mps' \
            else False

    @property
    def device_str(self) -> str:
        return self.device

    @cached_property
    def device(self) -> str:
        """
        The device used by this process.
        """
        try:
            mps_available = torch.backends.mps.is_available()
        except AttributeError:
            mps_available = False

        if mps_available and not self.no_mps:
            device = "mps"
        elif self.no_cuda or not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda"

        return device


@dataclass
class Config(Arguments):

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

    def get_meta(self,
                 meta_dir: Optional[str] = None,
                 meta_file_name: Optional[str] = 'meta.json'):

        if meta_dir is not None:
            meta_dir = meta_dir
        elif 'data_dir' in dir(self):
            meta_dir = getattr(self, 'data_dir')
        else:
            raise ValueError("To automatically load meta file, please either specify "
                             "the `meta_dir` argument or define a `data_dir` class attribute.")

        meta_dir = os.path.join(meta_dir, meta_file_name)
        with open(meta_dir, 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)

        invalid_keys = list()
        for k, v in meta_dict.items():
            if k in dir(self):
                setattr(self, k, v)
            else:
                invalid_keys.append(k)

        if invalid_keys:
            logger.warning(f"The following attributes in the meta file are not defined in config: {invalid_keys}")

        return self

    def from_args(self, args):
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: BertConfig)
        """
        arg_elements = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr))
                        and not attr.startswith("__") and not attr.startswith("_")}
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def validate(self):
        """
        Check and solve argument conflicts and throws warning if encountered any

        Returns
        -------

        """
        if self.uncertainty_method in [UncertaintyMethods.mc_dropout, UncertaintyMethods.swag, UncertaintyMethods.bbp] \
                and self.n_test == 1:
            logger.warning(f"The specified uncertainty estimation method {self.uncertainty_method} requires "
                           f"multiple test runs! Setting `n_test` to the default value 20.")
            self.n_test = 20

        if self.k_swa_checkpoints > self.n_swa_epochs:
            logger.warning("The number of SWA checkpoints should not exceeds that of SWA training epochs! "
                           "Setting `k_swa_checkpoints` = `n_swa_epochs`.")
            self.k_swa_checkpoints = self.n_swa_epochs

        if self.uncertainty_method == UncertaintyMethods.focal and not self.binary_classification_with_softmax:
            logger.warning("Focal Loss requires model with Softmax output! "
                           "Setting `binary_classification_with_softmax` to True.")
            self.binary_classification_with_softmax = True

        if self.uncertainty_method == UncertaintyMethods.sgld and not self.lr_scheduler_type == "constant":
            logger.warning("SGLD currently only works with constant lr scheduler. The argument will be modified")
            self.lr_scheduler_type = "constant"

        return self

    def log(self):
        elements = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr))
                    and not attr.startswith("__") and not attr.startswith("_")}
        logger.info(f"Configurations: ({type(self)})")
        for arg_element, value in elements.items():
            logger.info(f"  {arg_element}: {value}")

        return self

    def save(self, file_dir: str, file_name: Optional[str] = 'config'):
        """
        Save configuration to file

        Parameters
        ----------
        file_dir: file directory
        file_name: file name (suffix free)

        Returns
        -------
        self
        """
        if os.path.isdir(file_dir):
            file_path = os.path.join(file_dir, f'{file_name}.json')
        elif os.path.isdir(os.path.split(file_dir)[0]):
            file_path = file_dir
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Cannot save config file to {file_path}; "
                             f"encountered Error {e}")
            raise e
        return self

    def load(self, file_dir: str, file_name: Optional[str] = 'config'):
        """
        Load configuration from stored file

        Parameters
        ----------
        file_dir: file directory
        file_name: file name (suffix free)

        Returns
        -------
        self
        """
        if os.path.isdir(file_dir):
            file_path = os.path.join(file_dir, f'{file_name}.json')
            assert os.path.isfile(file_path), FileNotFoundError(f"{file_path} does not exist!")
        elif os.path.isfile(file_dir):
            file_path = file_dir
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        logger.info(f'Setting {type(self)} parameters from {file_path}.')

        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for attr, value in config.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self
