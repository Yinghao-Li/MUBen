"""
# Author: Yinghao Li
# Modified: March 4th, 2024
# ---------------------------------------
# Description: 

Base classes for arguments and configurations.

This module defines base classes for handling arguments and configurations across the application.
It includes classes for model descriptor arguments, general arguments, and configurations that
encompasses dataset, model, and training settings.
"""

import os
import os.path as osp
import json
import torch
import logging
from dataclasses import dataclass, field, asdict
from functools import cached_property

from muben.utils.macro import MODEL_NAMES, UncertaintyMethods
from muben.utils.io import prettify_json

logger = logging.getLogger(__name__)

__all__ = ["DescriptorArguments", "Arguments", "Config"]


@dataclass
class DescriptorArguments:
    """Model type arguments.

    This class holds the arguments related to the descriptor type of the model. It allows for
    specifying the type of descriptor used in model construction, with options including RDKit,
    Linear, 2D, and 3D descriptors.

    Attributes:
        descriptor_type (str): Descriptor type. Choices are ["RDKit", "Linear", "2D", "3D"].
    """

    descriptor_type: str = field(
        default=None,
        metadata={"help": "Descriptor type", "choices": ["RDKit", "Linear", "2D", "3D"]},
    )


@dataclass
class Arguments:
    """Base class for managing arguments related to model training, evaluation, and data handling.

    Attributes:
        wandb_api_key (str): The API key for Weights & Biases. Default is None.
        wandb_project (str): The project name on Weights & Biases. Default is None.
        wandb_name (str): The name of the model on Weights & Biases. Default is None.
        disable_wandb (bool): Disable integration with Weights & Biases. Default is False.
        dataset_name (str): Name of the dataset. Default is an empty string.
        data_folder (str): Folder containing all datasets. Default is an empty string.
        data_seed (int): Seed used for random data splitting. Default is None.
        result_folder (str): Directory to save model outputs. Default is "./output".
        ignore_preprocessed_dataset (bool): Whether to ignore pre-processed datasets. Default is False.
        disable_dataset_saving (bool): Disable saving of pre-processed datasets. Default is False.
        disable_result_saving (bool): Disable saving of training results and model checkpoints. Default is False.
        overwrite_results (bool): Whether to overwrite existing outputs. Default is False.
        log_path (str): Path for the logging file. Set to `disabled` to disable log saving. Default is None.
        descriptor_type (str): Descriptor type. Choices are ["RDKit", "Linear", "2D", "3D"]. Default is None.
        model_name (str): Name of the model. Default is "DNN". Choices are defined in MODEL_NAMES.
        dropout (float): Dropout ratio. Default is 0.1.
        binary_classification_with_softmax (bool): Use softmax for binary classification. Deprecated. Default is False.
        regression_with_variance (bool): Use two output heads for regression (mean and variance). Default is False.
        retrain_model (bool): Train model from scratch regardless of existing saved models. Default is False.
        ignore_uncertainty_output (bool): Ignore saved uncertainty models/results. Load no-uncertainty model if possible. Default is False.
        ignore_no_uncertainty_output (bool): Ignore checkpoints from no-uncertainty training processes. Default is False.
        batch_size (int): Batch size for training. Default is 32.
        batch_size_inference (int): Batch size for inference. Default is None.
        n_epochs (int): Number of training epochs. Default is 50.
        lr (float): Learning rate. Default is 1e-4.
        grad_norm (float): Gradient norm for clipping. 0 means no clipping. Default is 0.
        lr_scheduler_type (str): Type of learning rate scheduler. Default is "constant".
            Choices include ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].
        warmup_ratio (float): Warm-up ratio for learning rate scheduler. Default is 0.1.
        seed (int): Random seed for initialization. Default is 0.
        debug (bool): Enable debugging mode with fewer data. Default is False.
        deploy (bool): Enable deploy mode, avoiding runtime errors on bugs. Default is False.
        time_training (bool): Measure training time per training step. Default is False.
        freeze_backbone (bool): Freeze the backbone model during training. Only update the output layers. Default is False.
        valid_epoch_interval (int): Interval of training epochs between each validation step. Set to 0 to disable validation. Default is 1.
        valid_tolerance (int): Maximum allowed validation steps without performance increase. Default is 20.
        n_test (int): Number of test loops in one training process. Default is 1. For some Bayesian methods, default is 20.
        test_on_training_data (bool): Include test results on training data. Default is False.
        uncertainty_method (str): Method for uncertainty estimation. Default is UncertaintyMethods.none. Choices are defined in UncertaintyMethods.
        n_ensembles (int): Number of ensemble models in deep ensembles method. Default is 5.
        swa_lr_decay (float): Learning rate decay coefficient during SWA training. Default is 0.5.
        n_swa_epochs (int): Number of SWA training epochs. Default is 20.
        k_swa_checkpoints (int): Number of SWA checkpoints for Gaussian covariance matrix. Should not exceed `n_swa_epochs`. Default is 20.
        ts_lr (float): Learning rate for training temperature scaling parameters. Default is 0.01.
        n_ts_epochs (int): Number of Temperature Scaling training epochs. Default is 20.
        apply_temperature_scaling_after_focal_loss (bool): Apply temperature scaling after training with focal loss. Default is False.
        bbp_prior_sigma (float): Sigma value for Bayesian Backpropagation prior. Default is 0.1.
        apply_preconditioned_sgld (bool): Apply pre-conditioned Stochastic Gradient Langevin Dynamics instead of vanilla. Default is False.
        sgld_prior_sigma (float): Variance of the SGLD Gaussian prior. Default is 0.1.
        n_langevin_samples (int): Number of model checkpoints sampled from Langevin Dynamics. Default is 30.
        sgld_sampling_interval (int): Number of epochs per SGLD sampling operation. Default is 2.
        evidential_reg_loss_weight (float): Weight of evidential loss. Default is 1.
        evidential_clx_loss_annealing_epochs (int): Epochs before evidential loss weight increases to 1. Default is 10.
        no_cuda (bool): Disable CUDA even when available. Default is False.
        no_mps (bool): Disable Metal Performance Shaders (MPS) even when available. Default is False.
        num_workers (int): Number of threads for processing the dataset. Default is 0.
        num_preprocess_workers (int): Number of threads for preprocessing the dataset. Default is 8.
        pin_memory (bool): Pin memory for data loader for faster data transfer to CUDA devices. Default is False.
        n_feature_generating_threads (int): Number of threads for generating features. Default is 8.
        enable_active_learning (bool): Enable active learning. Default is False.
        n_init_instances (int): Number of initial instances for active learning. Default is 100.
        n_al_select (int): Number of instances to select in each active learning epoch. Default is 50.
        n_al_loops (int): Number of active learning loops. Default is 5.
        al_random_sampling (bool): Select instances randomly in active learning. Default is False.

    Note:
        This class contains many attributes. Each attribute controls a specific aspect of the training or evaluation process,
        including but not limited to data handling, model selection, training configurations, and evaluation metrics.
    """

    # --- wandb parameters ---
    wandb_api_key: str = field(
        default=None,
        metadata={
            "help": "The API key that indicates your wandb account suppose you want to use a user different from "
            "whom stored in the environment variables. Can be found here: https://wandb.ai/settings"
        },
    )
    wandb_project: str = field(default=None, metadata={"help": "name of the wandb project."})
    wandb_name: str = field(default=None, metadata={"help": "wandb model name."})
    disable_wandb: bool = field(
        default=False,
        metadata={"help": "Disable WandB even if relevant arguments are filled."},
    )

    # --- IO arguments ---
    dataset_name: str = field(default="", metadata={"help": "Dataset Name."})
    data_folder: str = field(default="", metadata={"help": "The folder containing all datasets."})
    data_seed: int = field(
        default=None,
        metadata={"help": "Seed used while constructing the random split dataset"},
    )
    result_folder: str = field(default="./output", metadata={"help": "where to save model outputs."})
    ignore_preprocessed_dataset: bool = field(
        default=False,
        metadata={"help": "Ignore pre-processed datasets and re-generate features if necessary."},
    )
    disable_dataset_saving: bool = field(default=False, metadata={"help": "Do not save pre-processed dataset."})
    disable_result_saving: bool = field(
        default=False,
        metadata={"help": "Do not save training results and trained model checkpoints."},
    )
    overwrite_results: bool = field(default=False, metadata={"help": "Whether overwrite existing outputs."})
    log_path: str = field(
        default=None,
        metadata={"help": "Path to the logging file. Set to `disabled` to disable log saving."},
    )
    descriptor_type: str = field(
        default=None,
        metadata={"help": "Descriptor type", "choices": ["RDKit", "Linear", "2D", "3D"]},
    )

    # --- Model Arguments ---
    model_name: str = field(
        default="DNN",
        metadata={"help": "Name of the model", "choices": MODEL_NAMES},
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout ratio."})
    binary_classification_with_softmax: bool = field(
        default=False,
        metadata={
            "help": "Use softmax output instead of sigmoid for binary classification. "
            "Notice that this argument is now deprecated"
        },
    )
    regression_with_variance: bool = field(
        default=False,
        metadata={"help": "Use two regression output heads, one for mean and the other for variance."},
    )

    # --- Training Arguments ---
    retrain_model: bool = field(
        default=False,
        metadata={"help": "Train the model from scratch even if there are models saved in result dir"},
    )
    ignore_uncertainty_output: bool = field(
        default=False,
        metadata={
            "help": "Ignore the saved uncertainty estimation models and results. "
            "Load model from the no-uncertainty output if possible."
        },
    )
    ignore_no_uncertainty_output: bool = field(
        default=False,
        metadata={"help": "Ignore the model checkpoints from no-uncertainty training processes."},
    )
    batch_size: int = field(default=32, metadata={"help": "Batch size."})
    batch_size_inference: int = field(default=None, metadata={"help": "Inference batch size."})
    n_epochs: int = field(default=50, metadata={"help": "How many epochs to train the model."})
    lr: float = field(default=1e-4, metadata={"help": "Learning Rate."})
    grad_norm: float = field(
        default=0,
        metadata={"help": "Gradient norm. Default is 0 (do not clip gradient)"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate scheduler with warm ups defined in `transformers`, Please refer to "
            "https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules for details",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    warmup_ratio: float = field(default=0.1, metadata={"help": "Learning rate scheduler warm-up ratio"})
    seed: int = field(
        default=0,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Debugging mode with fewer training data"},
    )
    deploy: bool = field(
        default=False,
        metadata={"help": "Deploy mode that does not throw run-time errors when bugs are encountered"},
    )
    time_training: bool = field(
        default=False,
        metadata={"help": "Measure training time in terms of training step."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={
            "help": "Whether freeze the backbone model during training. "
            "If set to True, only the output layers will be updated."
        },
    )
    disable_checkpoint_loading: bool = field(
        default=False,
        metadata={"help": "Disable loading pre-trained model from checkpoint."},
    )

    # --- Evaluation Arguments ---
    valid_epoch_interval: int = field(
        default=1,
        metadata={"help": "How many training epochs within each validation step. " "Set to 0 to disable validation."},
    )
    valid_tolerance: int = field(
        default=20,
        metadata={"help": "Maximum validation steps allowed for non-increasing model performance."},
    )
    n_test: int = field(
        default=1,
        metadata={
            "help": "How many test loops to run in one training process. "
            "The default value for some Bayesian methods such as MC Dropout is 20."
        },
    )
    test_on_training_data: bool = field(
        default=False,
        metadata={"help": "Whether include test results on training data."},
    )

    # --- Uncertainty Arguments ---
    uncertainty_method: str = field(
        default=UncertaintyMethods.none,
        metadata={
            "help": "Uncertainty estimation method",
            "choices": UncertaintyMethods.options(),
        },
    )

    # --- Ensemble Arguments ---
    n_ensembles: int = field(
        default=5,
        metadata={"help": "The number of ensemble models in the deep ensembles method."},
    )

    # --- SWAG Arguments ---
    swa_lr_decay: float = field(
        default=0.5,
        metadata={"help": "The learning rate decay coefficient during SWA training."},
    )
    n_swa_epochs: int = field(default=20, metadata={"help": "The number of SWA training epochs."})
    k_swa_checkpoints: int = field(
        default=20,
        metadata={
            "help": "The number of SWA checkpoints for Gaussian covariance matrix. "
            "This number should not exceed `n_swa_epochs`."
        },
    )

    # --- Temperature Scaling Arguments ---
    ts_lr: float = field(
        default=0.01,
        metadata={"help": "The learning rate to train temperature scaling parameters."},
    )
    n_ts_epochs: int = field(
        default=20,
        metadata={"help": "The number of Temperature Scaling training epochs."},
    )

    # --- Focal Loss Arguments ---
    apply_temperature_scaling_after_focal_loss: bool = field(
        default=False,
        metadata={"help": "Whether to apply temperature scaling after training model with focal loss."},
    )

    # --- BBP Arguments ---
    bbp_prior_sigma: float = field(default=0.1, metadata={"help": "Sigma value for BBP prior."})

    # --- SGLD Arguments ---
    apply_preconditioned_sgld: bool = field(
        default=False,
        metadata={"help": "Whether to apply pre-conditioned SGLD instead of the vanilla one."},
    )
    sgld_prior_sigma: float = field(default=0.1, metadata={"help": "Variance of the SGLD Gaussian prior."})
    n_langevin_samples: int = field(
        default=30,
        metadata={"help": "The number of model checkpoints sampled from the Langevin Dynamics."},
    )
    sgld_sampling_interval: int = field(
        default=2,
        metadata={"help": "The number of epochs per sampling operation."},
    )

    # --- Evidential Networks Arguments ---
    evidential_reg_loss_weight: float = field(default=1, metadata={"help": "The weight of evidential loss."})
    evidential_clx_loss_annealing_epochs: int = field(
        default=10,
        metadata={"help": "How many epochs before evidential loss weight increase to 1."},
    )

    # --- Device Arguments ---
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Disable CUDA even when it is available."},
    )
    no_mps: bool = field(
        default=False,
        metadata={"help": "Disable MPS even when it is available."},
    )
    num_workers: int = field(
        default=0,
        metadata={"help": "The number of threads to process the dataset."},
    )
    num_preprocess_workers: int = field(
        default=8,
        metadata={"help": "The number of threads to process the dataset."},
    )
    pin_memory: bool = field(default=False, metadata={"help": "Pin memory for data loader."})
    n_feature_generating_threads: int = field(default=8, metadata={"help": "Number of feature generation threads"})

    # --- Active Learning Arguments ---
    enable_active_learning: bool = field(default=False, metadata={"help": "Whether to enable active learning."})
    n_init_instances: int = field(default=100, metadata={"help": "Number of initial instances."})
    n_al_select: int = field(default=50, metadata={"help": "Number of instances to select in each epoch."})
    n_al_loops: int = field(default=5, metadata={"help": "Number of active learning loops."})
    al_random_sampling: bool = field(
        default=False, metadata={"help": "Whether to randomly select instances in active learning."}
    )

    def __post_init__(self):
        """Post-initialization to set up derived attributes and paths based on the provided arguments."""
        if self.model_name != "DNN":
            self.feature_type = "none"
            model_name_and_feature = self.model_name
        else:
            model_name_and_feature = f"{self.model_name}-{self.feature_type}"

        # update data and result dir
        if self.al_random_sampling:
            self.result_folder = f"{self.result_folder}-random"

        self.data_dir = osp.join(self.data_folder, self.dataset_name)
        self.result_dir = osp.join(
            self.result_folder,
            self.dataset_name,
            model_name_and_feature,
            self.uncertainty_method,
            f"seed-{self.seed}",
        )
        if self.data_seed is not None:
            self.data_dir = osp.join(self.data_dir, f"seed-{self.data_seed}")
            self.result_dir = osp.join(
                self.result_folder,
                self.dataset_name,
                f"data-seed-{self.data_seed}",
                model_name_and_feature,
                self.uncertainty_method,
                f"seed-{self.seed}",
            )
        if self.enable_active_learning:
            self.init_inst_path = osp.join(self.data_dir, f"al-{self.n_init_instances}.json")

        # wandb arguments
        self.apply_wandb = not self.disable_wandb and (self.wandb_api_key or os.getenv("WANDB_API_KEY"))
        if not self.wandb_name:
            self.wandb_name = (
                f"{self.model_name}{'' if self.feature_type == 'none' else f'-{self.feature_type}'}"
                f"-{self.uncertainty_method}"
            )
        if not self.wandb_project:
            self.wandb_project = f"MUBen-{self.dataset_name}"

    @cached_property
    def device(self) -> str:
        """Determine the device to use for training based on the current configuration and system availability.

        Returns:
            str: The device identifier (e.g., "cuda", "cpu", or "mps").
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
    """Extended configuration class inheriting from Arguments to include dataset-specific arguments.

    Inherits:
        Arguments: Inherits all attributes from Arguments for comprehensive configuration management.

    Attributes:
        classes (List[str]): All possible classification classes. Default is None.
        task_type (str): Type of task, e.g., "classification" or "regression". Default is "classification".
        n_tasks (int): Number of tasks (sets of labels to predict). Default is None.
        eval_metric (str): Metric for evaluating validation and test performance. Default is None.
        random_split (bool): Whether the dataset is split randomly. Default is False.

    Note:
        The attributes defined in Config are meant to be overridden by dataset-specific metadata when used.
    """

    # --- Dataset Arguments ---
    # The dataset attributes are to be overwritten by the dataset meta file when used
    classes = None  # all possible classification classes
    task_type = "classification"  # classification or regression
    n_tasks = None  # how many tasks (sets of labels to predict)
    eval_metric = None  # which metric for evaluating valid and test performance *during training*
    random_split = False  # whether the dataset is split randomly; False indicates scaffold split

    # --- Properties and Functions ---
    @cached_property
    def n_lbs(self):
        """Determines the number of labels to predict based on the task type and uncertainty method.

        Returns:
            int: The number of labels.
        """
        if self.task_type == "classification":
            if len(self.classes) == 2 and not self.uncertainty_method == UncertaintyMethods.evidential:
                return 1
            else:
                return len(self.classes)
        elif self.task_type == "regression":
            if self.uncertainty_method == UncertaintyMethods.evidential:
                return 4
            elif self.regression_with_variance:
                return 2
            else:
                return 1
        else:
            ValueError(f"Unrecognized task type: {self.task_type}")

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        else:
            raise ValueError("`Config` can only be subscribed by str!")

    def get_meta(
        self,
        meta_dir: str = None,
        meta_file_name: str = "meta.json",
    ):
        """Load meta file and update class attributes accordingly.

        Args:
            meta_dir (str): Directory containing the meta file. If not specified, uses `data_dir` attribute.
            meta_file_name (str): Name of the meta file to load. Default is "meta.json".

        Returns:
            Config: The instance itself after updating attributes based on the meta file.
        """
        if meta_dir is not None:
            meta_dir = meta_dir
        elif "data_dir" in dir(self):
            meta_dir = getattr(self, "data_dir")
        else:
            raise ValueError(
                "To automatically load meta file, please either specify "
                "the `meta_dir` argument or define a `data_dir` class attribute."
            )

        meta_dir = osp.join(meta_dir, meta_file_name)
        with open(meta_dir, "r", encoding="utf-8") as f:
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
        """Initialize configuration from an Arguments instance.

        This method updates the current configuration based on the values provided in an instance of the Arguments class or any subclass thereof.
        It's useful for transferring settings from command-line arguments or other configurations directly into this Config instance.

        Args:
            args: An instance of Arguments or a subclass containing configuration settings to be applied.

        Returns:
            Config: The instance itself, updated with the settings from `args`.

        Note:
            This method iterates through all attributes of `args` and attempts to set corresponding attributes in the Config instance.
            Attributes not present in Config will be ignored.
        """
        arg_elements = {
            attr: getattr(args, attr)
            for attr in dir(args)
            if not callable(getattr(args, attr)) and not attr.startswith("__") and not attr.startswith("_")
        }
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def validate(self):
        """Validate the configuration.

        Checks for argument conflicts and resolves them if possible, issuing warnings for any discrepancies found.
        Ensures that the model name, feature type, and uncertainty methods are compatible with the specified task type.

        Raises:
            AssertionError: If an incompatible configuration is detected that cannot be automatically resolved.
        """

        assert not (self.model_name == "DNN" and self.feature_type == "none"), "`feature_type` is required for DNN!"

        # Check whether the task type and uncertainty estimation method are compatible
        assert self.uncertainty_method in UncertaintyMethods.options(self.task_type), (
            f"Uncertainty estimation method {self.uncertainty_method} is not compatible with task type "
            f"{self.task_type}!"
        )

        if self.debug and self.deploy:
            logger.warning("`DEBUG` mode is not allowed when the program is in `DEPLOY`! Setting debug=False.")
            self.debug = False

        if (
            self.uncertainty_method
            in [
                UncertaintyMethods.none,
                UncertaintyMethods.ensembles,
                UncertaintyMethods.focal,
                UncertaintyMethods.temperature,
                UncertaintyMethods.evidential,
            ]
            and self.n_test > 1
        ):
            logger.warning(
                f"The specified uncertainty estimation method {self.uncertainty_method} requires "
                f"only single test run! Setting `n_test` to 1."
            )
            self.n_test = 1

        if (
            self.uncertainty_method
            in [
                UncertaintyMethods.mc_dropout,
                UncertaintyMethods.swag,
                UncertaintyMethods.bbp,
            ]
            and self.n_test == 1
        ):
            logger.warning(
                f"The specified uncertainty estimation method {self.uncertainty_method} requires "
                f"multiple test runs! Setting `n_test` to the default value 30."
            )
            self.n_test = 30

        if self.uncertainty_method == UncertaintyMethods.sgld:
            self.n_test = self.n_langevin_samples

        assert not (
            self.uncertainty_method in [UncertaintyMethods.temperature, UncertaintyMethods.focal]
            and self.task_type == "regression"
        ), f"{self.uncertainty_method} is not compatible with regression tasks!"
        # temporary for evidential networks
        assert not (
            self.uncertainty_method in [UncertaintyMethods.iso] and self.task_type == "classification"
        ), f"{self.uncertainty_method} is not compatible with classification tasks!"

        if self.uncertainty_method in [
            UncertaintyMethods.focal,
            UncertaintyMethods.bbp,
            UncertaintyMethods.sgld,
            UncertaintyMethods.evidential,
        ]:
            self.ignore_no_uncertainty_output = True

        if self.k_swa_checkpoints > self.n_swa_epochs:
            logger.warning(
                "The number of SWA checkpoints should not exceeds that of SWA training epochs! "
                "Setting `k_swa_checkpoints` = `n_swa_epochs`."
            )
            self.k_swa_checkpoints = self.n_swa_epochs

        if self.uncertainty_method == UncertaintyMethods.sgld and not self.lr_scheduler_type == "constant":
            logger.warning("SGLD currently only works with constant lr scheduler. The argument will be modified")
            self.lr_scheduler_type = "constant"

        if self.uncertainty_method == UncertaintyMethods.evidential:
            self.regression_with_variance = False

        return self

    def log(self):
        """Log the current configuration settings.

        Outputs the configuration settings to the logging system, formatted for easy reading.
        """
        elements = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not (attr.startswith("__") or attr.startswith("_"))
        }
        logger.info(f"Configurations:\n{prettify_json(json.dumps(elements, indent=2), collapse_level=2)}")

        return self

    def save(self, file_dir: str, file_name: str = "config"):
        """Save the current configuration to a JSON file.

        Args:
            file_dir (str): The directory where the configuration file will be saved.
            file_name (str): The name of the file (without the extension) to save the configuration. Defaults to "config".

        Raises:
            FileNotFoundError: If the specified directory does not exist.
            Exception: If there is an issue saving the file.
        """
        if osp.isdir(file_dir):
            file_path = osp.join(file_dir, f"{file_name}.json")
        elif osp.isdir(osp.split(file_dir)[0]):
            file_path = file_dir
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Cannot save config file to {file_path}; " f"encountered Error {e}")
            raise e
        return self

    def load(self, file_dir: str, file_name: str = "config"):
        """Load configuration from a JSON file.

        Args:
            file_dir (str): The directory where the configuration file is located.
            file_name (str): The name of the file (without the extension) from which to load the configuration. Defaults to "config".

        Raises:
            FileNotFoundError: If the specified file does not exist or the directory does not contain the configuration file.
        """
        if osp.isdir(file_dir):
            file_path = osp.join(file_dir, f"{file_name}.json")
            assert osp.isfile(file_path), FileNotFoundError(f"{file_path} does not exist!")
        elif osp.isfile(file_dir):
            file_path = file_dir
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        logger.info(f"Setting {type(self)} parameters from {file_path}.")

        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for attr, value in config.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self
