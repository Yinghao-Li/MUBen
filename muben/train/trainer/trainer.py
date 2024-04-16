"""
# Author: Yinghao Li
# Modified: April 16th, 2024
# ---------------------------------------
# Description:

Base trainer functions to facilitate training, validation, and testing 
of machine learning models. This Trainer class is designed to seamlessly 
integrate with various datasets, loss functions, metrics, and uncertainty 
estimation methods. It provides convenient mechanisms to standardize, 
initialize and manage training states, and is also integrated with logging 
and Weights & Biases (wandb) for experiment tracking.
"""

import copy
import wandb
import logging
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from typing import Optional, Union, Tuple
from scipy.special import expit
from transformers import get_scheduler, set_seed

from .scaler import StandardScaler
from .loss import GaussianNLLLoss
from .metric import (
    calculate_binary_classification_metrics,
    calculate_regression_metrics,
)
from .state import TrainerState
from .timer import Timer

from .container import CheckpointContainer, UpdateCriteria
from muben.uncertainty import (
    SWAModel,
    update_bn,
    TSModel,
    SigmoidFocalLoss,
    SGLDOptimizer,
    PSGLDOptimizer,
    IsotonicCalibration,
    EvidentialRegressionLoss,
    EvidentialClassificationLoss,
)

from muben.utils.macro import EVAL_METRICS, UncertaintyMethods
from muben.utils.io import save_results, init_dir

logger = logging.getLogger(__name__)

__all__ = ["Trainer"]


class Trainer:
    """
    This Trainer class is designed to facilitate the training, validation, and testing of machine learning models.
    It integrates with various datasets, loss functions, metrics, and uncertainty estimation methods, providing
    mechanisms to standardize, initialize, and manage training states. It supports logging and integration with
    Weights & Biases (wandb) for experiment tracking.
    """

    def __init__(
        self,
        config,
        model_class=None,
        training_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        collate_fn=None,
        scalar=None,
        **kwargs,
    ):
        """Initializes the Trainer object.

        Args:
            config (Config): Configuration object containing all necessary parameters for training.
            model_class (optional): The class of the model to be trained.
            training_dataset (Dataset, optional): Dataset for training the model.
            valid_dataset (Dataset, optional): Dataset for validating the model.
            test_dataset (Dataset, optional): Dataset for testing the model.
            collate_fn (Callable, optional): Function to collate data samples into batches.
            scalar (StandardScaler, optional): Scaler for standardizing input data.
            **kwargs: Additional keyword arguments for configuration adjustments.
        """
        # make a deep copy of the config to avoid modifying the original config
        for k, v in kwargs.items():
            setattr(config, k, v)

        self._config = config
        self._model_class = model_class
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn
        self._scaler = scalar
        self._device = getattr(config, "device", "cpu")

        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._loss_fn = None
        self._timer = Timer(device=self._device)

        # --- initialize wandb ---
        if config.apply_wandb and config.wandb_api_key:
            wandb.login(key=config.wandb_api_key)
        elif config.apply_wandb:  # use default login if not specified
            wandb.login()

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.to_dict(),
            mode="online" if config.apply_wandb else "disabled",
        )

        # Validation variables and flags
        self._valid_metric = (
            EVAL_METRICS[config.dataset_name].replace("-", "_") if not config.eval_metric else config.eval_metric
        )
        if config.valid_epoch_interval == 0:
            update_criteria = UpdateCriteria.always
        elif config.task_type == "classification":
            update_criteria = UpdateCriteria.metric_larger
        else:
            update_criteria = UpdateCriteria.metric_smaller
        self._update_criteria = update_criteria
        self._checkpoint_container = CheckpointContainer(self._update_criteria)

        self._model_name = "model_best.ckpt"

        # mutable class attributes
        self._status = TrainerState(
            lr=config.lr,
            lr_scheduler_type=config.lr_scheduler_type,
            n_epochs=config.n_epochs,
            valid_epoch_interval=config.valid_epoch_interval,
            model_name=self._model_name,  # mutable model name for ensemble
            result_dir=config.result_dir,  # mutable result directory for ensemble
            result_dir_no_uncertainty=UncertaintyMethods.none.join(  # substitute uncertainty method to none
                config.result_dir.rsplit(config.uncertainty_method, 1)
            ),
        )

        # uncertainty-specific variables
        self._swa_model = None  # SWAG
        self._ts_model = None  # Temperature Scaling
        self._sgld_optimizer = None  # SGLD
        self._sgld_model_buffer = None  # SGLD

        # flags
        self._model_frozen = False
        self._backbone_frozen = False

        # normalize training dataset labels for regression task
        self.standardize_training_lbs()

        # initialize training modules
        # self.initialize()
        self._init_args = None
        self._init_kwargs = None

    @property
    def model(self):
        """Retrieves the scaled model if available, otherwise returns the base model.

        Returns:
            torch.nn.Module: The model, potentially after applying temperature scaling.
        """
        if self._ts_model:
            return self._ts_model
        return self._model

    @property
    def config(self):
        """Retrieves the configuration of the Trainer.

        Returns:
            Config: The configuration object used by the Trainer.
        """
        return self._config

    @property
    def n_model_parameters(self):
        """Computes the total number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def n_update_steps_per_epoch(self):
        """Calculates the number of update steps required per epoch.

        This is based on the size of the training dataset and the configured batch size.

        Returns:
            int: Number of update steps per epoch.
        """
        return int(np.ceil(len(self._training_dataset) / self.config.batch_size))

    @property
    def backbone_params(self):
        """Retrieves parameters of the model's backbone, excluding the output layer.

        Useful for operations that need to differentiate between backbone and output layer parameters,
        such as freezing the backbone during training.

        Returns:
            list: Parameters of the model's backbone.
        """
        output_param_ids = [id(x[1]) for x in self._model.named_parameters() if "output_layer" in x[0]]
        backbone_params = list(
            filter(
                lambda p: id(p) not in output_param_ids,
                self._model.parameters(),
            )
        )
        return backbone_params

    def initialize(self, *args, **kwargs):
        """Initializes the trainer's status and its key components including the model,
        optimizer, learning rate scheduler, and loss function.

        This method sets up the training environment by initializing the model, optimizer,
        learning rate scheduler, and the loss function based on the provided configuration.
        It also prepares the trainer for logging and checkpointing mechanisms.

        Args:
            *args: Variable length argument list for model initialization.
            **kwargs: Arbitrary keyword arguments for model initialization.

        Returns:
            Trainer: The initialized Trainer instance ready for training.
        """
        self.initialize_model(*args, **kwargs)
        self.initialize_optimizer(*args, **kwargs)
        self.initialize_scheduler()
        self.initialize_loss()
        self._timer.init()
        self._status.init()
        self._checkpoint_container = CheckpointContainer(self._update_criteria)

        if self.config.freeze_backbone:
            self.freeze_backbone()

        logger.info(f"Trainer initialized. The model contains {self.n_model_parameters} parameters")

        self._init_args = args
        self._init_kwargs = kwargs

        return self

    def re_initialize(self):
        """Reinitializes the trainer's status and key components. Designed for Deep Ensembles.

        Returns:
            Trainer: The reinitialized Trainer instance.
        """
        return self.initialize(*self._init_args, **self._init_kwargs)

    def initialize_model(self, *args, **kwargs):
        """Abstract method to initialize the model.

        This method should be implemented in subclasses of Trainer, providing the specific
        logic to initialize the model that will be used for training.

        Returns:
            Trainer: The Trainer instance with the model initialized.
        """
        self._model = self._model_class(*args, **kwargs)
        return self

    def initialize_optimizer(self, *args, **kwargs):
        """Initializes the model's optimizer based on the set configurations.

        This method sets up the optimizer for the model's parameters. It includes special
        handling for SGLD-based uncertainty methods by differentiating between backbone
        and output layer parameters.

        Args:
            *args: Variable length argument list for optimizer initialization.
            **kwargs: Arbitrary keyword arguments for optimizer initialization.

        Returns:
            Trainer: The Trainer instance with the initialized optimizer.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = AdamW(params, lr=self._status.lr)

        # for sgld compatibility
        if self.config.uncertainty_method == UncertaintyMethods.sgld:
            output_param_ids = [id(x[1]) for x in self._model.named_parameters() if "output_layer" in x[0]]
            backbone_params = filter(
                lambda p: id(p) not in output_param_ids,
                self._model.parameters(),
            )
            output_params = filter(lambda p: id(p) in output_param_ids, self._model.parameters())

            self._optimizer = AdamW(backbone_params, lr=self._status.lr)
            sgld_optimizer = PSGLDOptimizer if self.config.apply_preconditioned_sgld else SGLDOptimizer
            self._sgld_optimizer = sgld_optimizer(
                output_params,
                lr=self._status.lr,
                norm_sigma=self.config.sgld_prior_sigma,
            )

        return self

    def initialize_scheduler(self):
        """Initializes the learning rate scheduler based on the training configuration.

        This method sets up the learning rate scheduler using the total number of training
        steps and the specified warmup ratio.

        Returns:
            Trainer: The Trainer instance with the initialized scheduler.
        """
        n_training_steps = int(np.ceil(self.n_update_steps_per_epoch * self._status.n_epochs))
        n_warmup_steps = int(np.ceil(n_training_steps * self.config.warmup_ratio))

        self._scheduler = get_scheduler(
            self._status.lr_scheduler_type,
            self._optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=n_training_steps,
        )
        return self

    def initialize_loss(self, disable_focal_loss=False):
        """Initializes the loss function based on the task type and specified uncertainty method.

        This method sets up the appropriate loss function for the training process, considering
        the task type (classification or regression) and whether any specific uncertainty
        methods (e.g., evidential or focal loss) are applied.

        Args:
            disable_focal_loss (bool, optional): If True, disables the use of focal loss, even if
                                                 specified by the uncertainty method. Defaults to False.

        Returns:
            Trainer: The Trainer instance with the initialized loss function.
        """
        # Notice that the reduction should always be 'none' here for the following masking operation
        if self.config.task_type == "classification":
            # evidential classification
            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                self._loss_fn = EvidentialClassificationLoss(
                    n_classes=2 if self.config.n_lbs == 1 else self.config.n_lbs,
                    n_steps_per_epoch=self.n_update_steps_per_epoch,
                    annealing_epochs=self.config.evidential_clx_loss_annealing_epochs,
                    device=self._device,
                )
            # focal loss
            elif self.config.uncertainty_method == UncertaintyMethods.focal and not disable_focal_loss:
                self._loss_fn = SigmoidFocalLoss(reduction="none")
            else:
                self._loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                self._loss_fn = EvidentialRegressionLoss(
                    coeff=self.config.evidential_reg_loss_weight,
                    reduction="none",
                )
            elif self.config.regression_with_variance:
                self._loss_fn = GaussianNLLLoss(reduction="none")
            else:
                self._loss_fn = nn.MSELoss(reduction="none")

        return self

    @property
    def training_dataset(self):
        return self._training_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def standardize_training_lbs(self):
        """Standardizes the label distribution of the training dataset for regression tasks.

        This method applies standardization to the labels of the training dataset, transforming
        them to a standard Gaussian distribution. It's applicable only for regression tasks.

        Returns:
            Trainer: The Trainer instance with standardized training labels.
        """
        if self.config.task_type == "classification":
            return self

        if self._training_dataset is None and self._scaler is None:
            logger.warning(
                "Encounter regression task with no training dataset specified and label scaling disabled! "
                "This may create inconsistency between training and inference label scales."
            )
            return self

        # just to make sure that the lbs are not already standardized
        self._training_dataset.toggle_standardized_lbs(False)

        lbs = copy.deepcopy(self._training_dataset.lbs)
        lbs[~self._training_dataset.masks.astype(bool)] = np.nan
        self._scaler = StandardScaler(replace_nan_token=0).fit(lbs)

        if not self._training_dataset.has_standardized_lbs:
            self._training_dataset.set_standardized_lbs(self._scaler.transform(self._training_dataset.lbs))

        return self

    @property
    def n_training_steps(self):
        """
        The number of total training steps
        """
        n_steps_per_epoch = int(np.ceil(len(self._training_dataset) / self.config.batch_size))
        return n_steps_per_epoch * self._status.n_epochs

    @property
    def n_valid_steps(self):
        """
        The number of total validation steps
        """
        n_steps_per_epoch = int(np.ceil(len(self._valid_dataset) / self.config.batch_size))
        return n_steps_per_epoch * self._status.n_epochs

    def set_mode(self, mode: str):
        """Sets the training mode for the model.

        Depending on the mode, the model is set to training, evaluation, or testing. This method
        is essential for correctly configuring the model's state for different phases of the
        training and evaluation process.

        Args:
            mode (str): The mode to set the model to. Should be one of 'train', 'eval', or 'test'.

        Returns:
            Trainer: The Trainer instance with the model set to the specified mode.
        """

        if mode == "train":
            self.model.train()

        elif mode == "eval":
            self.model.eval()

        elif mode == "test":
            self.model.eval()

            if self.config.uncertainty_method == UncertaintyMethods.mc_dropout:
                # activate the dropout layers during evaluation for MC Dropout
                for m in self.model.modules():
                    if m.__class__.__name__.startswith("Dropout"):
                        m.train()
        else:
            raise ValueError(f"Argument `mode` should be 'train', 'eval' or 'test'.")

        return self

    def run(self):
        """Executes the training and evaluation process.

        This method serves as the main entry point for the training process, orchestrating the
        execution based on the specified uncertainty method. It handles different training
        strategies like ensembles, SWAG, temperature scaling, and more.

        Returns:
            None
        """

        # deep ensembles
        if self.config.uncertainty_method == UncertaintyMethods.ensembles:
            self.run_ensembles()
        # swag
        elif self.config.uncertainty_method == UncertaintyMethods.swag:
            self.run_swag()
        # temperature scaling
        elif self.config.uncertainty_method == UncertaintyMethods.temperature:
            self.run_temperature_scaling()
        # isotonic calibration
        elif self.config.uncertainty_method == UncertaintyMethods.iso:
            self.run_iso_calibration()
        # focal loss
        elif self.config.uncertainty_method == UncertaintyMethods.focal:
            self.run_focal_loss()
        # sgld
        elif self.config.uncertainty_method == UncertaintyMethods.sgld:
            self.run_sgld()
        # none & MC Dropout & BBP & Evidential
        else:
            self.run_single_shot()

        wandb.finish()

        logger.info("Done.")

        return None

    def run_single_shot(self, apply_test=True):
        """Runs the training and evaluation pipeline for a single iteration.

        This method handles the process of training the model and optionally evaluating it on a test dataset.
        It is designed for a straightforward, single iteration of training and testing.

        Args:
            apply_test (bool, optional): Whether to run the test function as part of the process. Defaults to True.

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """

        set_seed(self.config.seed)
        if not self.load_checkpoint():
            logger.info("Training model")
            self.train()

        if apply_test:
            test_metrics = self.test()
            if test_metrics is not None:
                logger.info("[Single Shot] Test results:")
                self.log_results(test_metrics)

                # log results to wandb
                for k, v in test_metrics.items():
                    wandb.run.summary[f"test-single_shot/{k}"] = v

            if self.config.test_on_training_data:
                logger.info("[Single Shot] Testing on training data.")
                self.test_on_training_data()

        self.save_checkpoint()

        return self

    def run_ensembles(self):
        """Trains and evaluates an ensemble of models.

        This method is used for uncertainty estimation through model ensembles, training multiple
        models with different seeds and evaluating their collective performance.

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """

        for ensemble_idx in range(self.config.n_ensembles):
            # update random seed and re-initialize training status
            individual_seed = self.config.seed + ensemble_idx

            del self._model
            set_seed(individual_seed)
            self.re_initialize()
            logger.info(f"[Ensemble {ensemble_idx}] seed: {individual_seed}")

            # update result dir
            self._status.result_dir = osp.join(
                osp.dirname(osp.normpath(self._status.result_dir)),
                f"seed-{individual_seed}",
            )
            self._status.result_dir_no_uncertainty = osp.join(
                osp.dirname(osp.normpath(self._status.result_dir_no_uncertainty)),
                f"seed-{individual_seed}",
            )

            if not self.load_checkpoint():
                logger.info("Training model")
                self.train()

            test_metrics = self.test()
            if test_metrics is not None:
                logger.info(f"[Ensemble {ensemble_idx}] Test results:")
                self.log_results(test_metrics)

                # log results to wandb
                for k, v in test_metrics.items():
                    wandb.run.summary[f"test-ensemble_{ensemble_idx}/{k}"] = v

            if self.config.test_on_training_data:
                logger.info(f"[Ensemble {ensemble_idx}] Testing on training data.")
                self.test_on_training_data()

            self.save_checkpoint()

        return self

    def run_swag(self):
        """Executes the training and evaluation pipeline using the SWAG method.

        Stochastic Weight Averaging Gaussian (SWAG) is used for uncertainty estimation. This method
        involves training the model with early stopping and applying SWAG for post-training uncertainty
        estimation.

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """

        # Train the model with early stopping.
        self.run_single_shot(apply_test=False)
        self._load_model_state_dict()

        logger.info("SWA session start")
        self.swa_session()

        test_metrics = self.test(load_best_model=False)
        if test_metrics is not None:
            logger.info("[SWAG] Test results:")
            self.log_results(test_metrics)

            # log results to wandb
            for k, v in test_metrics.items():
                wandb.run.summary[f"test-swag/{k}"] = v

        if self.config.test_on_training_data:
            logger.info("[SWAG] Testing on training data.")
            self.test_on_training_data(load_best_model=False)

        return self

    def swa_session(self):
        """Executes the SWA session.

        This method is intended to be overridden by child classes for specialized handling of
        optimizer or model initialization required by SWA (Stochastic Weight Averaging).

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """
        # update hyper parameters
        self._status.lr *= self.config.swa_lr_decay
        self._status.lr_scheduler_type = "constant"
        self._status.n_epochs = self.config.n_swa_epochs
        # Can also set this to None; disable validation
        self._status.valid_epoch_interval = 0

        self.initialize_optimizer()
        self.initialize_scheduler()

        self._model.to(self._device)
        self._swa_model = SWAModel(
            model=self.model,
            k_models=self.config.k_swa_checkpoints,
            device=self._device,
        )

        logger.info("Training model")
        self.train()
        return self

    def run_temperature_scaling(self):
        """Executes the training and evaluation pipeline with temperature scaling.

        Temperature scaling is applied as a post-processing step to calibrate the confidence of the model's predictions.

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """

        # Train the model with early stopping.
        self.run_single_shot(apply_test=False)
        self._load_model_state_dict()

        logger.info("Temperature Scaling session start.")
        self.ts_session()

        test_metrics = self.test(load_best_model=False)
        if test_metrics is not None:
            logger.info("[Temperature Scaling] Test results:")
            self.log_results(test_metrics)

            # log results to wandb
            for k, v in test_metrics.items():
                wandb.run.summary[f"test-temperature_scaling/{k}"] = v

        if self.config.test_on_training_data:
            logger.info("[Temperature Scaling] Testing on training data.")
            self.test_on_training_data(load_best_model=False)

        return self

    def ts_session(self):
        """Executes the temperature scaling session.

        This session involves retraining the model on the validation set with a modified learning rate and
        epochs to apply temperature scaling for model calibration.

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """
        # update hyper parameters
        self._status.lr = self.config.ts_lr
        self._status.lr_scheduler_type = "constant"
        self._status.n_epochs = self.config.n_ts_epochs
        self._status.valid_epoch_interval = 0  # Can also set this to None; disable validation

        self.model.to(self._device)
        self.freeze()
        self._ts_model = TSModel(self._model, self.config.n_tasks)

        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss(disable_focal_loss=True)

        logger.info("Training model on validation")
        self.train(use_valid_dataset=True)
        self.unfreeze()
        return self

    def run_iso_calibration(self):
        """Performs isotonic calibration.

        Isotonic calibration is applied to calibrate the uncertainties of the model's predictions,
        based on the approach described in 'Accurate Uncertainties for Deep Learning Using Calibrated Regression'.

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """
        # Train the model with early stopping.
        self.run_single_shot(apply_test=False)
        self._load_model_state_dict()

        logger.info("Isotonic calibration session start.")

        # get validation predictions
        mean_cal, var_cal = self.inverse_standardize_preds(self.process_logits(self.inference(self.valid_dataset)))

        iso_calibrator = IsotonicCalibration(self.config.n_tasks)
        iso_calibrator.fit(mean_cal, var_cal, self.valid_dataset.lbs, self.valid_dataset.masks)

        mean_test, var_test = self.inverse_standardize_preds(self.process_logits(self.inference(self.test_dataset)))
        iso_calibrator.calibrate(mean_test, var_test, self.test_dataset.lbs, self.test_dataset.masks)

        return self

    def run_focal_loss(self):
        """Runs the training and evaluation pipeline utilizing focal loss.

        Focal loss is used to address class imbalance by focusing more on hard-to-classify examples.
        Temperature scaling can optionally be applied after training with focal loss.

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """
        # Train the model with early stopping. Do not need to load state dict as it is done during test
        self.run_single_shot()

        if self.config.apply_temperature_scaling_after_focal_loss:
            logger.info("[Focal Loss] Temperature Scaling session start.")
            self.ts_session()

            test_metrics = self.test(load_best_model=False)

            if test_metrics is not None:
                logger.info("[Temperature Scaling] Test results:")
                self.log_results(test_metrics)

                # log results to wandb
                for k, v in test_metrics.items():
                    wandb.run.summary[f"test-temperature_scaling/{k}"] = v

            if self.config.test_on_training_data:
                logger.info("[Temperature Scaling] Testing on training data.")
                self.test_on_training_data(load_best_model=False)

        return self

    def run_sgld(self):
        """Executes training and evaluation with Stochastic Gradient Langevin Dynamics (SGLD).

        SGLD is used for uncertainty estimation, incorporating random noise into the gradients to
        explore the model's parameter space more broadly.

        Returns:
            Trainer: Self reference to the Trainer object, allowing for method chaining.
        """
        self.run_single_shot(apply_test=False)
        self._load_model_state_dict()

        logger.info("Langevin Dynamics session start.")
        self._sgld_model_buffer = list()
        self._status.n_epochs = self.config.n_langevin_samples * self.config.sgld_sampling_interval
        self._status.valid_epoch_interval = 0  # Can also set this to None; disable validation

        logger.info("Training model")
        self.train()

        test_metrics = self.test(load_best_model=False)
        if test_metrics is not None:
            logger.info("[SGLD] Test results:")
            self.log_results(test_metrics)

            # log results to wandb
            for k, v in test_metrics.items():
                wandb.run.summary[f"test-sgld/{k}"] = v

        if self.config.test_on_training_data:
            logger.info("[SGLD] Testing on training data.")
            self.test_on_training_data(load_best_model=False)

        return self

    def train(self, use_valid_dataset=False):
        """Executes the training process for the model.

        Optionally allows for training using the validation dataset instead of the training dataset.
        This option can be useful for certain model calibration techniques like temperature scaling.

        Args:
            use_valid_dataset (bool, optional): Determines if the validation dataset should be used
                for training instead of the training dataset. Defaults to False.

        Returns:
            None: This method returns None.
        """

        self.model.to(self._device)
        self.training_dataset.toggle_standardized_lbs(True)
        data_loader = self.get_dataloader(
            self.training_dataset if not use_valid_dataset else self.valid_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
        )

        for epoch_idx in range(self._status.n_epochs):
            logger.info(f"[Training Epoch {self._status.train_log_idx}]")

            self._timer.clean_cache()
            training_loss = self.training_epoch(data_loader)

            # Print the averaged training loss.
            self.log_results(
                {
                    "Training Loss": training_loss,
                    "Average Time per Step": self._timer.time_elapsed_avg,
                }
            )
            wandb.log(
                data={"train/loss": training_loss},
                step=self._status.train_log_idx,
            )

            # Compatibility with SWAG
            if self._swa_model:
                self._swa_model.update_parameters(self.model)

            if self._sgld_model_buffer is not None and (epoch_idx + 1) % self.config.sgld_sampling_interval == 0:
                self.model.to("cpu")
                self._sgld_model_buffer.append(copy.deepcopy(self.model.state_dict()))
                self.model.to(self._device)

            if self._status.valid_epoch_interval and (epoch_idx + 1) % self._status.valid_epoch_interval == 0:
                self.eval_and_save()

            if self._status.n_eval_no_improve > self.config.valid_tolerance:
                logger.warning("Quit training because of exceeding valid tolerance!")
                self._status.n_eval_no_improve = 0
                break

        self.training_dataset.toggle_standardized_lbs()
        return None

    def training_epoch(self, data_loader):
        """Performs a single epoch of training using the provided data loader.

        This method iterates over the data loader, performs the forward pass, computes the loss,
        and updates the model parameters.

        Args:
            data_loader (DataLoader): DataLoader object providing batches of training data.

        Returns:
            float: The average training loss for the epoch.
        """

        self.set_mode("train")
        self.model.to(self._device)

        total_loss = 0.0
        n_instances = 0

        for batch in data_loader:
            batch.to(self._device)

            self._optimizer.zero_grad()
            if self._sgld_optimizer is not None:  # for sgld compatibility
                self._sgld_optimizer.zero_grad()

            # measure training time for a full batch
            if self.config.time_training and len(batch) == self.config.batch_size:
                self._timer.on_measurement_start()

            logits = self.model(batch)
            loss = self.get_loss(logits, batch, n_steps_per_epoch=len(data_loader))

            loss.backward()

            if self.config.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)

            self._optimizer.step()
            if self._sgld_optimizer is not None:  # for sgld compatibility
                self._sgld_optimizer.step()
            self._scheduler.step()

            # end training time measurement
            self._timer.on_measurement_end()

            total_loss += loss.detach().cpu().item() * len(batch)
            n_instances += len(batch)

        self._status.train_log_idx += 1
        avg_loss = total_loss / n_instances

        return avg_loss

    def get_loss(self, logits, batch, n_steps_per_epoch=None) -> torch.Tensor:
        """Computes the loss for a batch of data.

        This method can be overridden by subclasses to implement custom loss computation logic.

        Args:
            logits (torch.Tensor): The predictions or logits produced by the model for the given batch.
            batch (Batch): The batch of training data.
            n_steps_per_epoch (int, optional): Represents the number of batches in a training epoch,
                used specifically for certain uncertainty methods like Bayesian Backpropagation (BBP).

        Returns:
            torch.Tensor: The computed loss for the batch.
        """

        # modify data shapes to accommodate different tasks
        lbs, masks = (
            batch.lbs,
            batch.masks,
        )  # so that we don't mess up batch instances
        bool_masks = masks.to(torch.bool)
        masked_lbs = lbs[bool_masks]

        if self.config.task_type == "classification":
            masked_logits = logits.view(-1, self.config.n_tasks, self.config.n_lbs)[bool_masks]

            if self.config.n_lbs == 1:  # binary classification
                masked_logits = masked_logits.squeeze(-1)

        elif self.config.task_type == "regression":
            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                # gamma, nu, alpha, beta
                masked_logits = logits[bool_masks]
            else:
                assert self.config.regression_with_variance, NotImplementedError
                # mean and var for the last dimension
                masked_logits = logits.view(-1, self.config.n_tasks, self.config.n_lbs)[bool_masks]

        else:
            raise ValueError

        loss = self._loss_fn(masked_logits, masked_lbs)
        loss = torch.sum(loss) / masks.sum()

        # for bbp
        if self.config.uncertainty_method == UncertaintyMethods.bbp and n_steps_per_epoch is not None:
            loss += self.model.output_layer.kld / n_steps_per_epoch / len(batch)
        return loss

    def inference(self, dataset, **kwargs):
        """Conducts inference over an entire dataset using the model.

        Args:
            dataset (Dataset): The dataset for which inference needs to be performed.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: The model outputs as logits or a tuple of logits.
        """

        dataloader = self.get_dataloader(dataset, batch_size=self.config.batch_size_inference, shuffle=False)
        self.model.to(self._device)

        logits_list = list()

        with torch.no_grad():
            for batch in dataloader:
                batch.to(self._device)

                logits = self.model(batch)
                logits_list.append(logits.detach().cpu())

        logits = torch.cat(logits_list, dim=0).numpy()

        return logits

    def process_logits(self, logits: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Processes the output logits based on the training tasks or variants.

        Args:
            logits (numpy.ndarray): The raw logits produced by the model.

        Returns:
            numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]: Processed logits or a tuple containing
            processed logits based on the task type.
        """

        if self.config.task_type == "classification":
            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                logits = logits.reshape((-1, self.config.n_tasks, self.config.n_lbs))

                evidence = logits * (logits > 0)  # relu
                alpha = evidence + 1
                probs = alpha / np.sum(alpha, axis=-1, keepdims=True)

                return probs
            else:
                return expit(logits)  # sigmoid function

        elif self.config.task_type == "regression":
            # reshape the logits if the task and output-lbs (with shape config.n_lbs) dimensions are tangled
            if self.config.n_tasks > 1 and len(logits.shape) == 2:
                logits = logits.reshape((-1, self.config.n_tasks, self.config.n_lbs))

            # get the mean and variance of the preds
            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                gamma, _, alpha, beta = np.split(logits, 4, axis=-1)
                mean = gamma.squeeze(-1)
                var = (beta / (alpha - 1)).squeeze(-1)
                return mean, var

            # get the mean and variance of the preds
            elif self.config.regression_with_variance:
                mean = logits[..., 0]
                var = F.softplus(torch.from_numpy(logits[..., 1])).numpy()
                return mean, var

            else:
                return logits

        else:
            raise ValueError(f"Unrecognized task type: {self.config.task_type}")

    def inverse_standardize_preds(
        self, preds: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transforms predictions back to their original scale if they have been standardized.

        Args:
            preds (numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]): Model predictions, can either be a single
                array or a tuple containing two arrays for mean and variance, respectively.

        Returns:
            numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]: Inverse-standardized predictions.
        """
        if self._scaler is None:
            return preds

        if isinstance(preds, np.ndarray):
            return self._scaler.inverse_transform(preds)
        elif isinstance(preds, tuple):
            mean, var = self._scaler.inverse_transform(*preds)
            return mean, var
        else:
            raise TypeError(f"Unsupported prediction type: {type(preds)}!")

    def evaluate(
        self,
        dataset,
        n_run: Optional[int] = 1,
        return_preds: Optional[bool] = False,
    ):
        """Evaluates the model's performance on the given dataset.

        Args:
            dataset (Dataset): The dataset to evaluate the model on.
            n_run (int, optional): Number of runs for evaluation. Defaults to 1.
            return_preds (bool, optional): Whether to return the predictions along with metrics. Defaults to False.

        Returns:
            dict, or (dict, numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]): Evaluation metrics, or tuple
            containing metrics and predictions based on `return_preds`.
        """
        if self._sgld_model_buffer is not None and len(self._sgld_model_buffer) > 0:
            n_run = len(self._sgld_model_buffer)

        if n_run == 1:
            preds = self.inverse_standardize_preds(self.process_logits(self.inference(dataset)))
            if isinstance(preds, np.ndarray):
                metrics = self.get_metrics(dataset.lbs, preds, dataset.masks)
            else:  # isinstance(preds, tuple)
                metrics = self.get_metrics(dataset.lbs, preds[0], dataset.masks)

            return metrics if not return_preds else (metrics, preds)

        # Multiple test runs
        preds_list = list()
        vars_list = list()
        for test_run_idx in range(n_run):
            logger.info(f"[Test {test_run_idx + 1}]")

            individual_seed = self.config.seed + test_run_idx
            set_seed(individual_seed)

            if self._sgld_model_buffer:
                self._model.load_state_dict(self._sgld_model_buffer[test_run_idx])
                self._model.to(self._device)

            if self._swa_model:
                self._model = self._swa_model.sample_parameters()
                update_bn(
                    model=self.model,
                    training_loader=self.get_dataloader(self.training_dataset, shuffle=True),
                    device=self.config.device,
                )

            preds = self.inverse_standardize_preds(self.process_logits(self.inference(dataset)))
            if isinstance(preds, np.ndarray):
                preds_list.append(preds)
            else:
                preds_list.append(preds[0])
                vars_list.append(preds[1])

        preds_array = np.stack(preds_list)
        vars_array = None if not vars_list else np.stack(vars_list)
        metrics = self.get_metrics(dataset.lbs, preds_array.mean(axis=0), dataset.masks)

        if not return_preds:
            return metrics
        elif vars_array is None:
            return metrics, preds_array
        else:
            return metrics, (preds_array, vars_array)

    def eval_and_save(self):
        """Evaluates the model's performance on the validation dataset and saves it if its performance is improved.

        This method is part of the training loop where the model is periodically evaluated on the validation dataset,
        and the best-performing model state is saved.
        """
        self.set_mode("eval")

        valid_results = self.evaluate(self.valid_dataset)

        result_dict = {f"valid/{k}": v for k, v in valid_results.items()}
        wandb.log(data=result_dict, step=self._status.train_log_idx)

        logger.info(f"[Valid step {self._status.eval_log_idx}] results:")
        self.log_results(valid_results)

        # ----- check model performance and update buffer -----
        if self._checkpoint_container.check_and_update(self.model, valid_results[self._valid_metric]):
            self._status.n_eval_no_improve = 0
            logger.info("Model buffer is updated!")
        else:
            self._status.n_eval_no_improve += 1
        self._status.eval_log_idx += 1

        return None

    def test(self, load_best_model=True, return_preds=False):
        """Tests the model's performance on the test dataset.

        Args:
            load_best_model (bool, optional): Whether to load the best model saved during training for testing. Defaults to True.
            return_preds (bool, optional): Whether to return the predictions along with metrics. Defaults to False.

        Returns:
            dict, or tuple[dict, numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]]: Evaluation metrics (and predictions)
            for the test dataset.
        """
        if self.config.disable_test:
            logger.warning("Test is disabled!")
            return None

        self.set_mode("test")

        if load_best_model and self._checkpoint_container.state_dict:
            self._load_model_state_dict()

        metrics, preds = self.evaluate(self._test_dataset, n_run=self.config.n_test, return_preds=True)

        # save preds
        if self.config.n_test == 1:
            if isinstance(preds, np.ndarray):
                preds = preds.reshape(1, *preds.shape)
            else:
                preds = tuple([p.reshape(1, *p.shape) for p in preds])

        if isinstance(preds, np.ndarray):
            variances = [None] * len(preds)
        elif isinstance(preds, tuple) and len(preds) == 2:
            preds, variances = preds
        else:
            raise ValueError("Unrecognized type or shape of `preds`.")

        for idx, (pred, variance) in enumerate(zip(preds, variances)):
            if self.config.test_subset_ids_file_name:
                subset_ids_file_name_no_ext = (
                    osp.splitext(self.config.test_subset_ids_file_name)[0]
                    if "." in self.config.test_subset_ids_file_name
                    else self.config.test_subset_ids_file_name
                )
                file_path = osp.join(self._status.result_dir, f"preds-{subset_ids_file_name_no_ext}", f"{idx}.pt")
            else:
                file_path = osp.join(self._status.result_dir, "preds", f"{idx}.pt")

            self.save_results(
                path=file_path,
                preds=pred,
                variances=variance,
                lbs=self._test_dataset.lbs,
                masks=self.test_dataset.masks,
            )

        return metrics

    def test_on_training_data(self, load_best_model=True, return_preds=False, disable_result_saving=False):
        """Tests the model's performance on the training dataset.

        This method is useful for understanding the model's performance on the data it was trained on,
        which can provide insights into overfitting or underfitting.

        Args:
            load_best_model (bool, optional): If True, loads the best model saved during training. Defaults to True.
            return_preds (bool, optional): If True, returns the predictions along with the evaluation metrics. Defaults to False.
            disable_result_saving (bool, optional): If True, disables saving the results to disk. Defaults to False.

        Returns:
            dict, or tuple[dict, numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]]: Evaluation metrics, or a tuple containing metrics and predictions if `return_preds` is True.
        """
        self.set_mode("test")
        self._training_dataset.use_full_dataset = True
        self._training_dataset.toggle_standardized_lbs(False)

        if load_best_model and self._checkpoint_container.state_dict:
            self._load_model_state_dict()

        metrics, preds = self.evaluate(self._training_dataset, n_run=self.config.n_test, return_preds=True)

        # save preds
        if self.config.n_test == 1:
            if isinstance(preds, np.ndarray):
                preds = preds.reshape(1, *preds.shape)
            else:
                preds = tuple([p.reshape(1, *p.shape) for p in preds])

        if isinstance(preds, np.ndarray):
            means = preds
            variances = [None] * len(preds)
        elif isinstance(preds, tuple) and len(preds) == 2:
            means, variances = preds
        else:
            raise ValueError("Unrecognized type or shape of `preds`.")

        if not disable_result_saving:
            for idx, (mean, variance) in enumerate(zip(means, variances)):
                file_path = osp.join(self._status.result_dir, "preds-train", f"{idx}.pt")
                self.save_results(
                    path=file_path,
                    preds=mean,
                    variances=variance,
                    lbs=self._test_dataset.lbs,
                    masks=self.test_dataset.masks,
                )

        self._training_dataset.use_full_dataset = False

        return (metrics, preds) if return_preds else metrics

    def get_metrics(self, lbs, preds, masks):
        """Calculates evaluation metrics based on the given labels, predictions, and masks.

        This method computes the appropriate metrics based on the task type (classification or regression).

        Args:
            lbs (numpy.ndarray): Ground truth labels.
            preds (numpy.ndarray): Model predictions.
            masks (numpy.ndarray): Masks indicating valid entries in labels and predictions.

        Returns:
            dict: Computed metrics for evaluation.
        """
        if masks.shape[-1] == 1 and len(masks.shape) > 1:
            masks = masks.squeeze(-1)
        bool_masks = masks.astype(bool)

        if lbs.shape[-1] == 1 and len(lbs.shape) > 1:
            lbs = lbs.squeeze(-1)

        if self.config.task_type == "classification" and self.config.n_tasks > 1:
            preds = preds.reshape(-1, self.config.n_tasks, self.config.n_lbs)
            if self.config.n_lbs == 2:
                preds = preds[..., -1:]
        if preds.shape[-1] == 1 and len(preds.shape) > 1:  # remove tailing axis
            preds = preds.squeeze(-1)

        if self.config.task_type == "classification":
            metrics = calculate_binary_classification_metrics(lbs, preds, bool_masks, self._valid_metric)
        else:
            metrics = calculate_regression_metrics(lbs, preds, bool_masks, self._valid_metric)

        return metrics

    @staticmethod
    def log_results(metrics: dict, logging_func=logger.info):
        """Logs evaluation metrics using the specified logging function.

        Args:
            metrics (dict): Dictionary containing evaluation metrics to be logged.
            logging_func (function, optional): Logging function to which metrics will be sent. Defaults to `logger.info`.

        Returns:
            None
        """
        for k, v in metrics.items():
            try:
                logging_func(f"  {k}: {v:.4f}.")
            except TypeError:
                pass
        return None

    def freeze(self):
        """Freezes all model parameters, preventing them from being updated during training.

        Returns:
            Trainer: The current instance with model parameters frozen.
        """
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self._model_frozen = True
        return self

    def unfreeze(self):
        """Unfreezes all model parameters, allowing them to be updated during training.

        Returns:
            Trainer: The current instance with model parameters unfrozen.
        """
        for parameter in self.model.parameters():
            parameter.requires_grad = True

        self._model_frozen = False
        return self

    def freeze_backbone(self):
        """Freezes the backbone parameters of the model, preventing them from being updated during training.

        Returns:
            Trainer: The current instance with backbone parameters frozen.
        """
        for params in self.backbone_params:
            params.requires_grad = False

        self._backbone_frozen = True
        return self

    def unfreeze_backbone(self):
        """Unfreezes the backbone parameters of the model, allowing them to be updated during training.

        Returns:
            Trainer: The current instance with backbone parameters unfrozen.
        """
        for params in self.backbone_params:
            params.requires_grad = True

        self._backbone_frozen = False
        return self

    def save_checkpoint(self):
        """Saves the current model state as a checkpoint.

        This method checks the `disable_result_saving` configuration flag before saving. If saving is disabled,
        it logs a warning and does not perform the save operation.

        Returns:
            Trainer: The current instance after attempting to save the model checkpoint.
        """
        if not self.config.disable_result_saving:
            init_dir(self._status.result_dir, clear_original_content=False)
            self._checkpoint_container.save(osp.join(self._status.result_dir, self._status.model_name))
        else:
            logger.warning("Model is not saved because of `disable_result_saving` flag is set to `True`.")

        return self

    def _load_model_state_dict(self):
        """Loads the model state dictionary from the checkpoint container.

        This private method is used internally to load model parameters from a saved checkpoint. If the `freeze_backbone`
        configuration flag is set, the backbone parameters are frozen after loading.

        Returns:
            Trainer: The current instance with the model state loaded.
        """
        self._model.load_state_dict(self._checkpoint_container.state_dict)
        if self.config.freeze_backbone:
            self.freeze_backbone()
        return self

    def _load_from_container(self, model_path):
        """Loads a trained model from the specified path.

        Args:
            model_path (str): Path to the saved model.

        Returns:
            bool: True if the model is successfully loaded, otherwise False.
        """
        if not osp.exists(model_path):
            return False
        logger.info(f"Loading trained model from {model_path}.")
        self._checkpoint_container.load(model_path)
        self._load_model_state_dict()
        self._model.to(self._device)
        return True

    def load_checkpoint(self):
        """Loads the model from a checkpoint.

        This method attempts to load the model checkpoint from the configured path. It supports loading with and without
        considering the uncertainty estimation method used during training.

        Returns:
            bool: True if the model is successfully loaded from a checkpoint, otherwise False.
        """
        if self.config.retrain_model:
            return False

        if not self.config.ignore_uncertainty_output:
            model_path = osp.join(self._status.result_dir, self._status.model_name)
            if self._load_from_container(model_path):
                return True

        if not self.config.ignore_no_uncertainty_output:
            model_path = osp.join(self._status.result_dir_no_uncertainty, self._status.model_name)
            if self._load_from_container(model_path):
                return True

        return False

    def get_dataloader(
        self,
        dataset,
        shuffle: Optional[bool] = False,
        batch_size: Optional[int] = 0,
    ):
        """Creates a DataLoader for the specified dataset.

        Args:
            dataset: Dataset for which the DataLoader is to be created.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            batch_size (int, optional): Batch size for the DataLoader. Uses the batch size from the configuration if not specified.

        Returns:
            DataLoader: The created DataLoader for the provided dataset.
        """
        try:
            dataloader = DataLoader(
                dataset=dataset,
                collate_fn=self._collate_fn,
                batch_size=batch_size if batch_size else self.config.batch_size,
                num_workers=getattr(self.config, "num_workers", 0),
                pin_memory=getattr(self.config, "pin_memory", False),
                shuffle=shuffle,
                drop_last=False,
            )
        except Exception as e:
            logger.exception(e)
            raise e

        return dataloader

    def save_results(self, path, preds, variances, lbs, masks):
        """Saves the model predictions, variances, ground truth labels, and masks to disk.

        This method saves the results of model predictions to a specified path. It is capable of handling
        both the predictions and their associated variances, along with the ground truth labels and masks that
        indicate which data points should be considered in the analysis. If the configuration flag
        `disable_result_saving` is set to True, the method will log a warning and not perform any saving operation.

        Args:
            path (str): The destination path where the results will be saved.
            preds (array_like): The predictions generated by the model.
            variances (array_like): The variances associated with each prediction, indicating the uncertainty of the predictions.
            lbs (array_like): The ground truth labels against which the model's predictions can be evaluated.
            masks (array_like): Masks indicating which data points are valid and should be considered in the evaluation.

        Returns:
            None: This method does not return any value.
        """
        if not self.config.disable_result_saving:
            save_results(path, preds, variances, lbs, masks)
        else:
            logger.warning("Results are not saved because `disable_result_saving` flag is set to `True`.")

        return None
