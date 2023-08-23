"""
# Author: Yinghao Li
# Modified: August 23rd, 2023
# ---------------------------------------
# Description: Base trainer functions.
"""

import copy
import wandb
import logging
import numpy as np
import os.path as op
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

from ..args import Config
from ..model import CheckpointContainer, UpdateCriteria
from ..uncertainty import (
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
    def __init__(
        self,
        config,
        training_dataset=None,
        valid_dataset=None,
        test_dataset=None,
        collate_fn=None,
        scalar=None,
    ):
        self._config = config
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

        # Validation variables and flags
        self._valid_metric = (
            EVAL_METRICS[config.dataset_name].replace("-", "_")
            if not config.eval_metric
            else config.eval_metric
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
        self.initialize()

        logger.info(
            f"Trainer initialized. The model contains {self.n_model_parameters} parameters"
        )

    @property
    def model(self):
        # return the scaled model if it exits
        if self._ts_model:
            return self._ts_model
        return self._model

    @property
    def config(self) -> Config:
        return self._config

    @property
    def n_model_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def n_update_steps_per_epoch(self):
        return int(np.ceil(len(self._training_dataset) / self.config.batch_size))

    @property
    def backbone_params(self):
        output_param_ids = [
            id(x[1]) for x in self._model.named_parameters() if "output_layer" in x[0]
        ]
        backbone_params = list(
            filter(lambda p: id(p) not in output_param_ids, self._model.parameters())
        )
        return backbone_params

    def initialize(self):
        """
        Initialize trainer status
        """
        self.initialize_model()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss()
        self._timer.init()
        self._status.init()
        self._checkpoint_container = CheckpointContainer(self._update_criteria)

        if self.config.freeze_backbone:
            self.freeze_backbone()
        return self

    def initialize_model(self, *args, **kwargs):
        raise NotImplementedError

    def initialize_optimizer(self, *args, **kwargs):
        """
        Initialize model optimizer
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = AdamW(params, lr=self._status.lr)

        # for sgld compatibility
        if self.config.uncertainty_method == UncertaintyMethods.sgld:
            output_param_ids = [
                id(x[1])
                for x in self._model.named_parameters()
                if "output_layer" in x[0]
            ]
            backbone_params = filter(
                lambda p: id(p) not in output_param_ids, self._model.parameters()
            )
            output_params = filter(
                lambda p: id(p) in output_param_ids, self._model.parameters()
            )

            self._optimizer = AdamW(backbone_params, lr=self._status.lr)
            sgld_optimizer = (
                PSGLDOptimizer
                if self.config.apply_preconditioned_sgld
                else SGLDOptimizer
            )
            self._sgld_optimizer = sgld_optimizer(
                output_params,
                lr=self._status.lr,
                norm_sigma=self.config.sgld_prior_sigma,
            )

        return self

    def initialize_scheduler(self):
        """
        Initialize learning rate scheduler
        """
        n_training_steps = int(
            np.ceil(self.n_update_steps_per_epoch * self._status.n_epochs)
        )
        n_warmup_steps = int(np.ceil(n_training_steps * self.config.warmup_ratio))

        self._scheduler = get_scheduler(
            self._status.lr_scheduler_type,
            self._optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=n_training_steps,
        )
        return self

    def initialize_loss(self, disable_focal_loss=False):
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
            elif (
                self.config.uncertainty_method == UncertaintyMethods.focal
                and not disable_focal_loss
            ):
                self._loss_fn = SigmoidFocalLoss(reduction="none")
            else:
                self._loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                self._loss_fn = EvidentialRegressionLoss(
                    coeff=self.config.evidential_reg_loss_weight, reduction="none"
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
        """
        Convert the label distribution in the training dataset to standard Gaussian.
        Notice that this function only works for regression tasks
        """
        if self.config.task_type == "classification":
            return self
        if self._training_dataset is None and self._scaler is None:
            logger.warning(
                "Encounter regression task with no training dataset specified and label scaling disabled! "
                "This may create inconsistency between training and inference label scales."
            )
            return self

        lbs = copy.deepcopy(self._training_dataset.lbs)
        lbs[~self._training_dataset.masks.astype(bool)] = np.nan
        self._scaler = StandardScaler(replace_nan_token=0).fit(lbs)

        self._training_dataset.update_lbs(
            self._scaler.transform(self._training_dataset.lbs)
        )

        return self

    @property
    def n_training_steps(self):
        """
        The number of total training steps
        """
        n_steps_per_epoch = int(
            np.ceil(len(self._training_dataset) / self.config.batch_size)
        )
        return n_steps_per_epoch * self._status.n_epochs

    @property
    def n_valid_steps(self):
        """
        The number of total validation steps
        """
        n_steps_per_epoch = int(
            np.ceil(len(self._valid_dataset) / self.config.batch_size)
        )
        return n_steps_per_epoch * self._status.n_epochs

    def set_mode(self, mode: str):
        """
        Specify training mode using string.

        Parameters
        ----------
        mode: train or valid
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
        """
        Run the training and evaluation process. This is the main/entry function of the Trainer class.

        Returns
        -------
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
        """
        Run the training and evaluation pipeline once. No pre- or post-processing is applied.

        Parameters
        ----------
        apply_test: whether run the test function in the process.

        Returns
        -------
        self
        """

        set_seed(self.config.seed)
        if not self.load_checkpoint():
            logger.info("Training model")
            self.train()

        if apply_test:
            test_metrics = self.test()
            logger.info("[Single Shot] Test results:")
            self.log_results(test_metrics)

            # log results to wandb
            for k, v in test_metrics.items():
                wandb.run.summary[f"test-single_shot/{k}"] = v

        self.save_checkpoint()

        return self

    def run_ensembles(self):
        """
        Run an ensemble of models. Used as the implementation of the Model Ensembles for uncertainty estimation.
        """

        for ensemble_idx in range(self.config.n_ensembles):
            # update random seed and re-initialize training status
            individual_seed = self.config.seed + ensemble_idx

            del self._model
            set_seed(individual_seed)
            self.initialize()
            logger.info(f"[Ensemble {ensemble_idx}] seed: {individual_seed}")

            # update result dir
            self._status.result_dir = op.join(
                op.dirname(op.normpath(self._status.result_dir)),
                f"seed-{individual_seed}",
            )
            self._status.result_dir_no_uncertainty = op.join(
                op.dirname(op.normpath(self._status.result_dir_no_uncertainty)),
                f"seed-{individual_seed}",
            )

            if not self.load_checkpoint():
                logger.info("Training model")
                self.train()

            test_metrics = self.test()
            logger.info(f"[Ensemble {ensemble_idx}] Test results:")
            self.log_results(test_metrics)

            # log results to wandb
            for k, v in test_metrics.items():
                wandb.run.summary[f"test-ensemble_{ensemble_idx}/{k}"] = v

            self.save_checkpoint()

        return self

    def run_swag(self):
        """
        Run the training and evaluation pipeline with SWAG uncertainty estimation method.
        """

        # Train the model with early stopping.
        self.run_single_shot(apply_test=False)
        self._load_model_state_dict()

        logger.info("SWA session start")
        self.swa_session()

        test_metrics = self.test(load_best_model=False)
        logger.info("[SWAG] Test results:")
        self.log_results(test_metrics)

        # log results to wandb
        for k, v in test_metrics.items():
            wandb.run.summary[f"test-swag/{k}"] = v

        return self

    def swa_session(self):
        # update hyper parameters
        self._status.lr *= self.config.swa_lr_decay
        self._status.lr_scheduler_type = "constant"
        self._status.n_epochs = self.config.n_swa_epochs
        self._status.valid_epoch_interval = (
            0  # Can also set this to None; disable validation
        )

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
        """
        Run the training and evaluation pipeline with temperature scaling.
        """

        # Train the model with early stopping.
        self.run_single_shot(apply_test=False)
        self._load_model_state_dict()

        logger.info("Temperature Scaling session start.")
        self.ts_session()

        test_metrics = self.test(load_best_model=False)
        logger.info("[Temperature Scaling] Test results:")
        self.log_results(test_metrics)

        # log results to wandb
        for k, v in test_metrics.items():
            wandb.run.summary[f"test-temperature_scaling/{k}"] = v

        return self

    def ts_session(self):
        # update hyper parameters
        self._status.lr = self.config.ts_lr
        self._status.lr_scheduler_type = "constant"
        self._status.n_epochs = self.config.n_ts_epochs
        self._status.valid_epoch_interval = (
            0  # Can also set this to None; disable validation
        )

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
        """
        Run the isotonic calibration process proposed in
        `Accurate Uncertainties for Deep Learning Using Calibrated Regression`.
        """
        # Train the model with early stopping.
        self.run_single_shot(apply_test=False)
        self._load_model_state_dict()

        logger.info("Isotonic calibration session start.")

        # get validation predictions
        mean_cal, var_cal = self.inverse_standardize_preds(
            self.process_logits(self.inference(self.valid_dataset))
        )

        iso_calibrator = IsotonicCalibration(self.config.n_tasks)
        iso_calibrator.fit(
            mean_cal, var_cal, self.valid_dataset.lbs, self.valid_dataset.masks
        )

        mean_test, var_test = self.inverse_standardize_preds(
            self.process_logits(self.inference(self.test_dataset))
        )
        iso_calibrator.calibrate(
            mean_test, var_test, self.test_dataset.lbs, self.test_dataset.masks
        )

        return self

    def run_focal_loss(self):
        """
        Run the training and evaluation pipeline with focal loss.
        """
        # Train the model with early stopping. Do not need to load state dict as it is done during test
        self.run_single_shot()

        if self.config.apply_temperature_scaling_after_focal_loss:
            logger.info("[Focal Loss] Temperature Scaling session start.")
            self.ts_session()

            test_metrics = self.test(load_best_model=False)
            logger.info("[Temperature Scaling] Test results:")
            self.log_results(test_metrics)

            # log results to wandb
            for k, v in test_metrics.items():
                wandb.run.summary[f"test-temperature_scaling/{k}"] = v

        return self

    def run_sgld(self):
        """
        Run the training and evaluation steps with stochastic gradient Langevin Dynamics
        """
        self.run_single_shot(apply_test=False)
        self._load_model_state_dict()

        logger.info("Langevin Dynamics session start.")
        self._sgld_model_buffer = list()
        self._status.n_epochs = (
            self.config.n_langevin_samples * self.config.sgld_sampling_interval
        )
        self._status.valid_epoch_interval = (
            0  # Can also set this to None; disable validation
        )

        logger.info("Training model")
        self.train()

        test_metrics = self.test(load_best_model=False)
        logger.info("[SGLD] Test results:")
        self.log_results(test_metrics)

        # log results to wandb
        for k, v in test_metrics.items():
            wandb.run.summary[f"test-sgld/{k}"] = v

        return self

    def train(self, use_valid_dataset=False):
        """
        Run the training process

        Parameters
        ----------
        use_valid_dataset: whether to use the validation dataset to train the model.
            This argument could be true during temperature scaling.

        Returns
        -------
        None
        """

        self.model.to(self._device)
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
                data={"train/loss": training_loss}, step=self._status.train_log_idx
            )

            # Compatibility with SWAG
            if self._swa_model:
                self._swa_model.update_parameters(self.model)

            if (
                self._sgld_model_buffer is not None
                and (epoch_idx + 1) % self.config.sgld_sampling_interval == 0
            ):
                self.model.to("cpu")
                self._sgld_model_buffer.append(copy.deepcopy(self.model.state_dict()))
                self.model.to(self._device)

            if (
                self._status.valid_epoch_interval
                and (epoch_idx + 1) % self._status.valid_epoch_interval == 0
            ):
                self.eval_and_save()

            if self._status.n_eval_no_improve > self.config.valid_tolerance:
                logger.warning("Quit training because of exceeding valid tolerance!")
                self._status.n_eval_no_improve = 0
                break

        return None

    def training_epoch(self, data_loader):
        """
        Train the model for one epoch

        Parameters
        ----------
        data_loader: data loader

        Returns
        -------
        averaged training loss
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
        """
        Children trainers can directly reload this function instead of
        reloading `training epoch`, which could be more complicated

        Parameters
        ----------
        logits: logits predicted by the model
        batch: batched training data
        n_steps_per_epoch: how many batches in a training epoch; only used for BBP.

        Returns
        -------
        loss, torch.Tensor
        """

        # modify data shapes to accommodate different tasks
        lbs, masks = batch.lbs, batch.masks  # so that we don't mess up batch instances
        bool_masks = masks.to(torch.bool)
        masked_lbs = lbs[bool_masks]

        if self.config.task_type == "classification":
            masked_logits = logits.view(-1, self.config.n_tasks, self.config.n_lbs)[
                bool_masks
            ]

            if self.config.n_lbs == 1:  # binary classification
                masked_logits = masked_logits.squeeze(-1)

        elif self.config.task_type == "regression":
            if self.config.uncertainty_method == UncertaintyMethods.evidential:
                # gamma, nu, alpha, beta
                masked_logits = logits[bool_masks]
            else:
                assert self.config.regression_with_variance, NotImplementedError
                # mean and var for the last dimension
                masked_logits = logits.view(-1, self.config.n_tasks, self.config.n_lbs)[
                    bool_masks
                ]

        else:
            raise ValueError

        loss = self._loss_fn(masked_logits, masked_lbs)
        loss = torch.sum(loss) / masks.sum()

        # for bbp
        if (
            self.config.uncertainty_method == UncertaintyMethods.bbp
            and n_steps_per_epoch is not None
        ):
            loss += self.model.output_layer.kld / n_steps_per_epoch / len(batch)
        return loss

    def inference(self, dataset, **kwargs):
        """
        Run the forward inference for an entire dataset.

        Parameters
        ----------
        dataset: dataset

        Returns
        -------
        model outputs (logits or tuple of logits)
        """

        dataloader = self.get_dataloader(
            dataset, batch_size=self.config.batch_size_inference, shuffle=False
        )
        self.model.to(self._device)

        logits_list = list()

        with torch.no_grad():
            for batch in dataloader:
                batch.to(self._device)

                logits = self.model(batch)
                logits_list.append(logits.detach().cpu())

        logits = torch.cat(logits_list, dim=0).numpy()

        return logits

    def process_logits(
        self, logits: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Post-process the output logits according to the training tasks or variants.

        Parameters
        ----------
        logits

        Returns
        -------
        processed model output logits
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
        self, dataset, n_run: Optional[int] = 1, return_preds: Optional[bool] = False
    ):
        if self._sgld_model_buffer is not None and len(self._sgld_model_buffer) > 0:
            n_run = len(self._sgld_model_buffer)

        if n_run == 1:
            preds = self.inverse_standardize_preds(
                self.process_logits(self.inference(dataset))
            )
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
                    training_loader=self.get_dataloader(
                        self.training_dataset, shuffle=True
                    ),
                    device=self.config.device,
                )

            preds = self.inverse_standardize_preds(
                self.process_logits(self.inference(dataset))
            )
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
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """
        self.set_mode("eval")
        self._status.eval_log_idx += 1

        valid_results = self.evaluate(self.valid_dataset)

        result_dict = {f"valid/{k}": v for k, v in valid_results.items()}
        wandb.log(data=result_dict, step=self._status.eval_log_idx)

        logger.info(f"[Valid step {self._status.eval_log_idx}] results:")
        self.log_results(valid_results)

        # ----- check model performance and update buffer -----
        if self._checkpoint_container.check_and_update(
            self.model, valid_results[self._valid_metric]
        ):
            self._status.n_eval_no_improve = 0
            logger.info("Model buffer is updated!")
        else:
            self._status.n_eval_no_improve += 1

        return None

    def test(self, load_best_model=True):
        self.set_mode("test")

        if load_best_model and self._checkpoint_container.state_dict:
            self._load_model_state_dict()

        metrics, preds = self.evaluate(
            self._test_dataset, n_run=self.config.n_test, return_preds=True
        )

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
            file_path = op.join(self._status.result_dir, "preds", f"{idx}.pt")
            self.save_results(
                path=file_path,
                preds=pred,
                variances=variance,
                lbs=self._test_dataset.lbs,
                masks=self.test_dataset.masks,
            )

        return metrics

    def get_metrics(self, lbs, preds, masks):
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
            metrics = calculate_binary_classification_metrics(
                lbs, preds, bool_masks, self._valid_metric
            )
        else:
            metrics = calculate_regression_metrics(
                lbs, preds, bool_masks, self._valid_metric
            )

        return metrics

    @staticmethod
    def log_results(metrics: dict, logging_func=logger.info):
        """
        Print evaluation metrics to the logging destination
        """
        for k, v in metrics.items():
            try:
                logging_func(f"  {k}: {v:.4f}.")
            except TypeError:
                pass
        return None

    def freeze(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self._model_frozen = True
        return self

    def unfreeze(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = True

        self._model_frozen = False
        return self

    def freeze_backbone(self):
        for params in self.backbone_params:
            params.requires_grad = False

        self._backbone_frozen = True
        return self

    def unfreeze_backbone(self):
        for params in self.backbone_params:
            params.requires_grad = True

        self._backbone_frozen = False
        return self

    def save_checkpoint(self):
        if not self.config.disable_result_saving:
            init_dir(self._status.result_dir, clear_original_content=False)
            self._checkpoint_container.save(
                op.join(self._status.result_dir, self._status.model_name)
            )
        else:
            logger.warning(
                "Model is not saved because of `disable_result_saving` flag is set to `True`."
            )

        return self

    def _load_model_state_dict(self):
        self._model.load_state_dict(self._checkpoint_container.state_dict)
        if self.config.freeze_backbone:
            self.freeze_backbone()
        return self

    def _load_from_container(self, model_path):
        if not op.exists(model_path):
            return False
        logger.info(f"Loading trained model from {model_path}.")
        self._checkpoint_container.load(model_path)
        self._load_model_state_dict()
        self._model.to(self._device)
        return True

    def load_checkpoint(self):
        if self.config.retrain_model:
            return False

        if not self.config.ignore_uncertainty_output:
            model_path = op.join(self._status.result_dir, self._status.model_name)
            if self._load_from_container(model_path):
                return True

        if not self.config.ignore_no_uncertainty_output:
            model_path = op.join(
                self._status.result_dir_no_uncertainty, self._status.model_name
            )
            if self._load_from_container(model_path):
                return True

        return False

    def get_dataloader(
        self, dataset, shuffle: Optional[bool] = False, batch_size: Optional[int] = 0
    ):
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

    def save(
        self,
        output_dir: Optional[str] = None,
        save_optimizer: Optional[bool] = False,
        save_scheduler: Optional[bool] = False,
        model_name: Optional[str] = "model.ckpt",
        optimizer_name: Optional[str] = "optimizer.ckpt",
        scheduler_name: Optional[str] = "scheduler.ckpt",
    ):
        """
        Save model parameters as well as trainer parameters

        Parameters
        ----------
        output_dir: model directory
        save_optimizer: whether to save optimizer
        save_scheduler: whether to save scheduler
        model_name: model name (suffix free)
        optimizer_name: optimizer name (suffix free)
        scheduler_name: scheduler name (suffix free)

        Returns
        -------
        None
        """
        output_dir = output_dir if output_dir is not None else self._status.result_dir

        model_state_dict = self._model.state_dict()
        torch.save(model_state_dict, op.join(output_dir, model_name))

        self.config.save(output_dir)

        if save_optimizer:
            torch.save(
                self._optimizer.state_dict(), op.join(output_dir, optimizer_name)
            )
        if save_scheduler and self._scheduler is not None:
            torch.save(
                self._scheduler.state_dict(), op.join(output_dir, scheduler_name)
            )

        return None

    def load(
        self,
        input_dir: Optional[str] = None,
        load_optimizer: Optional[bool] = False,
        load_scheduler: Optional[bool] = False,
        model_name: Optional[str] = "model.ckpt",
        optimizer_name: Optional[str] = "optimizer.ckpt",
        scheduler_name: Optional[str] = "scheduler.ckpt",
    ):
        """
        Load model parameters.

        Parameters
        ----------
        input_dir: model directory
        load_optimizer: whether load other trainer parameters
        load_scheduler: whether load scheduler
        model_name: model name (suffix free)
        optimizer_name: optimizer name (suffix free)
        scheduler_name: scheduler name

        Returns
        -------
        self
        """
        input_dir = input_dir if input_dir is not None else self._status.result_dir

        logger.info(f"Loading model from {input_dir}")

        self.initialize_model()
        self._model.load_state_dict(torch.load(op.join(input_dir, model_name)))
        self._model.to(self._device)

        if load_optimizer:
            logger.info("Loading optimizer")

            if self._optimizer is None:
                self.initialize_optimizer()

            if op.isfile(op.join(input_dir, optimizer_name)):
                self._optimizer.load_state_dict(
                    torch.load(
                        op.join(input_dir, optimizer_name), map_location=self._device
                    )
                )
            else:
                logger.warning("Optimizer file does not exist!")

        if load_scheduler:
            logger.info("Loading scheduler")

            if self._scheduler is None:
                self.initialize_scheduler()

            if op.isfile(op.join(input_dir, scheduler_name)):
                self._optimizer.load_state_dict(
                    torch.load(
                        op.join(input_dir, scheduler_name), map_location=self._device
                    )
                )
            else:
                logger.warning("Scheduler file does not exist!")
        return self

    def save_results(self, path, preds, variances, lbs, masks):
        """
        Save results to disk as csv files
        """

        if not self.config.disable_result_saving:
            save_results(path, preds, variances, lbs, masks)
        else:
            logger.warning(
                "Results are not saved because `disable_result_saving` flag is set to `True`."
            )

        return None
