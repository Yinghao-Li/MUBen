"""
Yinghao Li @ Georgia Tech

Base trainer function.
"""

import os
import copy
import torch
import wandb
import logging
import numpy as np
import torch.nn as nn

from tqdm.auto import tqdm
from typing import Optional, Union, Tuple

from scipy.special import softmax, expit
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..utils.macro import EVAL_METRICS, UncertaintyMethods
from ..utils.container import ModelContainer, UpdateCriteria
from ..utils.scaler import StandardScaler
from ..utils.data import set_seed
from .metric import (
    GaussianNLL,
    calculate_classification_metrics,
    calculate_regression_metrics
)
from .dataset import Collator
from .model import DNN
from .args import Config
from .uncertainty.swag import SWAModel, update_bn
from .uncertainty.temperature_scaling import TSModel
from .uncertainty.focal_loss import FocalLoss

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 config: Config,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 collate_fn=None,
                 scalar=None):

        self._config = config
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn if collate_fn is not None else Collator(config)
        self._scalar = scalar
        self._device = getattr(config, "device", "cpu")

        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._loss_fn = None

        # Validation variables and flags
        self._valid_metric = EVAL_METRICS[config.dataset_name].lower().replace('-', '_')
        if config.valid_epoch_interval == 0:
            update_criteria = UpdateCriteria.always
        elif config.task_type == 'classification':
            update_criteria = UpdateCriteria.metric_larger
        else:
            update_criteria = UpdateCriteria.metric_smaller
        self._model_container = ModelContainer(update_criteria)

        self._model_name = 'model_best.ckpt'

        # Test variables and flags
        self._result_dir = os.path.join(
            config.result_dir, config.dataset_name, config.model_name, config.uncertainty_method
        )

        # mutable class attributes
        self._train_log_idx_ = 0  # will increase by 1 each time you call `train_epoch`
        self._eval_log_idx_ = 0  # will increase by 1 each time you call `eval_and_save`
        self._lr_ = config.lr
        self._lr_scheduler_type_ = config.lr_scheduler_type
        self._n_epochs_ = config.n_epochs
        self._valid_epoch_interval_ = config.valid_epoch_interval
        self._model_name_ = self._model_name  # mutable model name for ensemble
        self._result_dir_ = self._result_dir

        # uncertainty-specific variables
        self._swa_model = None  # SWAG
        self._ts_model = None  # Temperature Scaling

        # flags
        self._model_frozen = False

        # normalize training dataset labels for regression task
        self.standardize_training_lbs()

        # initialize training modules
        self.initialize()

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        if self._ts_model:
            return self._ts_model
        return self._model

    def initialize(self):
        self.initialize_model()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss()
        self._train_log_idx_ = 0  # will increase by 1 each time you call `train_epoch`
        self._eval_log_idx_ = 0  # will increase by 1 each time you call `eval_and_save`
        return self

    def initialize_model(self, *args, **kwargs):
        self._model = DNN(
            d_feature=self.config.d_feature,
            n_lbs=self.config.n_lbs,
            n_tasks=self.config.n_tasks,
            n_hidden_layers=self.config.n_dnn_hidden_layers,
            d_hidden=self.config.d_dnn_hidden,
            p_dropout=self.config.dropout,
        )

    def initialize_optimizer(self, *args, **kwargs):
        """
        Initialize model optimizer
        """
        self._optimizer = AdamW(self.model.parameters(), lr=self._lr_)
        return self

    def initialize_scheduler(self, *args, **kwargs):
        """
        Initialize learning rate scheduler
        """
        num_update_steps_per_epoch = int(np.ceil(
            len(self._training_dataset) / self.config.batch_size
        ))
        num_warmup_steps = int(np.ceil(
            num_update_steps_per_epoch * self.config.warmup_ratio * self._n_epochs_
        ))
        num_training_steps = int(np.ceil(num_update_steps_per_epoch * self._n_epochs_))

        self._scheduler = get_scheduler(
            self._lr_scheduler_type_,
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self

    def initialize_loss(self, ignore_focal_loss=False):

        # Notice that the reduction should always be 'none' here to facilitate
        # the following masking operation
        if self.config.task_type == 'classification':
            # for compatibility with focal loss
            if self.config.uncertainty_method == UncertaintyMethods.focal and not ignore_focal_loss:
                self._loss_fn = FocalLoss()
            elif self.config.binary_classification_with_softmax:
                self._loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self._loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        else:
            if self.config.regression_with_variance:
                self._loss_fn = GaussianNLL(reduction='none')
            else:
                self._loss_fn = nn.MSELoss(reduction='none')

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

        TODO: check whether this function behaves properly on multi-task regression
        """
        if self.config.task_type == 'classification':
            return self
        if self._training_dataset is None and self._scalar is None:
            logger.warning("Encounter regression task with no training dataset specified and label scaling disabled! "
                           "This may create inconsistency between training and inference label scales.")
            return self

        lbs = copy.deepcopy(self._training_dataset.lbs)
        lbs[~self._training_dataset.masks.astype(bool)] = np.nan
        self._scalar = StandardScaler(replace_nan_token=0).fit(lbs)

        self._training_dataset.update_lbs(self._scalar.transform(self._training_dataset.lbs))

        return self

    @property
    def n_training_steps(self):
        num_update_steps_per_epoch = int(np.ceil(len(self._training_dataset) / self.config.batch_size))
        return num_update_steps_per_epoch * self._n_epochs_

    @property
    def n_valid_steps(self):
        num_update_steps_per_epoch = int(np.ceil(len(self._valid_dataset) / self.config.batch_size))
        return num_update_steps_per_epoch * self._n_epochs_

    def train_mode(self):
        """
        Set the PyTorch model to train mode
        """
        self.model.train()
        return self

    def eval_mode(self):
        """
        Set the PyTorch model to evaluation mode
        """

        self.model.eval()

        if self.config.uncertainty_method == UncertaintyMethods.mc_dropout:
            # activate the dropout layers during evaluation for MC Dropout
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

        return self

    def set_mode(self, mode: str):
        assert mode in ('train', 'eval'), ValueError(f"Argument `mode` should be 'train' or 'eval'.")
        if mode == 'train':
            self.train_mode()
        else:
            self.eval_mode()
        return self

    def run(self):

        # deep ensembles
        if self.config.uncertainty_method == UncertaintyMethods.ensembles:
            self.run_ensembles()
        # swag
        elif self.config.uncertainty_method == UncertaintyMethods.swag:
            self.run_swag()
        # temperature scaling
        elif self.config.uncertainty_method == UncertaintyMethods.temperature:
            self.run_temperature_scaling()
        # focal loss
        elif self.config.uncertainty_method == UncertaintyMethods.focal:
            self.run_focal_loss()
        # none & MC Dropout
        else:
            self.run_single_shot()

        wandb.finish()

        logger.info('Done.')

    def run_single_shot(self, apply_test=True):

        set_seed(self.config.seed)
        if os.path.exists(os.path.join(self._result_dir_, self._model_name_)) and not self.config.retrain_model:
            logger.info("Find existing model, will skip training.")
            self.load_best_model()
        else:
            logger.info("Training model")
            self.train()

        if apply_test:
            test_metrics = self.test()
            logger.info("Test results:")
            self.log_results(test_metrics)

        self.save_best_model()

        return self

    def run_ensembles(self):

        for ensemble_idx in range(self.config.n_ensembles):
            # update random seed and re-initialize training status
            individual_seed = self.config.seed + ensemble_idx
            set_seed(individual_seed)
            self.initialize()
            logger.info(f"[Ensemble {ensemble_idx}] seed: {individual_seed}")

            # update the name of the best model
            self._result_dir_ = os.path.join(self._result_dir, str(ensemble_idx))

            if os.path.exists(os.path.join(self._result_dir_, self._model_name_)) and \
                    not self.config.retrain_model:
                logger.info("Find existing model, will skip training.")
                self.load_best_model()
            else:
                logger.info("Training model")
                self.train()

            test_metrics = self.test()
            logger.info("Test results:")
            self.log_results(test_metrics)

            self.save_best_model()

        return self

    def run_swag(self):
        # Train the model first. Do not need to load state dict as it is done during test
        self.run_single_shot(apply_test=False)

        logger.info("SWA session start")

        # update hyper parameters
        self._lr_ *= self.config.swa_lr_decay
        self._lr_scheduler_type_ = 'constant'
        self._n_epochs_ = self.config.n_swa_epochs
        self._valid_epoch_interval_ = 0  # Can also set this to None

        self.initialize_optimizer()
        self.initialize_scheduler(use_default=True)  # the argument is for GROVER compatibility

        self._model.to(self._device)
        self._swa_model = SWAModel(
            model=self.model,
            k_models=self.config.k_swa_checkpoints,
            device=self._device
        )

        logger.info("Training model")
        self.train()

        test_metrics = self.test(load_best_model=False)
        logger.info("Test results:")
        self.log_results(test_metrics)

        return self

    def run_temperature_scaling(self):
        # Train the model first. Do not need to load state dict as it is done during test
        self.run_single_shot(apply_test=False)

        logger.info("Temperature Scaling session start.")

        # update hyper parameters
        self._lr_ = self.config.ts_lr
        self._lr_scheduler_type_ = 'constant'
        self._n_epochs_ = self.config.n_ts_epochs
        self._valid_epoch_interval_ = 0  # Can also set this to None

        self.model.to(self._device)
        self.freeze()
        self._ts_model = TSModel(self._model)

        self.initialize_optimizer()
        self.initialize_scheduler(use_default=True)  # the argument is for GROVER compatibility

        logger.info("Training model on validation")
        self.train(use_valid_dataset=True)

        test_metrics = self.test(load_best_model=False)
        logger.info("Test results:")
        self.log_results(test_metrics)

        self.unfreeze()

        return self

    def run_focal_loss(self):
        # Train the model first. Do not need to load state dict as it is done during test
        self.run_single_shot()

        if self.config.apply_temperature_scaling_after_focal_loss:
            logger.info("[Focal Loss] Temperature Scaling session start.")

            # update hyper parameters
            self._lr_ = self.config.ts_lr
            self._lr_scheduler_type_ = 'constant'
            self._n_epochs_ = self.config.n_ts_epochs
            self._valid_epoch_interval_ = 0  # Can also set this to None

            self.model.to(self._device)
            self.freeze()
            self._ts_model = TSModel(self._model)

            self.initialize_optimizer()
            self.initialize_scheduler(use_default=True)  # the argument is for GROVER compatibility
            self.initialize_loss(ignore_focal_loss=True)  # re-initialize the loss to CE as described in the paper

            logger.info("Training model on validation")
            self.train(use_valid_dataset=True)

            test_metrics = self.test(load_best_model=False)
            logger.info("Test results:")
            self.log_results(test_metrics)

            self.unfreeze()

        return self

    def train(self, use_valid_dataset=False):

        self.model.to(self._device)
        data_loader = self.get_dataloader(
            self.training_dataset if not use_valid_dataset else self.valid_dataset,
            shuffle=True,
            batch_size=self.config.batch_size
        )

        with tqdm(total=self.n_training_steps if not use_valid_dataset else self.n_valid_steps) as pbar:
            pbar.set_description(f'[Epoch 0] Loss: {np.inf:.4f}')

            for epoch_idx in range(self._n_epochs_):

                training_loss = self.training_epoch(data_loader, pbar)

                # Print the averaged training loss so far.
                pbar.set_description(f'[Epoch {self._train_log_idx_ + 1}] Loss: {training_loss:.4f}')
                wandb.log(data={'train/loss': training_loss}, step=self._train_log_idx_ + 1)

                # Compatibility with SWAG
                if self._swa_model:
                    self._swa_model.update_parameters(self.model)

                if self._valid_epoch_interval_ and (epoch_idx + 1) % self._valid_epoch_interval_ == 0:
                    self.eval_and_save()

        return None

    def training_epoch(self, data_loader, pbar):

        self.train_mode()
        # Set the base model to evaluation mode for Temperature Scaling training
        if self._ts_model:
            self._model.eval()

        avg_loss = 0.
        num_items = 0
        for batch in data_loader:
            batch.to(self.config.device)

            self._optimizer.zero_grad()

            # mixed-precision training
            with torch.autocast(device_type=self.config.device_str, dtype=torch.bfloat16):
                logits = self.model(batch)
                loss = self.get_loss(logits, batch)

            loss.backward()

            if self.config.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)

            self._optimizer.step()
            self._scheduler.step()

            avg_loss += loss.item() * len(batch)
            num_items += len(batch)

            pbar.update()

        self._train_log_idx_ += 1

        return avg_loss / num_items

    def get_loss(self, logits, batch) -> torch.Tensor:
        """
        Children trainers can directly reload this function instead of
        reloading `training epoch`, which could be more complicated

        Parameters
        ----------
        logits: logits predicted by the model
        batch: batched training data

        Returns
        -------
        loss, torch.Tensor
        """

        # modify data shapes to accommodate different tasks
        lbs, masks = batch.lbs, batch.masks  # so that we don't mess up batch instances
        if self.config.task_type == 'classification' and self.config.binary_classification_with_softmax:
            # this works the same as logits.view(-1, n_tasks, n_lbs).view(-1, n_lbs)
            logits = logits.view(-1, self.config.n_lbs)
            lbs = lbs.view(-1)
            masks = masks.view(-1)
        if self.config.task_type == 'regression' and self.config.regression_with_variance:
            logits = logits.view(-1, self.config.n_tasks, 2)  # mean and var for the last dimension

        loss = self._loss_fn(logits, lbs)
        loss = torch.sum(loss * masks) / masks.sum()
        return loss

    def inference(self, dataset, batch_size: Optional[int] = 0):

        dataloader = self.get_dataloader(dataset, batch_size=batch_size, shuffle=False)
        self.eval_mode()

        logits_list = list()

        with torch.no_grad():
            for batch in dataloader:
                batch.to(self.config.device)

                with torch.autocast(device_type=self.config.device_str, dtype=torch.bfloat16):
                    logits = self.model(batch)
                logits_list.append(logits.to(torch.float).detach().cpu())

        logits = torch.cat(logits_list, dim=0).numpy()

        return logits

    def normalize_logits(self, logits: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if self.config.task_type == 'classification':
            if len(logits.shape) > 1 and logits.shape[-1] >= 2:
                preds = softmax(logits, axis=-1)
            else:
                preds = expit(logits)  # sigmoid function
        else:
            # get the mean of the preds
            if self.config.regression_with_variance:
                mean = logits[..., 0]
                var = logits[..., 1]
                return mean, var
            else:
                preds = logits

        return preds

    def inverse_standardize_preds(
            self, preds: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self._scalar is None:
            return preds
        if isinstance(preds, np.ndarray):
            return self._scalar.inverse_transform(preds)
        elif isinstance(preds, tuple):
            mean, var = self._scalar.inverse_transform(*preds)
            return mean, var
        else:
            raise TypeError(f"Unsupported prediction type: {type(preds)}!")

    def evaluate(self, dataset, n_run: Optional[int] = 1, return_preds: Optional[bool] = False):

        if n_run == 1:
            preds = self.inverse_standardize_preds(self.normalize_logits(self.inference(dataset)))
            if isinstance(preds, np.ndarray):
                metrics = self.get_metrics(dataset.lbs, preds, dataset.masks)
            else:  # isinstance(preds, tuple)
                metrics = self.get_metrics(dataset.lbs, preds[0], dataset.masks)

            return metrics if not return_preds else (metrics, preds)

        # Multiple test runs
        preds_list = list()
        vars_list = list()
        for test_run_idx in (pbar := tqdm(range(n_run))):
            pbar.set_description(f'[Test {test_run_idx}]')

            individual_seed = self.config.seed + test_run_idx
            set_seed(individual_seed)

            if self._swa_model:
                self._model = self._swa_model.sample_parameters()
                update_bn(
                    model=self.model,
                    training_loader=self.get_dataloader(self.training_dataset, shuffle=True),
                    device=self.config.device_str
                )

            preds = self.inverse_standardize_preds(self.normalize_logits(self.inference(dataset)))
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
        elif not vars_array:
            return metrics, preds_array
        else:
            return metrics, (preds_array, vars_array)

    def eval_and_save(self):
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """
        self._eval_log_idx_ += 1

        valid_results = self.evaluate(self.valid_dataset)

        result_dict = {f"valid/{k}": v for k, v in valid_results.items()}
        wandb.log(data=result_dict, step=self._eval_log_idx_)

        logger.debug(f"[Valid step {self._eval_log_idx_}] results:")
        self.log_results(valid_results, logging_func=logger.debug)

        # ----- check model performance and update buffer -----
        if self._model_container.check_and_update(self.model, valid_results[self._valid_metric]):
            logger.debug("Model buffer is updated!")

        return None

    def test(self, load_best_model=True):

        if load_best_model and self._model_container.state_dict:
            self._model.load_state_dict(self._model_container.state_dict)

        metrics, preds = self.evaluate(self._test_dataset, n_run=self.config.n_test, return_preds=True)

        # save preds
        if self.config.n_test == 1:
            if isinstance(preds, np.ndarray):
                preds = preds.reshape(1, *preds.shape)
            else:
                preds = tuple([p.reshape(1, *p.shape) for p in preds])

        for idx, pred in enumerate(preds):
            file_path = os.path.join(self._result_dir_, "preds", f"{idx}.pt")
            self.save_preds_to_pt(
                lbs=self._test_dataset.lbs, preds=preds, masks=self.test_dataset.masks, file_path=file_path
            )

        return metrics

    def get_metrics(self, lbs, preds, masks):
        if masks.shape[-1] == 1 and len(masks.shape) > 1:
            masks = masks.squeeze(-1)
        bool_masks = masks.astype(bool)

        if lbs.shape[-1] == 1 and len(lbs.shape) > 1:
            lbs = lbs.squeeze(-1)
        lbs = lbs[bool_masks]

        if self.config.n_tasks > 1:
            preds = preds.reshape(-1, self.config.n_tasks, self.config.n_lbs)
        if preds.shape[-1] == 1 and len(preds.shape) > 1:
            preds = preds.squeeze(-1)

        preds = preds[bool_masks]

        if self.config.task_type == 'classification':
            metrics = calculate_classification_metrics(lbs, preds, self._valid_metric)
        else:
            metrics = calculate_regression_metrics(lbs, preds, self._valid_metric)

        return metrics

    @staticmethod
    def log_results(metrics, logging_func=logger.info):
        """
        Print evaluation metrics to the logging destination
        """
        for k, v in metrics.items():
            logging_func(f"  {k}: {v:.4f}.")
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

    def save_best_model(self):

        os.makedirs(self._result_dir_, exist_ok=True)
        self._model_container.save(os.path.join(self._result_dir_, self._model_name_))

        return self

    def load_best_model(self):
        self._model_container.load(os.path.join(self._result_dir_, self._model_name_))
        self._model.load_state_dict(self._model_container.state_dict)

        return self

    def get_dataloader(self,
                       dataset,
                       shuffle: Optional[bool] = False,
                       batch_size: Optional[int] = 0):
        try:
            dataloader = DataLoader(
                dataset=dataset,
                collate_fn=self._collate_fn,
                batch_size=batch_size if batch_size else self._config.batch_size,
                num_workers=getattr(self._config, "num_workers", 0),
                pin_memory=getattr(self._config, "pin_memory", False),
                shuffle=shuffle,
                drop_last=False
            )
        except Exception as e:
            logger.exception(e)
            raise e

        return dataloader

    def save(self,
             output_dir: Optional[str] = None,
             save_optimizer: Optional[bool] = False,
             save_scheduler: Optional[bool] = False,
             model_name: Optional[str] = 'model.ckpt',
             optimizer_name: Optional[str] = 'optimizer.ckpt',
             scheduler_name: Optional[str] = 'scheduler.ckpt'):
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
        output_dir = output_dir if output_dir is not None else self._result_dir_

        model_state_dict = self._model.state_dict()
        torch.save(model_state_dict, os.path.join(output_dir, model_name))

        self._config.save(output_dir)

        if save_optimizer:
            torch.save(self._optimizer.state_dict(), os.path.join(output_dir, optimizer_name))
        if save_scheduler and self._scheduler is not None:
            torch.save(self._scheduler.state_dict(), os.path.join(output_dir, scheduler_name))

        return None

    def load(self,
             input_dir: Optional[str] = None,
             load_optimizer: Optional[bool] = False,
             load_scheduler: Optional[bool] = False,
             model_name: Optional[str] = 'model.ckpt',
             optimizer_name: Optional[str] = 'optimizer.ckpt',
             scheduler_name: Optional[str] = 'scheduler.ckpt'):
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
        input_dir = input_dir if input_dir is not None else self._result_dir_

        logger.info(f"Loading model from {input_dir}")

        self.initialize_model()
        self._model.load_state_dict(torch.load(os.path.join(input_dir, model_name)))
        self._model.to(self._device)

        if load_optimizer:
            logger.info("Loading optimizer")

            if self._optimizer is None:
                self.initialize_optimizer()

            if os.path.isfile(os.path.join(input_dir, optimizer_name)):
                self._optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, optimizer_name), map_location=self._device)
                )
            else:
                logger.warning("Optimizer file does not exist!")

        if load_scheduler:
            logger.info("Loading scheduler")

            if self._scheduler is None:
                self.initialize_scheduler()

            if os.path.isfile(os.path.join(input_dir, scheduler_name)):
                self._optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, scheduler_name), map_location=self._device)
                )
            else:
                logger.warning("Scheduler file does not exist!")
        return self

    @staticmethod
    def save_preds_to_pt(lbs, preds, masks, file_path: str):
        """
        Save results to disk as csv files
        """

        if not file_path.endswith('.pt'):
            file_path = f"{file_path}.pt"

        data_dict = {
            "lbs": lbs,
            "preds": preds,
            "masks": masks
        }

        os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)
        torch.save(data_dict, file_path)
