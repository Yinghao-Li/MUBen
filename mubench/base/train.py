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
from typing import Optional
from functools import cached_property

from scipy.special import softmax, expit
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..utils.macro import EVAL_METRICS, UncertaintyMethods
from ..utils.container import ModelContainer, UpdateCriteria
from ..utils.scaler import StandardScaler
from .metric import (
    GaussianNLL,
    calculate_classification_metrics,
    calculate_regression_metrics
)
from .dataset import Collator
from .model import DNN
from .args import Config

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
        self._eval_step = 0  # will increase by 1 each time you call `eval_and_save`

        self._best_model_name = 'model_best.ckpt'

        # Test variables and flags
        self._result_dir = os.path.join(
            self.config.result_dir,
            self.config.dataset_name,
            self.config.model_name,
            self.config.uncertainty_method,
        )

        # normalize training dataset labels for regression task
        self.check_and_update_training_dataset()

        # initialize training modules
        self.initialize()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        self._config = x

    @property
    def model(self):
        return self._model

    def initialize(self):
        self.initialize_model()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss()
        return self

    def initialize_model(self):
        self._model = DNN(
            d_feature=self.config.d_feature,
            n_lbs=self.config.n_lbs,
            n_tasks=self.config.n_tasks,
            n_hidden_layers=self.config.n_dnn_hidden_layers,
            d_hidden=self.config.d_dnn_hidden,
            p_dropout=self.config.dropout,
        )

    def initialize_optimizer(self):
        """
        Initialize model optimizer
        """
        self._optimizer = AdamW(self._model.parameters(), lr=self.config.lr)
        return self

    def initialize_scheduler(self):
        """
        Initialize learning rate scheduler
        """
        num_update_steps_per_epoch = int(np.ceil(
            len(self._training_dataset) / self.config.batch_size
        ))
        num_warmup_steps = int(np.ceil(
            num_update_steps_per_epoch * self.config.warmup_ratio * self.config.n_epochs
        ))
        num_training_steps = int(np.ceil(num_update_steps_per_epoch * self.config.n_epochs))

        self._scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self

    def initialize_loss(self):

        # Notice that the reduction should always be 'none' here to facilitate
        # the following masking operation
        if self.config.task_type == 'classification':
            if self.config.binary_classification_with_softmax:
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

    def check_and_update_training_dataset(self):
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

    @cached_property
    def n_training_steps(self):
        num_update_steps_per_epoch = int(np.ceil(len(self._training_dataset) / self.config.batch_size))
        return num_update_steps_per_epoch * self.config.n_epochs

    def run(self):

        if self.config.uncertainty_method == UncertaintyMethods.none:

            if os.path.exists(os.path.join(self._result_dir, self._best_model_name)) and not self.config.retrain_model:
                logger.info("Find existing model, will skip training.")
                self.load_best_model(model_dir=self._result_dir)
            else:
                logger.info("Training model")
                self.train()

            test_metrics = self.test()
            logger.info("Test results:")
            self.log_results(test_metrics)

            self.save_best_model(output_dir=self._result_dir)

        elif self.config.uncertainty_method == UncertaintyMethods.ensembles:

            if os.path.exists(os.path.join(self._result_dir, self._best_model_name)) and not self.config.retrain_model:
                logger.info("Find existing model, will skip training.")
                self.load_best_model(model_dir=self._result_dir)
            else:
                logger.info("Training model")
                self.train()

            test_metrics = self.test()
            logger.info("Test results:")
            self.log_results(test_metrics)

            self.save_best_model(output_dir=self._result_dir)

        wandb.finish()

        logger.info('Done.')

    def train(self):

        self._model.to(self.config.device)
        data_loader = self.get_dataloader(
            self.training_dataset,
            shuffle=True,
            batch_size=self.config.batch_size
        )

        with tqdm(total=self.n_training_steps) as pbar:
            pbar.set_description(f'[Epoch 0] Loss: {np.inf:.4f}')
            for epoch_idx in range(self.config.n_epochs):
                training_loss = self.training_epoch(data_loader, pbar)
                # Print the averaged training loss so far.
                pbar.set_description(f'[Epoch {epoch_idx+1}] Loss: {training_loss:.4f}')

                wandb.log(data={'train/loss': training_loss}, step=epoch_idx+1)

                if self.config.valid_epoch_interval and (epoch_idx + 1) % self.config.valid_epoch_interval == 0:
                    self.eval_and_save()

        return None

    def training_epoch(self, data_loader, pbar):

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
                nn.utils.clip_grad_norm_(self._model.parameters(), self.config.grad_norm)

            self._optimizer.step()
            self._scheduler.step()

            avg_loss += loss.item() * len(batch)
            num_items += len(batch)

            pbar.update()

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

    def inference(self, dataset, batch_size: Optional[int] = None):

        dataloader = self.get_dataloader(
            dataset,
            batch_size=batch_size if batch_size else self.config.batch_size,
            shuffle=False
        )
        self._model.eval()

        logits_list = list()

        with torch.no_grad():
            for batch in dataloader:
                batch.to(self.config.device)

                with torch.autocast(device_type=self.config.device_str, dtype=torch.bfloat16):
                    logits = self.model(batch)
                logits_list.append(logits.to(torch.float).detach().cpu())

        logits = torch.cat(logits_list, dim=0).numpy()

        return logits

    def normalize_logits(self, logits: np.ndarray) -> np.ndarray:

        if self.config.task_type == 'classification':
            if len(logits.shape) > 1 and logits.shape[-1] >= 2:
                preds = softmax(logits, axis=-1)
            else:
                preds = expit(logits)  # sigmoid function
        else:
            # get the mean of the preds
            preds = logits if logits.shape[-1] == 1 or len(logits.shape) == 1 else logits[..., 0]

        return preds

    def scale_back_lbs(self, preds: np.ndarray) -> np.ndarray:
        if self._scalar is None:
            return preds
        preds = self._scalar.inverse_transform(preds)
        return preds

    def evaluate(self, dataset, n_run: Optional[int] = 1, return_preds: Optional[bool] = False):

        if n_run == 1:

            preds = self.scale_back_lbs(self.normalize_logits(self.inference(dataset)))
            metrics = self.get_metrics(dataset.lbs, preds, dataset.masks)

        else:
            preds = list()
            for i_run in (tqdm_run := tqdm(range(n_run))):
                tqdm_run.set_description(f'[Test {i_run}]')
                preds.append(self.scale_back_lbs(self.normalize_logits(self.inference(dataset))))
            preds = np.stack(preds)
            metrics = self.get_metrics(dataset.lbs, preds.mean(axis=0), dataset.masks)

        return metrics if not return_preds else (metrics, preds)

    def eval_and_save(self):
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """
        self._eval_step += 1

        valid_results = self.evaluate(self.valid_dataset)

        result_dict = {f"valid/{k}": v for k, v in valid_results.items()}
        wandb.log(data=result_dict, step=self._eval_step)

        logger.debug(f"[Valid step {self._eval_step}] results:")
        self.log_results(valid_results, logging_func=logger.debug)

        # ----- check model performance and update buffer -----
        if self._model_container.check_and_update(self.model, getattr(valid_results, self._valid_metric)):
            logger.debug("Model buffer is updated!")

        return None

    def test(self):

        if self._model_container.state_dict:
            self._model.load_state_dict(self._model_container.state_dict)
        metrics, preds = self.evaluate(self._test_dataset, n_run=self.config.n_test, return_preds=True)

        # save preds
        if self.config.n_test == 1:
            preds = [preds]

        for idx, pred in enumerate(preds):
            file_path = os.path.join(self._result_dir, "preds", f"{idx}.pt")
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

        if isinstance(metrics, dict):
            for key, val in metrics.items():
                logging_func(f"[{key}]")
                for k, v in val.items():
                    logging_func(f"  {k}: {v:.4f}.")
        else:
            for k, v in metrics.items():
                logging_func(f"  {k}: {v:.4f}.")

    def save_best_model(self, output_dir: Optional[str] = None):

        os.makedirs(output_dir, exist_ok=True)

        output_dir = output_dir if output_dir is not None else getattr(self._config, 'output_dir', 'output')
        self._model_container.save(os.path.join(output_dir, self._best_model_name))

        return self

    def load_best_model(self, model_dir):
        self._model_container.load(os.path.join(model_dir, self._best_model_name))

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
        output_dir = output_dir if output_dir is not None else self._result_dir

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
        input_dir = input_dir if input_dir is not None else self._result_dir

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
