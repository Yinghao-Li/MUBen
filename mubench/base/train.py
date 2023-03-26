from abc import ABC

import os

import numpy as np
import torch
import wandb
import logging
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from typing import Optional
from functools import cached_property
from scipy.special import softmax, expit

from seqlbtoolkit.training.train import BaseTrainer
from seqlbtoolkit.training.status import Status

from ..utils.macro import EVAL_METRICS
from .metric import (
    GaussianNLL,
    calculate_classification_metrics,
    calculate_regression_metrics
)
from .dataset import Collator
from .model import DNN
from .args import Config

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    def __init__(self,
                 config: Config,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 collate_fn=None):

        if not collate_fn:
            collate_fn = Collator(config)

        super().__init__(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn
        )

        self.initialize()

        self._valid_metric = EVAL_METRICS[config.dataset_name].lower().replace('-', '_')
        self._status = Status(metric_smaller_is_better=True if config.task_type == 'regression' else False)
        self._result_dir = os.path.join(
            self.config.result_dir,
            self.config.dataset_name,
            self.config.model_name,
            self.config.uncertainty_method,
        )

        # constants
        self._best_model_name = 'model_best.bin'

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
        self._optimizer = Adam(self._model.parameters(), lr=self.config.lr)
        return self

    def initialize_scheduler(self):
        """
        Initialize learning rate scheduler
        """
        # Notice that this scheduler does not change the lr!
        # This implementation is for the compatibility with other models that are trained with functional schedulers
        self._scheduler = StepLR(optimizer=self._optimizer, step_size=int(1e9), gamma=1)
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

    @cached_property
    def n_training_steps(self):
        num_update_steps_per_epoch = int(np.ceil(len(self._training_dataset) / self.config.batch_size))
        return num_update_steps_per_epoch * self.config.n_epochs

    def run(self):

        # TODO: more appropriate continue training setup
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
                    self.eval_and_save(step_idx=epoch_idx+1, metric_name=self._valid_metric)

        return None

    def training_epoch(self, data_loader, pbar):

        avg_loss = 0.
        num_items = 0
        for batch in data_loader:
            batch.to(self.config.device)

            self._optimizer.zero_grad()
            logits = self.model(batch)

            loss = self.get_loss(logits, batch)
            loss.backward()

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
                logits = self.model(batch)
                logits_list.append(logits.detach().cpu())

        logits = torch.cat(logits_list, dim=0).numpy()

        return logits

    def evaluate(self, dataset, n_run: Optional[int] = 1, return_preds: Optional[bool] = False):

        if n_run == 1:

            preds = self.normalize_logits(self.inference(dataset))
            metrics = self.get_metrics(dataset.lbs, preds, dataset.masks)

        else:
            preds = list()
            for i_run in (tqdm_run := tqdm(range(n_run))):
                tqdm_run.set_description(f'[Test {i_run}]')
                preds.append(self.normalize_logits(self.inference(dataset)))
            preds = np.stack(preds)
            metrics = self.get_metrics(dataset.lbs, preds.mean(axis=0), dataset.masks)

        return metrics if not return_preds else (metrics, preds)

    def normalize_logits(self, logits):

        if self.config.task_type == 'classification':
            if len(logits.shape) > 1 and logits.shape[-1] >= 2:
                preds = softmax(logits, axis=-1)
            else:
                preds = expit(logits)  # sigmoid function
        else:
            preds = logits if logits.shape[-1] == 1 or len(logits.shape) == 1 else logits[..., 0]
        return preds

    def eval_and_save(self,
                      step_idx: Optional[int] = None,
                      metric_name: Optional[str] = 'f1'):
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """

        valid_results = self.evaluate(self.valid_dataset)

        step_idx = self._status.eval_step + 1 if step_idx is None else step_idx

        result_dict = {f"valid/{k}": v for k, v in valid_results.items()}
        wandb.log(data=result_dict, step=step_idx)

        logger.debug(f"[Valid step {step_idx}] results:")
        self.log_results(valid_results, logging_func=logger.debug)

        # ----- check model performance and update buffer -----
        if self._status.model_buffer.check_and_update(getattr(valid_results, metric_name), self.model):
            logger.debug("Model buffer is updated!")

        self._status.eval_step = step_idx

        return None

    def test(self):

        assert self._status.model_buffer.size == 1, \
            NotImplementedError("Function for multi-checkpoint caching & evaluation is not implemented!")

        if self._status.model_buffer.model_state_dicts:
            self._model.load_state_dict(self._status.model_buffer.model_state_dicts[0])
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
        self._status.model_buffer.save(os.path.join(output_dir, self._best_model_name))

        return self

    def load_best_model(self, model_dir):
        self._status.model_buffer.load(os.path.join(model_dir, self._best_model_name))

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
