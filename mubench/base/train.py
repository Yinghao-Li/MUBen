from abc import ABC

import os

import numpy as np
import torch
import wandb
import logging
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import Adam
from typing import Optional

from seqlbtoolkit.training.train import BaseTrainer
from seqlbtoolkit.training.status import Status

from ..utils.macro import EVAL_METRICS
from .metric import (
    GaussianNLL,
    calculate_classification_metrics,
    calculate_regression_metrics
)
from .collate import Collator
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
        self._optimizer = Adam(self._model.parameters(), lr=self.config.lr)

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

    def run(self):

        logger.info("Training model")
        self.train()

        test_metrics = self.test()
        self.log_results(test_metrics)

        self.save(output_dir=self._result_dir)

        wandb.finish()

        logger.info('Done.')

    def train(self):

        self._model.to(self.config.device)
        data_loader = self.get_dataloader(
            self.training_dataset,
            shuffle=True,
            batch_size=self.config.batch_size
        )

        for epoch in (tqdm_epoch := tqdm(range(self.config.n_epochs))):
            training_loss = self.training_epoch(data_loader)
            # Print the averaged training loss so far.
            tqdm_epoch.set_description(f'[Epoch {epoch}] average Loss: {training_loss:.4f}')

            wandb.log(data={'train/loss': training_loss}, step=epoch+1)

            if self.config.valid_epoch_interval and (epoch + 1) % self.config.valid_epoch_interval == 0:
                self.eval_and_save(step_idx=epoch+1, metric_name=self._valid_metric)

        return None

    def training_epoch(self, data_loader):

        avg_loss = 0.
        num_items = 0
        for batch in data_loader:
            batch.to(self.config.device)
            logits = self.model(batch)

            # modify data shapes to accommodate different tasks
            if self.config.task_type == 'classification' and self.config.binary_classification_with_softmax:
                # this works the same as logits.view(-1, n_tasks, n_lbs).view(-1, n_lbs)
                logits = logits.view(-1, self.config.n_lbs)
                batch.lbs = batch.lbs.view(-1)
                batch.masks = batch.masks.view(-1)
            if self.config.task_type == 'regression' and self.config.regression_with_variance:
                logits = logits.view(-1, self.config.n_tasks, 2)  # mean and var for the last dimension

            loss = self._loss_fn(logits, batch.lbs)
            loss = torch.sum(loss * batch.masks) / batch.masks.sum()
            loss.backward()

            self._optimizer.step()
            self._optimizer.zero_grad()

            avg_loss += loss.item() * len(batch)
            num_items += len(batch)

        return avg_loss / num_items

    def inference(self, dataset, batch_size: Optional[int] = None):

        dataloader = self.get_dataloader(
            dataset,
            batch_size=batch_size if batch_size else self.config.batch_size,
            shuffle=False
        )
        self._model.eval()

        preds = list()

        with torch.no_grad():
            for batch in dataloader:
                batch.to(self.config.device)
                logits = self.model(batch)
                preds.append(logits.detach().cpu())

        preds = torch.cat(preds, dim=0).numpy()

        return preds

    def evaluate(self, dataset, n_run: Optional[int] = 1, return_preds: Optional[bool] = False):

        if n_run == 1:

            preds = self.inference(dataset)
            metrics = self.get_metrics(dataset.lbs, preds, dataset.masks)

        else:
            preds = list()
            for i_run in (tqdm_run := tqdm(range(n_run))):
                tqdm_run.set_description(f'[Test {i_run}]')
                preds.append(self.inference(dataset))
            preds = np.stack(preds)

            metrics = self.get_metrics(dataset.lbs, preds.mean(axis=0), dataset.masks)

        return metrics if not return_preds else (metrics, preds)

    def eval_and_save(self,
                      step_idx: Optional[int] = None,
                      metric_name: Optional[str] = 'f1'):
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """

        valid_results = self.evaluate(self.valid_dataset)

        logger.info("Validation results:")
        self.log_results(valid_results)

        step_idx = self._status.eval_step + 1 if step_idx is None else step_idx

        result_dict = {f"valid/{k}": v for k, v in valid_results.items()}
        wandb.log(data=result_dict, step=step_idx)

        # ----- check model performance and update buffer -----
        if self._status.model_buffer.check_and_update(getattr(valid_results, metric_name), self.model):
            logger.info("Model buffer is updated!")

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
    def log_results(metrics):
        if isinstance(metrics, dict):
            for key, val in metrics.items():
                logger.info(f"[{key}]")
                for k, v in val.items():
                    logger.info(f"  {k}: {v:.4f}.")
        else:
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}.")

    def save(self,
             output_dir: Optional[str] = None,
             save_optimizer: Optional[bool] = False,
             save_scheduler: Optional[bool] = False,
             model_name: Optional[str] = 'model',
             optimizer_name: Optional[str] = 'optimizer',
             scheduler_name: Optional[str] = 'scheduler'):

        os.makedirs(output_dir, exist_ok=True)
        self._model.load_state_dict(self._status.model_buffer.model_state_dicts[0])
        super().save(output_dir, save_optimizer, save_scheduler, model_name, optimizer_name, scheduler_name)

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
