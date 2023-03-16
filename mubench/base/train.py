from abc import ABC

import os

import numpy as np
import pandas as pd
import torch
import wandb
import logging
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import Adam
from typing import Optional

from seqlbtoolkit.training.train import BaseTrainer
from seqlbtoolkit.training.status import Status

from .metric import get_classification_metrics, get_regression_metrics
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
            collate_fn = Collator(task=config.task_type)

        super().__init__(
            config=config,
            training_dataset=training_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            collate_fn=collate_fn
        )

        self.initialize()

        self._valid_metric = 'roc_auc' if config.task_type == 'classification' else 'mae'
        self._status = Status(metric_smaller_is_better=True if config.task_type == 'regression' else False)

    def initialize_model(self):
        self._model = DNN(
            d_feature=self.config.d_feature,
            n_lbs=self.config.n_lbs,
            n_hidden_layers=self.config.n_hidden_layers,
            d_hidden=self.config.d_hidden,
            p_dropout=self.config.dropout,
        )

    def initialize_optimizer(self):
        self._optimizer = Adam(self._model.parameters(), lr=self.config.lr)

    def initialize_loss(self):

        # classification task
        if self.config.n_lbs > 1:
            self._loss_fn = nn.CrossEntropyLoss(
                weight=getattr(self.config, "class_weights", None)
            )

        # regression task
        else:
            self._loss_fn = nn.MSELoss()

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

        wandb.finish()

        logger.info('Done.')

    def train(self):

        os.makedirs(self.config.output_dir, exist_ok=True)

        self._model.to(self.config.device)
        data_loader = self.get_dataloader(
            self.training_dataset,
            shuffle=True,
            batch_size=self.config.batch_size
        )

        for epoch in (tqdm_epoch := tqdm(range(self.config.n_epochs))):
            training_loss = self.training_step(data_loader)
            # Print the averaged training loss so far.
            tqdm_epoch.set_description(f'[Epoch {epoch}] average Loss: {training_loss:4f}')

            wandb.log(data={'train/loss': training_loss}, step=epoch+1)

            # if (epoch + 1) % self.config.valid_interval == 0:
            #     self.eval_and_save(step_idx=epoch+1, metric_name=self._valid_metric)

        return None

    def training_step(self, data_loader):

        avg_loss = 0.
        num_items = 0
        for batch in data_loader:
            batch.to(self.config.device)

            logits = self.model(batch)
            loss = self._loss_fn(logits, batch.lbs)
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

                if self.config.task == 'classification':
                    pred_y = torch.softmax(logits, dim=-1).detach().cpu()
                else:
                    pred_y = logits.detach().cpu()

                preds.append(pred_y)

        preds = torch.cat(preds, dim=0).numpy()

        return preds

    def evaluate(self, dataset, n_run: Optional[int] = 1, save_preds: Optional[bool] = False):

        if n_run == 1:
            preds = self.inference(dataset)
        else:
            preds = list()
            for i_run in (tqdm_run := tqdm(range(n_run))):
                tqdm_run.set_description(f'[Test {i_run}]')
                preds.append(self.inference(dataset))
            preds = np.stack(preds)

            if save_preds:
                for idx, pred in enumerate(preds):
                    file_path = os.path.join(self.config.output_dir, "preds", f"{idx}.csv")
                    self.save_preds(lbs=dataset.lbs, preds=pred, file_path=file_path)

        if self.config.task == 'classification':
            metrics = get_classification_metrics(dataset.lbs, preds.mean(axis=0))
        elif self.config.task == 'regression':
            metrics = get_regression_metrics(dataset.lbs, preds.mean(axis=0))
        else:
            raise ValueError("Undefined training task!")

        return metrics

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

        if self._status.model_buffer.size == 1:
            if self._status.model_buffer.model_state_dicts:
                self._model.load_state_dict(self._status.model_buffer.model_state_dicts[0])
            metrics = self.evaluate(self._test_dataset, n_run=self.config.n_test, save_preds=True)
            return metrics

        raise NotImplementedError("Function for multi-checkpoint caching & evaluation is not implemented!")

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

    @staticmethod
    def save_preds(lbs, preds: np.ndarray, file_path: str):
        """
        Save results to disk as csv files
        """

        data_dict = dict()

        data_dict['true'] = lbs

        assert len(preds.shape) < 3, ValueError("Cannot save results with larger ")
        if len(preds.shape) == 2:
            preds = [pred_tuple.tolist() for pred_tuple in preds]
        data_dict['pred'] = preds

        os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)

        df = pd.DataFrame(data_dict)
        df.to_csv(file_path)
