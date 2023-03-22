from abc import ABC

import os

import numpy as np
import torch
import wandb
import logging
from tqdm.auto import tqdm
from torch.optim import Adam
from typing import Optional

from seqlbtoolkit.training.status import Status

from ..utils.macro import EVAL_METRICS
from ..base.train import Trainer as BaseTrainer
from .collate import Collator
from .model import GROVERFinetuneModel
from .args import GroverConfig
from .util.utils import load_checkpoint

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer, ABC):
    def __init__(self,
                 config: GroverConfig,
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
        self._model = load_checkpoint(self.config)

    def initialize_optimizer(self):
        self._optimizer = Adam(self._model.parameters(), lr=self.config.lr)

    def run(self):

        logger.info("Training model")
        self.train()

        test_metrics = self.test()
        logger.info("Test results:")
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

            pbar.update()

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
