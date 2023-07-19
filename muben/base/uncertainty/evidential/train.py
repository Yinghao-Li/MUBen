import os
import logging
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional, List
from scipy.stats.stats import pearsonr

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from seqlbtoolkit.training.eval import Metric
from seqlbtoolkit.training.train import BaseTrainer

from ..general.dataset import MoleculeDataset
from ..general.metrics import rmse as rmse_func
from .args import EvidentialConfig
from .model import EvidentialRegressionModel
from .loss import EvidentialRegressionLoss

logger = logging.getLogger(__name__)


class EvidentialRegressionMetrics(Metric):
    def __init__(self, nll_loss=None, reg_loss=None, rmse=None, corr=None):
        super().__init__()
        self.remove_attrs()
        self.nll_loss = nll_loss
        self.reg_loss = reg_loss
        self.rmse = rmse
        self.corr = corr


class EvidentialTrainer(BaseTrainer):
    def __init__(self,
                 config: EvidentialConfig,
                 training_dataset: Optional[MoleculeDataset] = None,
                 valid_dataset: Optional[MoleculeDataset] = None,
                 test_dataset: Optional[MoleculeDataset] = None):

        super().__init__(config, training_dataset, valid_dataset, test_dataset)

        self._loss_func = EvidentialRegressionLoss(config.loss_reg_coeff)
        self._scaler = None

    def initialize_model(self):
        # ----- initialize model -----
        self._model = EvidentialRegressionModel(config=self._config)
        return self

    def initialize_optimizer(self, optimizer=None):
        # ----- initialize optimizer -----
        if optimizer is not None:
            self._optimizer = optimizer
        else:
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self._config.lr, weight_decay=1E-5
            )
        return self

    def run(self):

        logger.info(" --- Start training --- ")

        writer = SummaryWriter(self._config.tensorboard_dir)

        if self._config.enable_lb_scaling:
            self._scaler = self._training_dataset.scale_lbs()
        else:
            self._scaler = None

        training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)
        features = next(iter(training_dataloader))['features']

        # Write graph to tensorboard
        self._model.return_tensor = True
        writer.add_graph(self._model, features)
        self._model.return_tensor = False

        self._model.to(self._config.device)

        best_valid_rmse = np.inf
        for epoch_i in tqdm(range(self._config.epochs)):

            train_nll_loss, train_reg_loss = self.training_step(training_dataloader)

            valid_metrics = self.valid()

            writer.add_scalar("train-loss/nll", train_nll_loss, global_step=epoch_i + 1)
            writer.add_scalar("train-loss/reg", train_reg_loss, global_step=epoch_i + 1)
            writer.add_scalar("valid-loss/nll", valid_metrics.nll_loss, global_step=epoch_i + 1)
            writer.add_scalar("valid-loss/reg", valid_metrics.reg_loss, global_step=epoch_i + 1)
            writer.add_scalar("valid-metric/rmse", valid_metrics.rmse, global_step=epoch_i + 1)
            writer.add_scalar("valid-metric/corr", valid_metrics.corr, global_step=epoch_i + 1)

            if (epoch_i + 1) % self._config.plot_interval == 0:
                if self._config.use_toy_dataset:
                    self.plot_toy(epoch=epoch_i + 1, tensorboard_writer=writer)
                else:
                    self.plot_results(epoch=epoch_i + 1, tensorboard_writer=writer)

            if valid_metrics.rmse < best_valid_rmse:
                self.save()
                best_valid_rmse = valid_metrics.rmse

        self.load()
        self.write_preds(dataset=self._test_dataset)

        writer.close()

        return None

    def training_step(self, data_loader: DataLoader):
        self._model.train()

        train_nll_loss = 0
        train_reg_loss = 0
        num_samples = 0

        for batch in data_loader:

            # get data
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self._device)
            batch_size = len(batch['labels'])
            num_samples += batch_size

            self._optimizer.zero_grad()
            preds = self._model(**batch)

            loss_nll, loss_reg = self._loss_func(batch['labels'], *preds)
            loss = loss_nll + loss_reg
            loss.backward()
            self._optimizer.step()

            train_nll_loss += loss_nll.item() * batch_size
            train_reg_loss += loss_reg.item() * batch_size

        train_nll_loss /= num_samples
        train_reg_loss /= num_samples

        return train_nll_loss, train_reg_loss

    def valid(self) -> EvidentialRegressionMetrics:
        metrics = self.evaluate(
            self._valid_dataset,
            metric_funcs=['loss', 'rmse', 'corr']
        )
        return metrics

    def test(self):
        test_metric = self.evaluate(self._test_dataset)
        return test_metric

    def evaluate(self,
                 dataset: MoleculeDataset,
                 metric_funcs: Optional[List[str]] = None):
        metrics = EvidentialRegressionMetrics()

        if metric_funcs is None:
            metric_funcs = ['rmse']
        true_lbs = dataset.labels
        preds = self.predict(dataset)

        if 'loss' in metric_funcs:
            metrics.nll_loss, metrics.reg_loss = self._loss_func(preds, torch.from_numpy(true_lbs))

        mu, v, alpha, beta = preds.numpy()

        if 'rmse' in metric_funcs:
            metrics.rmse = rmse_func(true_lbs, mu)

        if 'corr' in metric_funcs:
            var = self.get_var(v, alpha, beta)
            metrics.corr = pearsonr(np.abs(true_lbs-mu), var)[0]

        return metrics

    def predict(self, dataset: MoleculeDataset):

        preds = super().predict(dataset)

        # TODO: this is incorrect, need to be fixed
        if self._scaler is not None:
            for i in range(len(preds)):
                preds[i] = self._scaler.inverse_transform(preds[i])

        return preds

    def write_results(self, valid_results: List[float], test_result: Optional[float] = None):
        with open(os.path.join(self._config.output_dir, 'results.txt'), 'w') as f:
            for i, valid_result in enumerate(valid_results):
                f.write(f"Epoch {i+1} validation result: {valid_result}\n")

            if test_result is not None:
                f.write(f"Test result: {test_result}\n")
        return None

    def write_preds(self, dataset: MoleculeDataset):
        true_lbs = dataset.labels

        preds = self.predict(dataset)
        mu, v, alpha, beta = preds.numpy()
        var = self.get_var(v, alpha, beta)
        df = pd.DataFrame({"mu": mu, "var": var, "true_lbs": true_lbs})

        df.to_csv(os.path.join(self._config.output_dir, "preds.csv"), index=False)

        return None

    def plot_toy(self,
                 dataset=None,
                 epoch=None,
                 tensorboard_writer: Optional = None,
                 directory: Optional[str] = 'plots-toy'):
        """
        Plot results.
        """
        dataset = dataset if dataset is not None else self._test_dataset

        preds = self.predict(dataset)
        mu, v, alpha, beta = (x.numpy() for x in preds)

        epistemic_var = np.sqrt(beta / (v * (alpha - 1)))
        epistemic_var = np.minimum(epistemic_var, 1e3)

        aleatoric_var = np.sqrt(beta / (alpha - 1))
        aleatoric_var = np.minimum(aleatoric_var, 1e3)

        train_x = self._training_dataset.features.squeeze()
        train_y = self._training_dataset.labels.squeeze()
        test_x = self._test_dataset.features.squeeze()
        test_y = self._test_dataset.labels.squeeze()

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 6), dpi=200)
        ax1.scatter(train_x, train_y, s=1., c='#463c3c', zorder=0, label="Train")

        ax1.plot(test_x, test_y, 'r--', zorder=2, label="True")
        ax1.plot(test_x, mu, color='#007cab', zorder=3, label="Pred")

        for k in np.linspace(1, 3, 3):
            ax1.fill_between(
                test_x, (mu - k * epistemic_var), (mu + k * epistemic_var),
                alpha=0.3,
                edgecolor=None,
                facecolor='#00aeef',
                linewidth=0,
                zorder=1,
                label="Unc." if k == 1. else None
            )
        ax1.set_ylim(-150, 150)
        ax1.set_xlim(-7, 7)
        ax1.set_title("Epistemic Uncertainty")
        ax1.legend()

        ax2.scatter(train_x, train_y, s=1., c='#463c3c', zorder=0, label="Train")

        ax2.plot(test_x, test_y, 'r--', zorder=2, label="True")
        ax2.plot(test_x, mu, color='#007cab', zorder=3, label="Pred")

        for k in np.linspace(1, 3, 3):
            ax2.fill_between(
                test_x, (mu - k * aleatoric_var * 5), (mu + k * aleatoric_var * 5),
                alpha=0.3,
                edgecolor=None,
                facecolor='#00aeef',
                linewidth=0,
                zorder=1,
                label="Unc." if k == 0 else None
            )
        ax2.set_ylim(-150, 150)
        ax2.set_xlim(-7, 7)
        ax2.set_title("Aleatoric Uncertainty")
        ax2.legend()

        plt.tight_layout()

        # present figure
        if tensorboard_writer is None:
            self.save_fig(fig, epoch, directory)
        else:
            self.log_tensorboard_fig(tensorboard_writer, fig, epoch, directory)

    def plot_results(self,
                     dataset=None,
                     epoch=None,
                     tensorboard_writer: Optional = None,
                     directory: Optional[str] = "plots-results"):

        dataset = dataset if dataset is not None else self._test_dataset

        preds = self.predict(dataset)
        mu, v, alpha, beta = (x.numpy() for x in preds)

        epistemic_var = np.sqrt(beta / (v * (alpha - 1)))
        var = epistemic_var

        test_y = self._test_dataset.labels.squeeze()

        lb_range = np.linspace(test_y.min() - 1, test_y.max() + 1, 2000)

        fig, ax1 = plt.subplots(ncols=1, figsize=(5, 4), dpi=200)

        ax1.plot(lb_range, lb_range, color='darkorange', alpha=0.6, zorder=10, label='true lbs')
        ax1.scatter(test_y, mu, color='firebrick', s=20, alpha=0.8, zorder=100, label='pred lbs')
        ax1.errorbar(test_y, mu, yerr=np.stack([var, var], axis=0), fmt='None', label='3 sigma uncertainty')
        ax1.legend()
        ax1.set_title('Evidential regression results and uncertainty')

        plt.tight_layout()

        # present figure
        if tensorboard_writer is None:
            self.save_fig(fig, epoch, directory)
        else:
            self.log_tensorboard_fig(tensorboard_writer, fig, epoch, directory)

    @staticmethod
    def get_var(v, alpha, beta):
        return np.sqrt(beta / (v * (alpha - 1)))
