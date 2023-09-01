from typing import Any, Callable
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from enum import Enum
import time
import numpy as np


class BaseTrainer:
    """
    Base class for trainer logic. Contains common function
    """

    def __init__(
        self,
        num_epochs: int,
        train_dl: DataLoader,
        test_dl: DataLoader,
        model: nn.Module,
        optimizer_list: list[opt.Optimizer],
        scheduler_list: list[Any],
        metrics_list: list[tuple[str, Callable]],
        best_score_key: str,
        model_checkpoint_dir: str,
        description: str = "",
        uniq_desc: bool = True,
    ) -> None:
        super().__init__()
        self.num_epochs = num_epochs
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.model = model
        self.optimizer_list = optimizer_list
        self.scheduler_list = scheduler_list
        self.metrics_list = metrics_list
        self.best_score_key = best_score_key
        self.best_train_score = -1
        self.best_test_score = -1
        self.mode = Enum("MODE", ["TRAIN", "EVAL"])
        self.ckpt_dir = model_checkpoint_dir
        self.exp_name = (
            f"{description}_{int(time.time())}" if uniq_desc else description
        )
        self.ckpt_full_path = f"{self.ckpt_dir}/{self.exp_name}.pt"
        print("Experiment: ", self.exp_name)

    def get_default_result_dict(self, mode):
        return {
            "losses": [],
            "metrics": None,
            "mode": mode,
            "epoch": None,
            "best": self.best_train_score
            if mode == self.mode.TRAIN
            else self.best_test_score,
        }

    def batch_2_model_input(self, batch) -> Any:
        pass

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        pass

    def update_results_and_log(self, results, epoch_losses, epoch_metrics):
        results["losses"] = np.mean(epoch_losses)
        results["metrics"] = {name: np.mean(met) for name, met in epoch_metrics.items()}

        print()
        print(f"Epoch: {results['epoch']}")
        print(f"Mode: {results['mode']}")
        print(f"Losses: {results['losses']}")
        print(f"Metrics: {results['metrics']}")
        print(f"Best {self.best_score_key} till now: {results['best']}")
        print()

        return results

    def update_best_score(self, results):
        best_res = results["metrics"][self.best_score_key]
        if results["mode"] == self.mode.TRAIN:
            self.best_train_score = max(self.best_train_score, best_res)
        else:
            self.best_test_score = max(self.best_test_score, best_res)

    def save_model_checkpoint(self):
        torch.save(
            {
                "model": self.model,
                "opt": [opt.state_dict() for opt in self.optimizer_list],
                "tags": ["vit"],
            },
            self.ckpt_full_path,
        )
        print(f"Model CKPT saved at: {self.ckpt_full_path}")

    def run(self, **kwargs):
        print("Starting Train/Eval...")

        for epoch in range(self.num_epochs):
            start_time = time.time()
            self._train(epoch, **kwargs)
            val_results = self.validate(epoch, **kwargs)
            for scheduler in self.scheduler_list:
                try:
                    scheduler.step(metrics=val_results["metrics"][self.best_score_key])
                except Exception:
                    scheduler.step()
            end_time = time.time()
            print(f"Epoch took {end_time-start_time:.2f} s")

    def _init_params(self, mode, epoch):
        results = self.get_default_result_dict(mode)
        results["epoch"] = epoch
        epoch_losses = []
        epoch_metrics = {name: [] for name, _ in self.metrics_list}
        return epoch_metrics, epoch_losses, results

    def evaluate_metrics(self, epoch_metrics, metrics):
        for met_name in epoch_metrics.keys():
            epoch_metrics[met_name].append(metrics[met_name])
        return epoch_metrics

    def _train(self, epoch, **kwargs):
        self.model.train()
        epoch_metrics, epoch_losses, results = self._init_params(self.mode.TRAIN, epoch)

        for _, batch in enumerate(self.train_dl):
            loss_list, metrics = self._get_losses_and_metrics(batch, **kwargs)

            epoch_metrics = self.evaluate_metrics(epoch_metrics, metrics)

            for loss in loss_list:
                loss.backward()
                epoch_losses.append(loss.item())

            for opt in self.optimizer_list:
                opt.step()
                opt.zero_grad()

        results = self.update_results_and_log(results, epoch_losses, epoch_metrics)

        if results["metrics"][self.best_score_key] > results["best"]:
            print(
                f"Train Metrics increased from: {results['best']} -> {results['metrics'][self.best_score_key]}"
            )
            self.update_best_score(results)

        return results

    @torch.no_grad()
    def validate(self, epoch, save_chkpt=True, **kwargs):
        self.model.eval()
        epoch_metrics, epoch_losses, results = self._init_params(self.mode.EVAL, epoch)

        for _, batch in enumerate(self.test_dl):
            loss_list, metrics = self._get_losses_and_metrics(batch, **kwargs)

            epoch_metrics = self.evaluate_metrics(epoch_metrics, metrics)

            for loss in loss_list:
                epoch_losses.append(loss.item())

        results = self.update_results_and_log(results, epoch_losses, epoch_metrics)

        if results["metrics"][self.best_score_key] > results["best"]:
            print(
                f"Eval Metrics increased from: {results['best']} -> {results['metrics'][self.best_score_key]}"
            )
            self.update_best_score(results)

            if save_chkpt:
                self.save_model_checkpoint()
        return results

    def _get_losses_and_metrics(self, batch, **kwargs):
        model_inputs = self.batch_2_model_input(batch)
        loss_list, metrics = self.get_outputs(model_inputs, **kwargs)
        return loss_list, metrics
