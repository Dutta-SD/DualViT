from collections import defaultdict

import numpy as np
import torch

from vish.model.common.loss import (
    empty_if_problem,
    broad_criterion,
    bnf_alternate_loss,
)
from vish.model.tp.dual import TPDualVit
from vish.trainer.base import BaseTrainer


class VitDualModelTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Starting Dual Model Training Training with Modified Loss ...")
        self.criterion_name_fine = "fine_class_CE"
        self.criterion_name_broad = "broad_class_CE"

    def batch_2_model_input(self, batch):
        images, fine_labels, broad_labels = batch

        return {
            "pixel_values": images,
            "fine_labels": fine_labels,
            "broad_labels": broad_labels,
        }

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        emb_fine, output_fine, emb_broad, output_broad = self.model(
            model_inputs["pixel_values"]
        )

        fine_emb_clone = emb_fine.detach().clone()
        fine_emb_clone.requires_grad = False

        broad_emb_seg = []
        fine_emb_seg = []

        for idx in range(2):
            broad_emb_seg.append(
                empty_if_problem(emb_broad[model_inputs["broad_labels"] == idx])
            )

        for idx in range(10):
            fine_emb_seg.append(
                empty_if_problem(fine_emb_clone[model_inputs["fine_labels"] == idx])
            )

        mean_broad_0 = empty_if_problem(torch.mean(broad_emb_seg[0], 0).unsqueeze(0))
        mean_broad_1 = empty_if_problem(torch.mean(broad_emb_seg[1], 0).unsqueeze(0))

        fine_means = [
            empty_if_problem(torch.mean(fine_emb_seg[i], 0).unsqueeze(0))
            for i in range(10)
        ]

        fine_0_cat = torch.cat(
            (
                fine_means[0],
                fine_means[1],
                fine_means[8],
                fine_means[9],
            ),
            dim=0,
        )

        mean_fine_0 = torch.mean(fine_0_cat, 0).unsqueeze(0)

        fine_1_cat = torch.cat(
            (
                fine_means[2],
                fine_means[3],
                fine_means[4],
                fine_means[5],
                fine_means[6],
                fine_means[7],
            ),
            dim=0,
        )
        mean_fine_1 = torch.mean(fine_1_cat, 0).unsqueeze(0)

        loss_0 = broad_criterion(mean_fine_0, mean_broad_0)
        loss_1 = broad_criterion(mean_fine_1, mean_broad_1)

        loss_0 = torch.nan_to_num(loss_0)

        loss_1 = torch.nan_to_num(loss_1)

        criterion_fine = kwargs[self.criterion_name_fine]
        # criterion_broad = kwargs[self.criterion_name_broad]

        loss_fine = criterion_fine(output_fine, model_inputs["fine_labels"])

        loss_broad = loss_0 + loss_1
        # loss_broad = criterion_broad(output_broad, model_inputs["broad_labels"])

        metrics = {}

        with torch.no_grad():
            for metric_key, metric_fxn in self.metrics_list:
                # Eg: Acc@1_broad, Acc@1_fine
                metrics[f"{metric_key}_broad"] = metric_fxn(
                    output_broad.clone().detach().cpu(),
                    model_inputs["broad_labels"].clone().detach().cpu(),
                )
                metrics[f"{metric_key}_fine"] = metric_fxn(
                    output_fine.clone().detach().cpu(),
                    model_inputs["fine_labels"].clone().detach().cpu(),
                )

        return [loss_broad, loss_fine], metrics

    def _train(self, epoch, **kwargs):
        print("\nCURR EPOCH: ", epoch)
        self.model.train()
        epoch_metrics, epoch_losses, results = self._init_params(self.mode.TRAIN, epoch)

        for batch_idx, batch in enumerate(self.train_dl):
            kwargs["epoch"] = epoch
            epoch_metrics = self.evaluate_model(
                batch, epoch_losses, epoch_metrics, kwargs
            )

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

    def evaluate_model(self, batch, epoch_losses, epoch_metrics, kwargs):
        loss_broad, loss_fine, metrics = self.get_model_outputs(batch, kwargs)
        epoch_metrics = self.evaluate_metrics(epoch_metrics, metrics)
        self.train_backward(loss_broad, loss_fine)
        epoch_losses.append([loss_broad.item(), loss_fine.item()])
        return epoch_metrics

    def train_backward(self, loss_broad, loss_fine):
        # Different model with no connections, so no need of retain_graph
        loss_broad.backward()
        loss_fine.backward()

    @torch.no_grad()
    def validate(self, epoch, save_chkpt=True, **kwargs):
        self.model.eval()
        epoch_metrics, epoch_losses, results = self._init_params(self.mode.EVAL, epoch)

        for batch_idx, batch in enumerate(self.test_dl):
            kwargs["epoch"] = epoch
            loss_broad, loss_fine, metrics = self.get_model_outputs(batch, kwargs)
            epoch_metrics = self.evaluate_metrics(epoch_metrics, metrics)

            epoch_losses.append([loss_broad.item(), loss_fine.item()])

        results = self.update_results_and_log(results, epoch_losses, epoch_metrics)

        if results["metrics"][self.best_score_key] > results["best"]:
            print(
                f"Eval Metrics increased from: {results['best']} -> {results['metrics'][self.best_score_key]}"
            )
            self.update_best_score(results)

            if save_chkpt:
                self.save_model_checkpoint()
        return results

    def get_model_outputs(self, batch, kwargs):
        loss_list, metrics = self._get_losses_and_metrics(batch, **kwargs)
        loss_broad, loss_fine = loss_list
        return loss_broad, loss_fine, metrics

    def evaluate_metrics(self, epoch_metrics, metrics):
        for met_name in metrics.keys():
            epoch_metrics[met_name].append(metrics[met_name])
        return epoch_metrics

    def _init_params(self, mode, epoch):
        results = self.get_default_result_dict(mode)
        results["epoch"] = epoch
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        return epoch_metrics, epoch_losses, results

    def update_results_and_log(self, results, epoch_losses, epoch_metrics):
        results["losses"] = np.mean(epoch_losses, axis=0)
        results["metrics"] = {name: np.mean(met) for name, met in epoch_metrics.items()}

        print()
        print(f"Epoch: {results['epoch']} Summary")
        print(f"Mode: {results['mode']}")
        print(f"Losses: {results['losses']}")
        print(f"Metrics: {results['metrics']}")
        print(f"Best {self.best_score_key} till now: {results['best']}")
        print()

        return results


class TPDualTrainer(VitDualModelTrainer):
    """
    Dual Training for a TP model
    """

    model: TPDualVit

    def train_backward(self, loss_broad, loss_fine):
        loss_broad.backward(retain_graph=True)
        loss_fine.backward()

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        # [broad_embedding, broad_logits], [fine_embedding, fine_logits]
        broad_output, fine_output = self.model(model_inputs["pixel_values"])
        fine_labels = model_inputs["fine_labels"]
        broad_labels = model_inputs["broad_labels"]

        # broad_loss, fine_loss = bnf_embedding_loss(
        #     broad_output,
        #     fine_output,
        #     broad_labels,
        #     fine_labels,
        #     criterion_fine,
        # )

        # broad_loss, fine_loss = bnf_embedding_cluster_loss(
        #     broad_output,
        #     fine_output,
        #     broad_labels,
        #     fine_labels,
        #     criterion_fine,
        # )

        broad_loss, fine_loss = bnf_alternate_loss(
            broad_output,
            fine_output,
            broad_labels,
            fine_labels,
            kwargs["epoch"],
            classifier=self.model.fine_model.to_logits,
        )

        metrics = {}

        with torch.no_grad():
            for metric_key, metric_fxn in self.metrics_list:
                # Eg: Acc@1_broad, Acc@1_fine
                metrics[f"{metric_key}_broad"] = metric_fxn(
                    broad_output[1].clone().detach().cpu(),
                    broad_labels.clone().detach().cpu(),
                )
                # Fine outputs are list as adapted from a multi token model
                metrics[f"{metric_key}_fine"] = metric_fxn(
                    fine_output[1][0].clone().detach().cpu(),
                    fine_labels.clone().detach().cpu(),
                )

        return [broad_loss, fine_loss], metrics
