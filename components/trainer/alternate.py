from components.trainer.base import BaseTrainer
from constants import ALT_FREQ, EPOCHS


import numpy as np
import torch


from collections import defaultdict


class BroadFineAlternateModifiedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Starting Broad Fine Training with Modified Loss ...")
        self.criterion_name_fine = "fine_class_CE"

    def batch_2_model_input(self, batch):
        imgs, fine_labels, broad_labels = batch
        return {
            "pixel_values": imgs,
            "fine_labels": fine_labels,
            "broad_labels": broad_labels,
        }

    def get_filtered_tensor(self, tensor: torch.IntTensor):
        _, *rest = tensor.shape
        if torch.isnan(tensor).sum() > 0:
            return torch.empty((0, *rest), device=tensor.device)
        return tensor

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        # fine is last
        embedding_list, ouput_list = self.model(model_inputs["pixel_values"])
        broad_logits, fine_logits = ouput_list
        broad_emb, fine_emb = embedding_list

        fine_emb_clone = fine_emb.detach().clone()
        fine_emb_clone.requires_grad=False

        broad_emb_seg = []
        fine_emb_seg = []

        for idx in range(2):
            broad_emb_seg.append(
                self.get_filtered_tensor(broad_emb[model_inputs["broad_labels"] == idx])
            )

        for idx in range(10):
            fine_emb_seg.append(
                self.get_filtered_tensor(fine_emb_clone[model_inputs["fine_labels"] == idx])
            )

        mean_broad_0 = self.get_filtered_tensor(
            torch.mean(broad_emb_seg[0], 0).unsqueeze(0)
        )
        mean_fine_0 = self.get_filtered_tensor(
            torch.mean(broad_emb_seg[1], 0).unsqueeze(0)
        )

        fine_means = [
            self.get_filtered_tensor(torch.mean(fine_emb_seg[i], 0).unsqueeze(0))
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

        broad_criteria = lambda x, y: torch.linalg.norm(x - y)

        loss_0 = broad_criteria(mean_fine_0, mean_broad_0)
        loss_1 = broad_criteria(mean_fine_1, mean_fine_0)

        criterion_fine = kwargs[self.criterion_name_fine]

        loss_fine = criterion_fine(fine_logits, model_inputs["fine_labels"])

        loss_broad = loss_0 + loss_1

        metrics = {}

        with torch.no_grad():
            for metric_key, metric_fxn in self.metrics_list:
                # Eg: Acc@1_broad, Acc@1_fine
                metrics[f"{metric_key}_broad"] = metric_fxn(
                    broad_logits.clone().detach().cpu(),
                    model_inputs["broad_labels"].clone().detach().cpu(),
                )
                metrics[f"{metric_key}_fine"] = metric_fxn(
                    fine_logits.clone().detach().cpu(),
                    model_inputs["fine_labels"].clone().detach().cpu(),
                )

        return [loss_broad, loss_fine], metrics

    @staticmethod
    def get_curr_loss_idx(epoch):
        lim = int(EPOCHS // 3)
        if epoch < lim:
            return 1

        # Val % len(loss_list) for more general
        val, _ = divmod(epoch - lim, ALT_FREQ)
        return val % 2

    def _train(self, epoch, **kwargs):
        print("\n\nCURR EPOCH: ", epoch)
        self.model.train()
        epoch_metrics, epoch_losses, results = self._init_params(self.mode.TRAIN, epoch)

        for batch_idx, batch in enumerate(self.train_dl):
            loss_list, metrics = self._get_losses_and_metrics(batch, **kwargs)

            epoch_metrics = self.evaluate_metrics(epoch_metrics, metrics)

            loss_idx = BroadFineAlternateModifiedTrainer.get_curr_loss_idx(epoch)
            loss = loss_list[loss_idx]
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

        for batch_idx, batch in enumerate(self.test_dl):
            loss_list, metrics = self._get_losses_and_metrics(batch, **kwargs)

            epoch_metrics = self.evaluate_metrics(epoch_metrics, metrics)

            loss_idx = BroadFineAlternateModifiedTrainer.get_curr_loss_idx(epoch)

            loss = loss_list[loss_idx]
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
        results["losses"] = np.mean(epoch_losses)
        results["metrics"] = {name: np.mean(met) for name, met in epoch_metrics.items()}

        print()
        print(f"Epoch: {results['epoch']} Summary")
        print(f"Mode: {results['mode']}")
        print(f"Losses: {results['losses']}")
        print(f"Metrics: {results['metrics']}")
        print(f"Best {self.best_score_key} till now: {results['best']}")
        print()

        return results