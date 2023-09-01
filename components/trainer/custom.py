# Trainer for fine classes training only
from components.trainer.base import BaseTrainer
import torch
from collections import defaultdict
from constants import ALT_FREQ


class BroadClassTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Starting Broad Class Training...")
        self.criterion_name = "broad_class_CE"

    def batch_2_model_input(self, batch):
        imgs, fine_labels, broad_labels = batch
        return {
            "pixel_values": imgs,
            "fine_labels": fine_labels,
            "broad_labels": broad_labels,
        }

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        # fine is last
        logits, *_ = self.model(model_inputs["pixel_values"])
        criterion = kwargs[self.criterion_name]
        loss = criterion(logits, model_inputs["broad_labels"])

        metrics = {}

        with torch.no_grad():
            for metric_key, metric_fxn in self.metrics_list:
                metrics[metric_key] = metric_fxn(
                    logits.clone().detach().cpu(),
                    model_inputs["broad_labels"].clone().detach().cpu(),
                )

        return [loss], metrics


class FineClassTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Starting Fine Class Training...")
        self.criterion_name = "fine_class_CE"

    def batch_2_model_input(self, batch):
        imgs, fine_labels, broad_labels = batch
        return {
            "pixel_values": imgs,
            "fine_labels": fine_labels,
            "broad_labels": broad_labels,
        }

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        # fine is last
        *_, logits = self.model(model_inputs["pixel_values"])
        criterion = kwargs[self.criterion_name]
        loss = criterion(logits, model_inputs["fine_labels"])

        metrics = {}

        with torch.no_grad():
            for metric_key, metric_fxn in self.metrics_list:
                metrics[metric_key] = metric_fxn(
                    logits.clone().detach().cpu(),
                    model_inputs["fine_labels"].clone().detach().cpu(),
                )

        return [loss], metrics


class BroadAndFineAlternateTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Starting Broad Fine Alternate Class Training...")
        self.criterion_name_broad = "broad_class_CE"
        self.criterion_name_fine = "fine_class_CE"

    def batch_2_model_input(self, batch):
        imgs, fine_labels, broad_labels = batch
        return {
            "pixel_values": imgs,
            "fine_labels": fine_labels,
            "broad_labels": broad_labels,
        }

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        # fine is last
        broad_logits, fine_logits = self.model(model_inputs["pixel_values"])
        criterion_broad = kwargs[self.criterion_name_broad]
        criterion_fine = kwargs[self.criterion_name_fine]

        loss_broad = criterion_broad(broad_logits, model_inputs["broad_labels"])
        loss_fine = criterion_fine(fine_logits, model_inputs["fine_labels"])

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

    def _train(self, epoch, **kwargs):
        self.model.train()
        epoch_metrics, epoch_losses, results = self._init_params(self.mode.TRAIN, epoch)

        for _, batch in enumerate(self.train_dl):
            loss_list, metrics = self._get_losses_and_metrics(batch, **kwargs)

            epoch_metrics = self.evaluate_metrics(epoch_metrics, metrics)

            curr_epoch_loss_idx = 0 if (epoch % ALT_FREQ) == 0 else 1
            loss = loss_list[curr_epoch_loss_idx]
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
