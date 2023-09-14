# Trainer for fine classes training only
from components.trainer.base import BaseTrainer
import torch
from torch.nn import functional as F
from collections import defaultdict
from constants import ALT_FREQ

# from components.model.pretrained import VitImageClassificationBroadFin


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

        broad_emb_seg = []
        fine_emb_seg = []

        for idx in range(2):
            broad_emb_seg.append(
                self.get_filtered_tensor(broad_emb[model_inputs["broad_labels"] == idx])
            )

        for idx in range(10):
            fine_emb_seg.append(
                self.get_filtered_tensor(fine_emb[model_inputs["fine_labels"] == idx])
            )

        mean_broad_0 = self.get_filtered_tensor(
            torch.mean(broad_emb_seg[0], 0).unsqueeze(0)
        )
        mean_fine_0 = self.get_filtered_tensor(
            torch.mean(broad_emb_seg[1], 0).unsqueeze(0)
        )

        fine_class_0 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[0], 0).unsqueeze(0)
        )
        fine_class_1 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[1], 0).unsqueeze(0)
        )
        fine_class_2 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[2], 0).unsqueeze(0)
        )
        fine_class_3 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[3], 0).unsqueeze(0)
        )
        fine_class_4 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[4], 0).unsqueeze(0)
        )
        fine_class_5 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[5], 0).unsqueeze(0)
        )
        fine_class_6 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[6], 0).unsqueeze(0)
        )
        fine_class_7 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[7], 0).unsqueeze(0)
        )
        fine_class_8 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[8], 0).unsqueeze(0)
        )
        fine_class_9 = self.get_filtered_tensor(
            torch.mean(fine_emb_seg[9], 0).unsqueeze(0)
        )

        fine_0_cat = torch.cat(
            (
                fine_class_0,
                fine_class_1,
                fine_class_8,
                fine_class_9,
            ),
            dim=0,
        )

        mean_fine_0 = torch.mean(fine_0_cat, 0).unsqueeze(0)
        fine_1_cat = torch.cat(
            (
                fine_class_2,
                fine_class_3,
                fine_class_4,
                fine_class_5,
                fine_class_6,
                fine_class_7,
            ),
            dim=0,
        )
        mean_fine_1 = torch.mean(fine_1_cat, 0).unsqueeze(0)

        broad_criteria = lambda x, y: torch.linalg.norm(x - y)

        loss_0 = broad_criteria(mean_fine_0, mean_broad_0)
        loss_1 = broad_criteria(mean_fine_1, mean_fine_0)

        # criterion_broad = kwargs[self.criterion_name_broad]
        criterion_fine = kwargs[self.criterion_name_fine]

        # loss_broad = criterion_broad(broad_logits, model_inputs["broad_labels"])
        loss_fine = criterion_fine(fine_logits, model_inputs["fine_labels"])

        loss_broad = loss_0 + loss_1
        # print(
        #     "Loss 0: ",
        #     loss_0.detach().item(),
        #     "Loss 1: ",
        #     loss_1.detach().item(),
        #     "Loss Fine: ",
        #     loss_fine.detach().item(),
        #     "Loss Broad: ",
        #     loss_broad.detach().item(),
        # )

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

    def get_curr_loss_idx(self, epoch):
        # Val % len(loss_list) for more general
        val, _ = divmod(epoch, ALT_FREQ)
        return val % 2

    def _train(self, epoch, **kwargs):
        print("\n\nCURR EPOCH: ", epoch)
        self.model.train()
        epoch_metrics, epoch_losses, results = self._init_params(self.mode.TRAIN, epoch)

        for _, batch in enumerate(self.train_dl):
            loss_list, metrics = self._get_losses_and_metrics(batch, **kwargs)

            epoch_metrics = self.evaluate_metrics(epoch_metrics, metrics)

            loss_idx = self.get_curr_loss_idx(epoch)
            print("Traing Current Loss IDX: ", loss_idx)
            loss = loss_list[loss_idx]
            loss.backward()
            epoch_losses.append(loss.item())

            for opt in self.optimizer_list:
                opt.step()
                opt.zero_grad()

            # break

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

            loss_idx = self.get_curr_loss_idx(epoch)

            loss = loss_list[loss_idx]
            epoch_losses.append(loss.item())

            # break

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
