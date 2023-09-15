# Trainer for fine classes training only
from components.trainer.base import BaseTrainer
import torch
from torch.nn import functional as F
from constants import EPOCHS

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


