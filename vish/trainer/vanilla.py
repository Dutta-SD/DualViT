from typing import Union

import torch

from vish.model.tp.tp_vit import TPVitImageClassification
from vish.model.vanilla.vit import ViTBasicForImageClassification
from vish.trainer.base import BaseTrainer


class VanillaVitTrainer(BaseTrainer):
    """
    Useful for training Vanilla ViT, TP Vit etc
    """

    model: TPVitImageClassification | ViTBasicForImageClassification

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Starting Training...")
        self.criterion_name = "criterion"

    def batch_2_model_input(self, batch):
        img_tensors, fine_labels, broad_labels = batch
        return {
            "pixel_values": img_tensors,
            "fine_labels": fine_labels,
            "broad_labels": broad_labels,
        }

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        # fine is last
        logits, *_ = self.model(model_inputs["pixel_values"])
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
