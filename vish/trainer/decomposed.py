import os

import torch
from torch import nn

from vish.model.decomposed.decomposed import VitClassificationDecomposed
from vish.model.decomposed.entity import TransformerData
from vish.trainer.base import BaseTrainer
from vish.utils import DEVICE

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def rearrange_outputs(class_wise_logits: dict[str, TransformerData], num_outputs):
    # y_pred, y_true
    y_pred = torch.empty(0, num_outputs, device=DEVICE, requires_grad=True)
    y_true = torch.empty(0, 1, device=DEVICE, dtype=torch.int8)

    for fine_class_data in class_wise_logits.values():
        t = fine_class_data
        y_pred = torch.cat([y_pred, t.data], dim=0)
        y_true = torch.cat([y_true, t.labels], dim=0)

    y_true.squeeze_(1)

    return y_pred, y_true


class VitDecomposedTrainer(BaseTrainer):
    """
    Trainer for VitClassificationDecomposed
    """

    model: VitClassificationDecomposed

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Training Started for ViT Decomposed model")

        self.criterion = nn.CrossEntropyLoss()

    def batch_2_model_input(self, batch):
        imgs, fine_labels, _ = batch
        return {
            "pixel_values": imgs,
            "labels": fine_labels,
        }

    def get_outputs(self, model_inputs, **kwargs) -> tuple[list, dict]:
        # fine is last
        num_outputs = self.model.num_leaves
        class_wise_logits = self.model(model_inputs)
        y_pred, y_true = rearrange_outputs(class_wise_logits, num_outputs)
        loss = self.criterion(y_pred, y_true)

        metrics = {}

        with torch.no_grad():
            for metric_key, metric_fxn in self.metrics_list:
                metrics[metric_key] = metric_fxn(
                    y_pred.clone().detach().cpu(),
                    y_true.clone().detach().cpu(),
                )

        return [loss], metrics
