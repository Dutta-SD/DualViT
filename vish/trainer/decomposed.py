import os

import torch
from torch import nn

from vish.model.decomposed.decomposed import VitClassificationDecomposed
from vish.model.decomposed.entity import TransformerData
from vish.trainer.base import BaseTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class VitDecomposedTrainer(BaseTrainer):
    """
    Trainer for VitClassificationDecomposed
    """

    model: VitClassificationDecomposed

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Training Started for ViT Decomposed model")

        self.criterion = nn.CrossEntropyLoss()

    def get_aggregate_loss(self, class_wise_logits: dict[str, TransformerData]):
        all_losses = []

        for op in class_wise_logits.values():
            logits = op.data
            labels = op.labels

            if 0 not in logits.shape:
                all_losses.append(self.criterion(logits, labels))

        return torch.sum(torch.stack(all_losses))

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
        loss = self.get_aggregate_loss(class_wise_logits)

        metrics = {}

        with torch.no_grad():
            for metric_key, metric_fxn in self.metrics_list:
                metrics[metric_key] = metric_fxn(
                    y_pred.detach(),
                    y_true.detach(),
                )

        return [loss], metrics
