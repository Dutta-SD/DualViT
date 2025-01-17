import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch import nn
from torchmetrics.functional import accuracy

from dualvit.constants import WEIGHT_DECAY, LEARNING_RATE
from dualvit.lightning.loss import BroadFineEmbeddingLoss, BELMode
from dualvit.model.models.dualvit import DualViT

VALIDATION_METRIC_NAME = "val_af"  # Should be same as defined in Trainer


def convert_fine_to_broad_logits(
    logits: Tensor,
    broad_labels: Tensor,
    fine_labels: Tensor,
    num_broad: int,
):
    """
    Converts fine logits to broad logits for CIFAR 10

    Args:
        logits: logits of shape [B, 1, C_fine] or [B, C_fine]
        broad_labels: Labels of shape [B, ]
        fine_labels: Labels of shape [B, ]

    Returns:

    """

    if len(logits.shape) == 3 and logits.shape[1] == 1:
        logits = logits.squeeze(1)

    batch, *_ = logits.shape
    b_logits = torch.empty((batch, 0), device=logits.device)

    for b_idx in range(num_broad):
        fine_indexes = torch.unique(fine_labels[broad_labels == b_idx])

        if torch.numel(fine_indexes) == 0:
            curr_idx_logits = torch.zeros(
                (batch,), device=logits.device, requires_grad=True
            )
        else:
            curr_idx_logits, _ = torch.max(logits[:, fine_indexes], dim=1)
        curr_idx_logits = curr_idx_logits.unsqueeze(1)

        b_logits = torch.cat([b_logits, curr_idx_logits], dim=1)

    return b_logits


class BroadFineModelLM(LightningModule):
    model: DualViT

    def __init__(
        self,
        model: DualViT,
        num_fine_outputs,
        num_broad_outputs,
        lr,
        loss_mode,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_fine_outputs = num_fine_outputs
        self.num_broad_outputs = num_broad_outputs
        self.loss_mode = loss_mode if loss_mode is not None else BELMode.M3M

        self.save_hyperparameters(ignore=["model"])

        self.ce_loss = nn.CrossEntropyLoss()
        self.emb_loss = BroadFineEmbeddingLoss(num_broad_classes=self.num_broad_outputs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        pixel_values, fine_labels, broad_labels = batch
        # [B, 1, D], [B, 1, D], [B, C_fine]
        broad_embedding, fine_embedding, fine_logits = self(pixel_values)

        loss_emb = self.get_embedding_loss(
            broad_embedding,
            broad_labels,
            fine_embedding,
            fine_labels,
        )

        loss_fine_ce = self.ce_loss(fine_logits, fine_labels)

        self.log("tE", loss_emb, prog_bar=True)
        self.log("tCE", loss_fine_ce, prog_bar=True)

        return loss_emb + loss_fine_ce

    def _mdl_outputs(self, pixel_values):
        broad_embedding, fine_embedding, fine_logits = self(pixel_values)
        return broad_embedding, fine_embedding, fine_logits

    def get_embedding_loss(
        self,
        broad_embedding,
        broad_labels,
        fine_embedding,
        fine_labels,
    ):
        if (self.current_epoch + 1) % 2 == 0:
            # Broad Train
            fine_embedding = fine_embedding.detach().clone()
            loss_emb = self._compute_emb_loss(
                broad_embedding, broad_labels, fine_embedding, fine_labels
            )
        else:
            # Fine Train
            broad_embedding = broad_embedding.detach().clone()
            loss_emb = self._compute_emb_loss(
                broad_embedding, broad_labels, fine_embedding, fine_labels
            )

        loss_emb = torch.log10(loss_emb)
        return loss_emb

    def _compute_emb_loss(
        self, broad_embedding, broad_labels, fine_embedding, fine_labels
    ):
        loss_emb = torch.mean(
            torch.stack(
                self.emb_loss(
                    broad_embedding,
                    broad_labels,
                    fine_embedding,
                    fine_labels,
                    mode=self.loss_mode,
                )
            )
        )
        return loss_emb

    def evaluate(self, batch, stage=None):
        # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
        pixel_values, fine_labels, broad_labels = batch
        broad_embedding, fine_embedding, fine_logits = self._mdl_outputs(pixel_values)
        loss_emb = self._compute_emb_loss(
            broad_embedding, broad_labels, fine_embedding, fine_labels
        )
        loss_fine_ce = self.ce_loss(fine_logits, fine_labels)

        acc_fine = self.get_fine_acc(fine_labels, fine_logits)

        acc_broad, loss_broad_ce = self.get_broad_statistics_via_fine(
            broad_embedding,
            broad_labels,
            fine_labels,
        )

        if stage:
            # Fine
            self.log(f"{stage}CEF", loss_fine_ce, prog_bar=True)
            self.log(
                VALIDATION_METRIC_NAME if stage == "val" else f"{stage}_af",
                acc_fine,
                prog_bar=True,
            )
            # Embedding
            self.log(f"{stage}E", loss_emb, prog_bar=True)
            # Broad
            # self.log(f"{stage}_ceb", loss_broad_ce, prog_bar=True)
            self.log(f"{stage}AB", acc_broad, prog_bar=True)

    def get_broad_statistics_via_fine(self, broad_embedding, broad_labels, fine_labels):
        # Broad Class
        f_logits_b = self.model.to_logits(broad_embedding)
        if isinstance(f_logits_b, (list, tuple)):
            f_logits_b = f_logits_b[-1]

        b_logits = convert_fine_to_broad_logits(
            f_logits_b, broad_labels, fine_labels, num_broad=self.num_broad_outputs
        )
        preds = torch.argmax(b_logits, dim=1)
        acc_broad = accuracy(
            preds, broad_labels, task="multiclass", num_classes=self.num_broad_outputs
        )
        loss_broad_ce = self.ce_loss(b_logits, broad_labels)
        return acc_broad, loss_broad_ce

    def get_fine_acc(self, fine_labels, fine_logits):
        # Fine Class
        preds = torch.argmax(fine_logits, dim=1)
        acc_fine = accuracy(
            preds, fine_labels, task="multiclass", num_classes=self.num_fine_outputs
        )
        return acc_fine

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.get_param_groups(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": VALIDATION_METRIC_NAME,
        }

    def get_param_groups(self):
        return [
            {"params": self.model.embeddings.parameters(), "lr": 1e-5},
            {"params": self.model.broad_encoders.parameters(), "lr": 1e-5},
            {"params": self.model.fine_encoders.parameters(), "lr": 1e-3},
            {"params": self.model.mlp_heads.parameters(), "lr": 1e-3},
        ]
