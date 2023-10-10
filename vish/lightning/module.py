# import torch
# from pytorch_lightning import LightningModule
# from torch import nn
# from torchmetrics.functional import accuracy
# from transformers import ViTConfig

# from tp_model import TP_MODEL, TP_MODEL_CIFAR100
# from vish.constants import LEARNING_RATE, WEIGHT_DECAY
# from vish.lightning.loss import BroadFineEmbeddingLoss
# from vish.lightning.model import (
#     SplitHierarchicalTPViT,
#     SplitViTHierarchicalTPVitHalfPretrained,
#     DualVitSemiPretrained,
# )
# from vish.model.common.loss import convert_fine_to_broad_logits
# from vish.model.tp.dual import TPDualVit


# class SplitVitModule(LightningModule):
#     def __init__(self, lr=0.05):
#         super().__init__()

#         self.lr = lr
#         self.save_hyperparameters()
#         self.model = SplitHierarchicalTPViT(
#             ViTConfig(), num_broad_outputs=2, num_fine_outputs=10
#         )
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.emb_loss = BroadFineEmbeddingLoss(num_broad_classes=2)

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
#         pixel_values, fine_labels, broad_labels = batch
#         broad_embedding, fine_embedding, broad_logits, fine_logits = self(pixel_values)

#         loss_emb = self.get_embedding_loss(
#             broad_embedding, broad_labels, fine_embedding, fine_labels
#         )
#         loss_broad_ce = self.ce_loss(broad_logits, broad_labels)
#         loss_fine_ce = self.ce_loss(fine_logits, fine_labels)

#         self.log("train_loss_emb", loss_emb)
#         self.log("train_loss_fine_ce", loss_fine_ce)
#         self.log("train_loss_broad_ce", loss_broad_ce)

#         return loss_emb + loss_broad_ce + loss_fine_ce

#     def get_embedding_loss(
#         self, broad_embedding, broad_labels, fine_embedding, fine_labels
#     ):
#         if (self.current_epoch + 1) % 2 == 0:
#             # Broad Train
#             fine_embedding = fine_embedding.detach().clone()
#             loss_emb = self._compute_emb_loss(
#                 broad_embedding, broad_labels, fine_embedding, fine_labels
#             )
#             # scale = 1e5
#         else:
#             # Fine Train With Scaling
#             broad_embedding = broad_embedding.detach().clone()
#             loss_emb = self._compute_emb_loss(
#                 broad_embedding, broad_labels, fine_embedding, fine_labels
#             )
#             # scale = 1e5
#         loss_emb = torch.log10(loss_emb + 1)
#         return loss_emb

#     def _compute_emb_loss(
#         self, broad_embedding, broad_labels, fine_embedding, fine_labels
#     ):
#         loss_emb = torch.mean(
#             torch.stack(
#                 self.emb_loss(
#                     broad_embedding, broad_labels, fine_embedding, fine_labels
#                 )
#             )
#         )
#         return loss_emb

#     def evaluate(self, batch, stage=None):
#         # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
#         pixel_values, fine_labels, broad_labels = batch
#         broad_embedding, fine_embedding, broad_logits, fine_logits = self(pixel_values)

#         loss_emb = self.get_embedding_loss(
#             broad_embedding, broad_labels, fine_embedding, fine_labels
#         )
#         loss_broad_ce = self.ce_loss(broad_logits, broad_labels)
#         loss_fine_ce = self.ce_loss(fine_logits, fine_labels)

#         # Broad Class
#         preds = torch.argmax(broad_logits, dim=1)
#         acc_broad = accuracy(preds, broad_labels, task="binary")

#         # Fine Class
#         preds = torch.argmax(fine_logits, dim=1)
#         acc_fine = accuracy(preds, fine_labels, task="multiclass", num_classes=10)

#         if stage:
#             self.log(f"{stage}_loss_fine", loss_fine_ce, prog_bar=True)
#             self.log(f"{stage}_acc_fine", acc_fine, prog_bar=True)
#             self.log(f"{stage}_loss_broad", loss_broad_ce, prog_bar=True)
#             self.log(f"{stage}_acc_broad", acc_broad, prog_bar=True)
#             self.log(f"{stage}_loss_emb", loss_emb, prog_bar=True)

#     def validation_step(self, batch, batch_idx):
#         self.evaluate(batch, "val")

#     def test_step(self, batch, batch_idx):
#         self.evaluate(batch, "test")

#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(
#             self.parameters(),
#             lr=self.hparams.lr,
#             momentum=0.9,
#             weight_decay=WEIGHT_DECAY,
#         )
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode="max",
#             verbose=True,
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": lr_scheduler,
#             "monitor": "val_acc_fine",  # Monitor the 'train_loss' metric
#         }


# class PreTrainedSplitHierarchicalViTModule(SplitVitModule):
#     def __init__(self, wt_name: str, num_fine_outputs, num_broad_outputs, lr=0.05):
#         super().__init__()

#         self.lr = lr
#         self.wt_name = wt_name
#         self.save_hyperparameters()
#         self.model = SplitViTHierarchicalTPVitHalfPretrained.from_pretrained(wt_name)
#         self.model.init_broad_fine(num_fine_outputs)

#         self.ce_loss = nn.CrossEntropyLoss()
#         self.emb_loss = BroadFineEmbeddingLoss(num_broad_classes=num_broad_outputs)

#     def forward(self, x):
#         # pixel, broad_clf, fine_clf
#         return self.model(x, False, True)

#     def training_step(self, batch, batch_idx):
#         # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
#         pixel_values, fine_labels, broad_labels = batch
#         broad_embedding, fine_embedding, fine_logits = self._mdl_outputs(pixel_values)

#         loss_emb = self.get_embedding_loss(
#             broad_embedding, broad_labels, fine_embedding, fine_labels
#         )
#         loss_fine_ce = self.ce_loss(fine_logits, fine_labels)

#         self.log("train_loss_emb", loss_emb)
#         self.log("train_loss_fine_ce", loss_fine_ce)

#         return loss_emb + loss_fine_ce

#     def _mdl_outputs(self, pixel_values):
#         broad_embedding, fine_embedding, _, fine_logits = self(pixel_values)
#         return broad_embedding, fine_embedding, fine_logits

#     def evaluate(self, batch, stage=None):
#         # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
#         pixel_values, fine_labels, broad_labels = batch
#         broad_embedding, fine_embedding, fine_logits = self._mdl_outputs(pixel_values)
#         loss_emb = self._compute_emb_loss(
#             broad_embedding, broad_labels, fine_embedding, fine_labels
#         )
#         loss_fine_ce = self.ce_loss(fine_logits, fine_labels)

#         # Fine Class
#         preds = torch.argmax(fine_logits, dim=1)
#         acc_fine = accuracy(preds, fine_labels, task="multiclass", num_classes=10)

#         # Broad Class
#         f_logits_b = self.model.classify_fine(broad_embedding)
#         b_logits = convert_fine_to_broad_logits(f_logits_b, broad_labels, fine_labels)
#         preds = torch.argmax(b_logits, dim=1)
#         acc_broad = accuracy(preds, broad_labels, task="binary")
#         loss_broad_ce = self.ce_loss(b_logits, broad_labels)

#         if stage:
#             # Fine
#             self.log(f"{stage}_loss_fine", loss_fine_ce, prog_bar=True)
#             self.log(f"{stage}_acc_fine", acc_fine, prog_bar=True)
#             # Embedding
#             self.log(f"{stage}_loss_emb", loss_emb, prog_bar=True)
#             # Broad
#             self.log(f"{stage}_loss_broad", loss_broad_ce, prog_bar=True)
#             self.log(f"{stage}_acc_broad", acc_broad, prog_bar=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(
#             self.model.get_large_lr_params(self.hparams.lr)
#             + self.model.get_small_lr_parameters(5e-6),
#             momentum=0.9,
#             weight_decay=WEIGHT_DECAY,
#         )
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode="max", verbose=True, patience=4
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": lr_scheduler,
#             "monitor": "val_acc_fine",  # Monitor the 'train_loss' metric
#         }


# class DualVitSemiPretrainedLightningModule(PreTrainedSplitHierarchicalViTModule):
#     """
#     Broad Pretrained, Trained using Embedding Loss
#     Fine from scratch, Trained using Cross Entropy and Embedding Loss
#     HP from Broad to Fine
#     """

#     def __init__(self, wt_name: str, num_fine_outputs, num_broad_outputs, lr=0.05):
#         super().__init__(wt_name, num_fine_outputs, num_broad_outputs, lr)
#         self.model = DualVitSemiPretrained.from_pretrained(wt_name)
#         self.model.init_broad_fine(num_fine_outputs)


# class TPDualVitLightningModule(PreTrainedSplitHierarchicalViTModule):
#     """
#     Broad Pretrained, Trained using Embedding Loss
#     Fine from scratch, Trained using Cross Entropy and Embedding Loss
#     HP from Broad to Fine
#     """

#     def __init__(self, wt_name: str, num_fine_outputs, num_broad_outputs, lr=0.05):
#         super().__init__(wt_name, num_fine_outputs, num_broad_outputs, lr)
#         self.model: TPDualVit = TP_MODEL
#         self.num_fine_outputs = num_fine_outputs
#         self.num_broad_outputs = num_broad_outputs

#     def _mdl_outputs(self, pixel_values):
#         [broad_embedding, _], [fine_embedding, fine_logits] = self(pixel_values)
#         fine_logits = fine_logits[0]
#         return broad_embedding, fine_embedding, fine_logits

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
#         pixel_values, fine_labels, broad_labels = batch
#         broad_embedding, fine_embedding, fine_logits = self._mdl_outputs(pixel_values)

#         loss_emb = self.get_embedding_loss(
#             broad_embedding, broad_labels, fine_embedding, fine_labels
#         )
#         loss_fine_ce = self.ce_loss(fine_logits, fine_labels)

#         self.log("train_loss_emb", loss_emb)
#         self.log("train_loss_fine_ce", loss_fine_ce)

#         return loss_emb + loss_fine_ce

#     def evaluate(self, batch, stage=None):
#         # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
#         pixel_values, fine_labels, broad_labels = batch
#         broad_embedding, fine_embedding, fine_logits = self._mdl_outputs(pixel_values)
#         loss_emb = self._compute_emb_loss(
#             broad_embedding, broad_labels, fine_embedding, fine_labels
#         )
#         loss_fine_ce = self.ce_loss(fine_logits, fine_labels)

#         # Fine Class
#         preds = torch.argmax(fine_logits, dim=1)
#         acc_fine = accuracy(
#             preds, fine_labels, task="multiclass", num_classes=self.num_fine_outputs
#         )

#         # Broad Class
#         f_logits_b = self.model.fine_model.to_logits(broad_embedding)[0]
#         b_logits = convert_fine_to_broad_logits(
#             f_logits_b, broad_labels, fine_labels, self.num_broad_outputs
#         )
#         preds = torch.argmax(b_logits, dim=1)
#         acc_broad = accuracy(
#             preds, broad_labels, task="multiclass", num_classes=self.num_broad_outputs
#         )
#         loss_broad_ce = self.ce_loss(b_logits, broad_labels)

#         if stage:
#             # Fine
#             self.log(f"{stage}_loss_fine", loss_fine_ce, prog_bar=True)
#             self.log(f"{stage}_acc_fine", acc_fine, prog_bar=True)
#             # Embedding
#             self.log(f"{stage}_loss_emb", loss_emb, prog_bar=True)
#             # Broad
#             self.log(f"{stage}_loss_broad", loss_broad_ce, prog_bar=True)
#             self.log(f"{stage}_acc_broad", acc_broad, prog_bar=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(
#             [
#                 {"params": self.model.fine_model.parameters(), "lr": 1e-3},
#                 {"params": self.model.broad_model.parameters(), "lr": 5e-6},
#             ],
#             lr=LEARNING_RATE,
#             weight_decay=WEIGHT_DECAY,
#             momentum=0.99,
#             nesterov=True,
#         )
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode="max",
#             verbose=True,
#             patience=5,
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": lr_scheduler,
#             "monitor": "val_acc_fine",  # Monitor the 'train_loss' metric
#         }


# class TPDualVitLightningModuleCifar100(TPDualVitLightningModule):
#     def __init__(self, wt_name: str, num_fine_outputs, num_broad_outputs, lr=0.05):
#         super().__init__(wt_name, num_fine_outputs, num_broad_outputs, lr)
#         self.model: TPDualVit = TP_MODEL_CIFAR100
#         self.num_fine_outputs = num_fine_outputs
#         self.num_broad_outputs = num_broad_outputs

#     # def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
#     #     _, fine_labels, broad_labels = batch
#     #     print(torch.unique(fine_labels))
#     #     print(torch.unique(broad_labels))
#     #     return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(
#             [
#                 {"params": self.model.fine_model.parameters(), "lr": 1e-2},
#                 {"params": self.model.broad_model.parameters(), "lr": 1e-5},
#             ],
#             lr=LEARNING_RATE,
#             weight_decay=WEIGHT_DECAY,
#             momentum=0.99,
#             nesterov=True,
#         )
#         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode="max",
#             verbose=True,
#             patience=5,
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": lr_scheduler,
#             "monitor": "val_acc_fine",  # Monitor the 'train_loss' metric
#         }
