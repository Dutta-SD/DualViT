import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torchmetrics.functional import accuracy
from transformers import ViTForImageClassification, ViTConfig
from transformers.utils import logging

from vish.lightning.data.imagenet import ImageNet1kMultiLabelDataModule

logging.set_verbosity_warning()
warnings.filterwarnings("ignore")
pl.seed_everything(42, workers=True)

# Data Module
datamodule = ImageNet1kMultiLabelDataModule(
    is_test=True,
    depth=2,
)
datamodule.prepare_data()
datamodule.setup()


class VitImageNetTrainer(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.configuration = ViTConfig()
        self.num_classes = 1000
        self.configuration.num_labels = self.num_classes
        self.model = ViTForImageClassification(self.configuration)
        self.save_hyperparameters(ignore=["model"])
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)["logits"]

    def training_step(self, batch, batch_idx):
        pixel_values, fine_labels, _ = batch
        fine_logits = self(pixel_values)
        loss_fine_ce = self.ce_loss(fine_logits, fine_labels)
        self.log("train_loss_ce", loss_fine_ce)
        return loss_fine_ce

    def get_fine_acc(self, fine_labels, fine_logits):
        preds = torch.argmax(fine_logits, dim=1)
        acc_fine = accuracy(
            preds,
            fine_labels,
            task="multiclass",
            num_classes=self.num_classes,
        )
        return acc_fine

    def evaluate(self, batch, stage=None):
        pixel_values, fine_labels, broad_labels = batch
        fine_logits = self(pixel_values)
        loss_fine_ce = self.ce_loss(fine_logits, fine_labels)
        acc_fine = self.get_fine_acc(fine_labels, fine_logits)

        if stage:
            self.log(f"{stage}_ce_f", loss_fine_ce, prog_bar=True)
            self.log(f"{stage}_acc_f", acc_fine, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.get_param_groups(),
            lr=0.1,
            weight_decay=1e-5,
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
            "monitor": "val_ce_f",
        }

    def get_param_groups(self):
        return [
            {"params": self.model.parameters(), "lr": 0.1},
        ]


l_module = VitImageNetTrainer()

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc_f",
    filename="vit-from-scratch-imagenet-subset{epoch:02d}-{val_acc_fine:.3f}",
    save_top_k=2,
    mode="max",
)

LOAD_CKPT = False

kwargs = {
    "max_epochs": 300,
    "accelerator": "gpu",
    "gpus": 1,
    "logger": CSVLogger(save_dir="logs/imagenet1k/vit_scratch"),
    "callbacks": [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback,
    ],
    "num_sanity_val_steps": 2,
    "gradient_clip_val": 1,
}

trainer = Trainer(**kwargs)

if __name__ == "__main__":
    ckpt_path = None
    if LOAD_CKPT:
        ckpt_path = "<FILL PATH>"

    trainer.fit(l_module, datamodule=datamodule, ckpt_path=ckpt_path)
