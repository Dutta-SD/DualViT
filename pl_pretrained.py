import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import logging

from vish.constants import EPOCHS, LEARNING_RATE, LOAD_CKPT, VIT_PRETRAINED_MODEL_2
from vish.lightning.data import (
    CIFAR10MultiLabelDataModule,
    test_transform,
    train_transform,
)
from vish.lightning.utils import early_stopping_callback
from vish.lightning.module import PreTrainedSplitHierarchicalViTModule

logging.set_verbosity_warning()

warnings.filterwarnings("ignore")

# Data Module
datamodule = CIFAR10MultiLabelDataModule(
    is_test=False,
    train_transform=train_transform,
    val_transform=test_transform,
)

datamodule.prepare_data()
datamodule.setup()

# Model Define
# Fine output ok
model = PreTrainedSplitHierarchicalViTModule(
    wt_name=VIT_PRETRAINED_MODEL_2,
    num_fine_outputs=10,
    num_broad_outputs=2,
    lr=LEARNING_RATE,
)

LOAD_CKPT = False

checkpoint_path = "logs/pretrained_split_vit/lightning_logs/version_14/checkpoints/epoch=51-step=160888.ckpt"

if LOAD_CKPT:
    # Load from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model = PreTrainedSplitHierarchicalViTModule.load_from_checkpoint(checkpoint_path)
    model.lr = 1e-4
    model.save_hyperparameters()


# Trainer
trainer = Trainer(
    max_epochs=200,
    accelerator="auto",
    devices=1,
    logger=CSVLogger(save_dir="logs/pretrained_split_vit"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        # checkpoint_callback,
        # early_stopping_callback,
    ],
    num_sanity_val_steps=2,
    gradient_clip_val=1,
    resume_from_checkpoint=checkpoint_path,
)


if __name__ == "__main__":
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
