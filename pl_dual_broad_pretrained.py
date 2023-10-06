import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import logging

from vish.constants import LEARNING_RATE, VIT_PRETRAINED_MODEL_2
from vish.lightning.data import (
    CIFAR10MultiLabelDataModule,
    test_transform,
    train_transform,
)
from vish.lightning.module import (
    DualVitSemiPretrainedLightningModule,
)

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
model = DualVitSemiPretrainedLightningModule(
    wt_name=VIT_PRETRAINED_MODEL_2,
    num_fine_outputs=10,
    num_broad_outputs=2,
    lr=LEARNING_RATE,
)

LOAD_CKPT = False

CKPT_PATH = ""

if LOAD_CKPT:
    # Load from checkpoint
    checkpoint = torch.load(CKPT_PATH)
    model = DualVitSemiPretrainedLightningModule.load_from_checkpoint(CKPT_PATH)
    model.lr = 1e-4
    model.save_hyperparameters()


kwargs = {
    "max_epochs": 200,
    "accelerator": "auto",
    "devices": 1,
    "logger": CSVLogger(save_dir="logs/dual_broad_pretrained"),
    "callbacks": [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ],
    "num_sanity_val_steps": 5,
    "gradient_clip_val": 1,
}

if LOAD_CKPT:
    kwargs = {
        **kwargs,
        "resume_from_checkpoint": CKPT_PATH,
    }

trainer = Trainer(**kwargs)

if __name__ == "__main__":
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
