import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import logging

from vish.constants import LEARNING_RATE, VIT_PRETRAINED_MODEL_2
from vish.lightning.data import (
    CIFAR100MultiLabelDataModule,
    test_transform,
    train_transform,
)
from vish.lightning.module import (
    TPDualVitLightningModuleCifar100,
)
from vish.lightning.utils import checkpoint_callback

logging.set_verbosity_warning()

warnings.filterwarnings("ignore")

# Data Module
datamodule = CIFAR100MultiLabelDataModule(
    is_test=True,
    train_transform=train_transform,
    val_transform=test_transform,
)

datamodule.prepare_data()
datamodule.setup()

# Model Define
# Fine output ok
model = TPDualVitLightningModuleCifar100(
    wt_name=VIT_PRETRAINED_MODEL_2,
    num_fine_outputs=100,
    num_broad_outputs=20,
    lr=LEARNING_RATE,
)

LOAD_CKPT = False

CKPT_PATH = ""

if LOAD_CKPT:
    # Load from checkpoint
    checkpoint = torch.load(CKPT_PATH)
    model = TPDualVitLightningModuleCifar100.load_from_checkpoint(CKPT_PATH)


kwargs = {
    "max_epochs": 300,
    "accelerator": "auto",
    "devices": 1,
    "logger": CSVLogger(save_dir="logs/dual_broad_pretrained_cifar100"),
    "callbacks": [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback,
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
    if LOAD_CKPT:
        trainer.test(model, datamodule=datamodule)
        
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
