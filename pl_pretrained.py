import warnings

import torch
from pytorch_lightning import (
    Trainer,
)
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from vish.constants import LOAD_CKPT, EPOCHS, LEARNING_RATE, VIT_PRETRAINED_MODEL_2
from vish.lightning.data import (
    CIFAR10MultiLabelDataModule,
    train_transform,
    test_transform,
)
from vish.lightning.module import SplitVitModule, PreTrainedSplitHierarchicalViTModule
from vish.lightning.utils import checkpoint_callback, early_stopping_callback

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

if LOAD_CKPT:
    # Load from checkpoint
    checkpoint_path = "NAN"
    checkpoint = torch.load(checkpoint_path)
    model = SplitVitModule.load_from_checkpoint(checkpoint_path)


# Trainer
trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    devices=1,
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback,
        early_stopping_callback,
    ],
    num_sanity_val_steps=2,
    gradient_clip_val=1,
)


if __name__ == "__main__":
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
