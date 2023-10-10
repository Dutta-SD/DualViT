import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import logging

from tp_model import TP_MODEL_MODIFIED_CIFAR10
from vish.constants import LEARNING_RATE
from vish.lightning.data import (
    CIFAR10MultiLabelDataModule,
    test_transform,
    train_transform,
)
from vish.lightning.loss import BELMode
from vish.lightning.modulev2 import BroadFineModelLM
from vish.lightning.utils import checkpoint_callback

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


LOAD_CKPT = False

CKPT_PATH = ""

l_module = BroadFineModelLM(
    model=TP_MODEL_MODIFIED_CIFAR10,
    num_fine_outputs=10,
    num_broad_outputs=2,
    lr=LEARNING_RATE,
    loss_mode=BELMode.CLUSTER,
)


kwargs = {
    "max_epochs": 300,
    "accelerator": "gpu",
    "gpus": 1,
    "logger": CSVLogger(save_dir="logs/cifar10/modified_dual_tpvit_fulldataset"),
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
        "resume_from_checkpoint": CKPT_PATH
    }

trainer = Trainer(**kwargs)

if __name__ == "__main__":
    if LOAD_CKPT:
        trainer.test(l_module, datamodule=datamodule, ckpt_path=CKPT_PATH)

    trainer.fit(l_module, datamodule=datamodule)
    trainer.test(l_module, datamodule=datamodule)
