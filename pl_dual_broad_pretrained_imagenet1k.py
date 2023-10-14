import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import logging

from tp_model import TP_MODEL_MODIFIED_IMAGENET1K
from vish.constants import LEARNING_RATE
from vish.lightning.data.common import train_transform, test_transform
from vish.lightning.data.imagenet import ImageNet1kMultiLabelDataModule
from vish.lightning.loss import BELMode
from vish.lightning.modulev2 import BroadFineModelLM

logging.set_verbosity_warning()

warnings.filterwarnings("ignore")

# Data Module
datamodule = ImageNet1kMultiLabelDataModule(
    is_test=False,
    depth=2,
    train_transform=train_transform,
    test_transform=test_transform,
)

datamodule.prepare_data()
datamodule.setup()


LOAD_CKPT = False

CKPT_PATH = ""

l_module = BroadFineModelLM(
    model=TP_MODEL_MODIFIED_IMAGENET1K,
    num_fine_outputs=1000,
    num_broad_outputs=datamodule.num_broad_classes,
    lr=LEARNING_RATE,
    loss_mode=BELMode.CLUSTER,
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc_fine",
    filename="tpdualvit-imagenet1k-{epoch:02d}-{val_acc_fine:.3f}",
    save_top_k=2,
    mode="max",  
)


kwargs = {
    "max_epochs": 300,
    "accelerator": "gpu",
    "gpus": 1,
    "logger": CSVLogger(save_dir="logs/imagenet1k/modified_dual_tpvit_fulldataset"),
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
