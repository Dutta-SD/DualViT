import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from transformers import logging
from dualvit.lightning.modulev2 import VALIDATION_METRIC_NAME

from dualvit.factory import TPModelFactory
from dualvit.constants import IMG_SIZE, LEARNING_RATE
from dualvit.lightning.data.cifar import CIFAR10MultiLabelDataModule
from dualvit.lightning.data.common import train_transform, test_transform
from dualvit.lightning.loss import BELMode
from dualvit.lightning.modulev2 import BroadFineModelLM
from pytorch_lightning.callbacks import ModelCheckpoint


logging.set_verbosity_warning()

warnings.filterwarnings("ignore")

# Data Module
# is_test = True then subset, else Full Dataset
datamodule = CIFAR10MultiLabelDataModule(
    is_test=False,
    train_transform=train_transform,
    val_transform=test_transform,
)

datamodule.prepare_data()
datamodule.setup()


LOAD_CKPT = False

CKPT_PATH = "logs/cifar10/tpdualvit-p16-224/lightning_logs/version_1/checkpoints/modelepoch=31-val_af=0.957.ckpt"

NUM_FINE_CLASSES = 10
NUM_BROAD_CLASSES = 2
l_module = BroadFineModelLM(
    model=TPModelFactory.get_model("CIFAR10"),
    num_fine_outputs=NUM_FINE_CLASSES,
    num_broad_outputs=NUM_BROAD_CLASSES,
    lr=LEARNING_RATE,
    loss_mode=BELMode.CLUSTER,
)

checkpoint_callback = ModelCheckpoint(
    monitor=VALIDATION_METRIC_NAME,  # Monitor the validation loss
    filename="model" + "{epoch:02d}" + f"-{{{VALIDATION_METRIC_NAME}:.3f}}",
    save_top_k=2,
    mode="max",  # 'max' -> More is monitor, the better
)


kwargs = {
    "max_epochs": 100,
    "accelerator": "gpu",
    "gpus": 1,
    "logger": CSVLogger(save_dir=f"logs/cifar10/tpdualvit-p16-{IMG_SIZE}"),
    "deterministic": True,
    "callbacks": [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback,
    ],
    "num_sanity_val_steps": 5,
    "gradient_clip_val": 1,
}

if LOAD_CKPT:
    kwargs = {**kwargs, "resume_from_checkpoint": CKPT_PATH}

trainer = Trainer(**kwargs)

if __name__ == "__main__":
    if LOAD_CKPT:
        # Ensure test dataset is defined else comment out
        trainer.test(l_module, datamodule=datamodule, ckpt_path=CKPT_PATH)

    trainer.fit(l_module, datamodule=datamodule)

    # Ensure test dataset is defined else comment out
    trainer.test(l_module, datamodule=datamodule)
