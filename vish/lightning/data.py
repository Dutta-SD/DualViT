import os
from typing import Any

import torch
from lightning_fabric import seed_everything
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as tf

from vish.constants import IMG_SIZE

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")
BATCH_SIZE = 32 if torch.cuda.is_available() else 4
NUM_WORKERS = int(os.cpu_count() / 2)


class CIFAR10MultiLabelDataset(CIFAR10):
    def __init__(self, is_test: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fine_to_broad = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        self.is_test = is_test

    def _broad(self, idx):
        return self.fine_to_broad[idx]

    def __len__(self):
        if self.is_test:
            return 8192 if self.train else 4096
        return super().__len__()

    def __getitem__(self, index: int) -> tuple[Any, Any, int | list[int]]:
        img_tensor, fine_label = super().__getitem__(index)
        return img_tensor, fine_label, self._broad(fine_label)


class CIFAR10MultiLabelDataModule(LightningDataModule):
    def __init__(
        self,
        is_test,
        train_transform,
        val_transform,
    ):
        super().__init__()
        self.is_test = is_test
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.data_dir = PATH_DATASETS
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS

    def prepare_data(self):
        # Download data
        CIFAR10MultiLabelDataset(
            self.is_test, root=self.data_dir, train=True, download=True
        )
        CIFAR10MultiLabelDataset(
            self.is_test, root=self.data_dir, train=False, download=True
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10MultiLabelDataset(
                self.is_test,
                root=self.data_dir,
                train=True,
                transform=self.train_transform,
            )

            # use 20% of training data for validation
            train_set_size = int(len(cifar_full) * 0.8)
            valid_set_size = len(cifar_full) - train_set_size

            seed_everything(42)

            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [train_set_size, valid_set_size]
            )

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10MultiLabelDataset(
                self.is_test,
                root=self.data_dir,
                train=False,
                transform=self.val_transform,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


train_transform = tf.Compose(
    [
        tf.PILToTensor(),
        # tf.AutoAugment(tf.AutoAugmentPolicy.CIFAR10),
        tf.Resize(IMG_SIZE, antialias=True),
        tf.RandomHorizontalFlip(0.5),
        tf.RandomVerticalFlip(0.5),
        # tf.RandomResizedCrop(),
        tf.ConvertImageDtype(torch.float32),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = tf.Compose(
    [
        tf.PILToTensor(),
        tf.Resize(IMG_SIZE, antialias=True),
        tf.ConvertImageDtype(torch.float32),
        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)