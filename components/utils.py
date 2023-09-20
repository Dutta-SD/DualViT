import torch
import os
import numpy as np
import random
from typing import Any, Tuple
from torchvision.datasets import CIFAR10
from constants import *
import torchvision.transforms as tf
from torch.utils.data import DataLoader


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_broad_label_cifar10(idx):
    fine_to_broad = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    return fine_to_broad[idx]


class CIFAR10MultiLabelDataset(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        # return 80 if self.train else 20
        return super().__len__()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_tensor, fine_label = super().__getitem__(index)
        return img_tensor, fine_label, get_broad_label_cifar10(fine_label)


DEVICE = get_default_device()

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

train_dataset = CIFAR10MultiLabelDataset(
    "./data", download=True, transform=train_transform
)
test_dataset = CIFAR10MultiLabelDataset(
    "./data", download=True, transform=test_transform, train=False
)

set_seed(42)

train_dl = DataLoader(
    train_dataset,
    BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

test_dl = DataLoader(
    test_dataset,
    BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

train_dl, test_dl = (
    DeviceDataLoader(train_dl, DEVICE),
    DeviceDataLoader(test_dl, DEVICE),
)
