import os

import torch
from torchvision.transforms import transforms as tf

from vish.constants import IMG_SIZE

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")
BATCH_SIZE = 4
NUM_WORKERS = int(os.cpu_count()) // 2

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
