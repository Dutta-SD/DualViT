import json
from functools import lru_cache
from typing import Any

import torch
from lightning import seed_everything
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torchvision.datasets import ImageNet

from vish.lightning.data.common import PATH_DATASETS, NUM_WORKERS

# Need different value for this in case of ImageNet
BATCH_SIZE = 16 if torch.cuda.is_available() else 4


class ImageNet1kMultilabelDataset(ImageNet):
    def __init__(self, is_test: bool, depth: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_test = is_test
        self.depth = depth

        f = self.read_label_json()
        self._set_and_check_depth(depth, f)

        self.label_tree = f["label_tree"]
        self.nodes_at_depth = f["depthwise_nodes"][str(depth)]
        self.num_broad_classes = len(self.nodes_at_depth)

    def _set_and_check_depth(self, depth, f):
        self.max_allowable_depth = int(f["meta"]["min_depth"])
        if depth >= self.max_allowable_depth:
            raise ValueError(f"{depth} is not < {self.max_allowable_depth}")

    @staticmethod
    def read_label_json():
        with open("imagenet.json") as fp:
            f = json.load(fp)
        return f

    @lru_cache(maxsize=1100)
    def get_broad_label(self, fine_label: int):
        label = self._compute_broad(fine_label)
        return self.nodes_at_depth[label]

    def _compute_broad(self, fine_label):
        fl_str = str(fine_label)
        labels_set = set(self.label_tree[fl_str])
        depth_nodes = set(self.nodes_at_depth.keys())
        broad_label = labels_set.intersection(depth_nodes)
        if len(broad_label) > 1:
            raise ValueError(f"{broad_label} has more than 1 labels")
        l, *_ = broad_label
        return l

    def __len__(self):
        if self.is_test:
            # more images for Imagenet 1k
            return 32768 if self.split else 16384
        return super().__len__()

    def __getitem__(self, index: int) -> tuple[Any, Any, int | list[int]]:
        img_tensor, fine_label = super().__getitem__(index)
        return img_tensor, fine_label, self.get_broad_label(fine_label)


class ImageNet1kMultiLabelDataModule(LightningDataModule):
    """
    ImageNet 1k data module with broad and fine labels
    """

    def __init__(
        self,
        is_test,
        depth,
        train_transform,
        test_transform,
    ):
        super().__init__()
        self._test = None
        self._val = None
        self._train = None
        self.is_test = is_test
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.depth = depth
        self.data_dir = PATH_DATASETS
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS
        self.num_broad_classes=None

    def prepare_data(self):
        # Download data
        ImageNet1kMultilabelDataset(
            self.is_test, depth=self.depth, root=self.data_dir, train=True, download=True,
        )
        ImageNet1kMultilabelDataset(
            self.is_test, depth=self.depth, root=self.data_dir, train=False, download=True,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = ImageNet1kMultilabelDataset(
                self.is_test,
                depth=self.depth,
                root=self.data_dir,
                train=True,
                transform=self.train_transform,
            )
            self.num_broad_classes = full.num_broad_classes

            # use 10% of training data for validation
            train_set_size = int(len(full) * 0.9)
            valid_set_size = len(full) - train_set_size

            seed_everything(42)

            self._train, self._val = random_split(
                full, [train_set_size, valid_set_size]
            )

        if stage == "test" or stage is None:
            self._test = ImageNet1kMultilabelDataset(
                self.is_test,
                depth=self.depth,
                root=self.data_dir,
                train=False,
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self._test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
