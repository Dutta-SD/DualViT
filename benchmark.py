import time

import torch
from transformers import ViTConfig

from vish.model.tp.single_split import SplitHierarchicalTPViT

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from vish.utils import DEVICE

model = SplitHierarchicalTPViT(ViTConfig()).to(DEVICE)


def time_model(fxn, exp_name, cuda):
    times = []
    for _ in range(10):
        if cuda:
            torch.cuda.synchronize()
        rand_tensor = torch.randn(8, 3, 224, 224, device=DEVICE)

        start_epoch = time.time()

        _ = fxn(rand_tensor)

        if cuda:
            torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)
    print(f"{exp_name} time needed: ", sum(times) / 10)


time_model(model.forward, "Normal Forward", True)
time_model(model.forward2, "Forward 2", True)
