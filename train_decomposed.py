import sys
from datetime import datetime

import torch

from vish.constants import (
    DATE_TIME_FORMAT,
    EPOCHS,
    IS_TEST_RUN,
    LEARNING_RATE,
    LOAD_CKPT,
    MIN_LR,
    MOMENTUM,
    WEIGHT_DECAY,
    WEIGHT_FOLDER_PATH,
)
from vish.model.common.tree import LabelHierarchyTree
from vish.model.decomposed.decomposed import VitClassificationDecomposed
from vish.trainer.decomposed import VitDecomposedTrainer
from vish.utils import DEVICE, accuracy, test_dl, to_device, train_dl

# TODO: Overwrite for every train file
DESC = "vit-decomposed-cifar10"
CKPT_DESC = ""

CURR_TIME = datetime.now().strftime(DATE_TIME_FORMAT)

LOG_FILE_NAME = f"logs/EXP-{CURR_TIME}_{DESC}.txt"

if IS_TEST_RUN:
    DESC = f"TEST-{DESC}"
    LOG_FILE_NAME = f"logs/{DESC}.txt"
    EPOCHS = 30

log_file = open(LOG_FILE_NAME, "a")
print("Logging @:", LOG_FILE_NAME)
sys.stdout = log_file

ckpt = None

if LOAD_CKPT:
    DESC = CKPT_DESC
    ckpt = torch.load(f"./checkpoints/{DESC}.pt")
    print("Checkpoint Loaded...")


# Main Evaluation
print("*" * 120)
print("Started at: ", CURR_TIME)

# Model Definition
# Label tree
lt = LabelHierarchyTree("vish/data/cifar10.xml")

# MODEL - CIFAR 10
model = VitClassificationDecomposed(
    img_height=224,
    img_width=224,
    img_in_channels=3,
    patch_dim=16,
    emb_dim=768,
    label_tree=lt,
    max_depth_before_clf=2,
    num_blocks_per_group=2,
)

if LOAD_CKPT:
    model = ckpt["model"]
    print("Model Loaded from checkpoint")
# print(model)

model = to_device(model, DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    momentum=MOMENTUM,
)

if LOAD_CKPT:
    optimizer.load_state_dict(ckpt["opt"][0])
    print("Optimizer loaded from checkpoint")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    verbose=True,
    min_lr=MIN_LR,
    patience=5,
)

# Training
trainer_params = {
    "num_epochs": EPOCHS,
    "train_dl": train_dl,
    "test_dl": test_dl,
    "model": model,
    "optimizer_list": [optimizer],
    "scheduler_list": [scheduler],
    "metrics_list": [("Acc@1", accuracy)],
    "best_score_key": "Acc@1_fine",  # Set it as per need - for dual specify broad or fine
    "model_checkpoint_dir": WEIGHT_FOLDER_PATH,
    "description": DESC,
}


if LOAD_CKPT:
    load_kwargs = {
        "uniq_desc": False,  # Uncomment if loading checkpoint
        "best_train_score": ckpt["best_train_score"],
        "best_test_score": ckpt["best_test_score"],
    }
    trainer_params = {**trainer_params, **load_kwargs}

trainer = VitDecomposedTrainer(**trainer_params)
run_kwargs = {}

trainer.run(**run_kwargs)

print(f"Final Model saved @ {trainer.ckpt_full_path}")
