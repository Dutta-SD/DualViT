import sys

import torch
from torch import nn

from components.model.pretrained import VitImageClassificationBroadFine
from components.trainer.alternate import BroadFineAlternateModifiedTrainer
from components.utils import *
from constants import *
from datetime import datetime

# Settings for directing output to a file
str_format = "%Y_%m_%d_%H_%M_%S"
CURR_TIME = datetime.now().strftime(str_format)
LOG_FILE_NAME = f"logs/EXP-{CURR_TIME}_{DESC}.txt"
# LOG_FILE_NAME = f"logs/TEST-{DESC}.txt"
log_file = open(LOG_FILE_NAME, "a")
sys.stdout = log_file

# DESC="vit-b-16-modified-loss-updated-google-weights_1694713704"

# Main Evaluation
print("*" * 120)
print("Started at: ", CURR_TIME)

# ckpt = torch.load("./checkpoints/vit-b-16-modified-loss-updated-google-weights_1694713704.pt")
# print("Checkpoint Loaded...")

model = VitImageClassificationBroadFine.from_pretrained(VIT_PRETRAINED_MODEL_2)
# model = ckpt["model"]
# print(model)
model.pre_forward_adjust((2, 10))
model = to_device(model, DEVICE)

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=LEARNING_RATE,
#     weight_decay=WEIGHT_DECAY,
# )
optimizer = torch.optim.SGD(
    [
        {"params": model.vit.embeddings.parameters(), "lr": 1e-3},
        {"params": model.vit.encoder.parameters(), "lr": 1e-5},
        {"params": model.vit.layernorm.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ],
    weight_decay=WEIGHT_DECAY,
    momentum=0.9,
)
# optimizer.load_state_dict(ckpt["opt"][0])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    verbose=True,
    min_lr=MIN_LR
)

### Training
trainer_params = {
    "num_epochs": EPOCHS,
    "train_dl": train_dl,
    "test_dl": test_dl,
    "model": model,
    "optimizer_list": [optimizer],
    "scheduler_list": [scheduler],
    "metrics_list": [("Acc@1", accuracy)],
    "best_score_key": "Acc@1_fine",  # Set it as per need
    "model_checkpoint_dir": WEIGHT_FOLDER_PATH,
    "description": DESC,
    # "uniq_desc": False,
    # "best_train_score": ckpt["best_train_score"],
    # "best_test_score": ckpt["best_test_score"],
}
trainer = BroadFineAlternateModifiedTrainer(**trainer_params)
run_kawrgs = {"fine_class_CE": nn.CrossEntropyLoss()}

trainer.run(**run_kawrgs)


print(f"Final Model saved @ {trainer.ckpt_full_path}")
