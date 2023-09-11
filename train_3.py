import sys

import torch
from torch import nn

from components.model.vit import ViTBasicForImageClassification
from components.trainer.custom import BroadFineAlternateModifiedTrainer
from components.utils import *
from constants import *
from datetime import datetime

# Settings for directing output to a file
str_format = "%Y_%m_%d_%H_%M_%S"
log_file = open(f"logs/{datetime.now().strftime(str_format)}_{DESC}.txt", "a")
sys.stdout = log_file

# Main Evaluation
print("*" * 120)

train_dl, test_dl = DeviceDataLoader(train_dl, DEVICE), DeviceDataLoader(
    test_dl, DEVICE
)

model_params = {
    "img_height": 224,
    "img_width": 224,
    "img_in_channels": 3,
    "patch_dim": 16,  # reduced patch size
    "emb_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 3072,
    "num_classification_heads": 2,
    "mlp_outputs_list": (2, 10),  # Cifar 10 default
    "p_dropout": 0.0,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}

model = ViTBasicForImageClassification(**model_params)
model = to_device(model, DEVICE)
print(model)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    # min_lr=MIN_LR
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
}
trainer = BroadFineAlternateModifiedTrainer(**trainer_params)
run_kawrgs = {
    "broad_class_CE": nn.CrossEntropyLoss(),
    "fine_class_CE": nn.CrossEntropyLoss(),
}

trainer.run(**run_kawrgs)


print(f"Final Model saved @ {trainer.ckpt_full_path}")
