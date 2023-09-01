import sys

import torch
from torch import nn

from components.model.vit import ViTBasicForImageClassification
from components.trainer.custom import BroadClassTrainer, FineClassTrainer
from components.utils import *
from constants import *

# Settings for directing output to a file
log_file = open(f"log-vit.txt", "a")
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

### Broad Training
broad_trainer_params = {
    "num_epochs": BROAD_EPOCHS,
    "train_dl": train_dl,
    "test_dl": test_dl,
    "model": model,
    "optimizer_list": [optimizer],
    "scheduler_list": [scheduler],
    "metrics_list": [("Acc@1", accuracy)],
    "best_score_key": "Acc@1",
    "model_checkpoint_dir": WEIGHT_FOLDER_PATH,
    "description": DESC,
}
broad_trainer = BroadClassTrainer(**broad_trainer_params)
broad_run_kwargs = {"broad_class_CE": nn.CrossEntropyLoss()}

broad_trainer.run(**broad_run_kwargs)

### Fine Training
MODEL_SAVE_NAME = broad_trainer.ckpt_full_path

print(f"loading saved model @ {MODEL_SAVE_NAME}")
ckpt = torch.load(MODEL_SAVE_NAME)
model: ViTBasicForImageClassification = ckpt["model"]

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

optimizer.load_state_dict(ckpt["opt"][0])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    # min_lr=MIN_LR,
)


fine_train_params = {
    "num_epochs": FINE_EPOCHS,
    "train_dl": train_dl,
    "test_dl": test_dl,
    "model": model,
    "optimizer_list": [optimizer],
    "scheduler_list": [scheduler],
    "metrics_list": [("Acc@1", accuracy)],
    "best_score_key": "Acc@1",
    "model_checkpoint_dir": WEIGHT_FOLDER_PATH,
    "description": broad_trainer.exp_name,  # Same name
    "uniq_desc": False,
}
fine_trainer = FineClassTrainer(**fine_train_params)

fine_run_kwargs = {"fine_class_CE": nn.CrossEntropyLoss()}
fine_trainer.run(**fine_run_kwargs)

print(f"Final Model saved @ {fine_trainer.ckpt_full_path}")
