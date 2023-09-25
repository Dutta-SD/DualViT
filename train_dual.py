import sys
from datetime import datetime

from torch import nn

from vish.model.dual.dual import VitDualModelBroadFine
from vish.model.common.pretrained import VitImageClassificationSingleClassToken
from vish.trainer.dual import VitDualModelTrainer
from vish.utils import *
from vish.constants import *

# Settings for directing output to a file
str_format = "%Y_%m_%d_%H_%M_%S"
CURR_TIME = datetime.now().strftime(str_format)
LOG_FILE_NAME = f"logs/EXP-{CURR_TIME}_{DESC}.txt"
# LOG_FILE_NAME = f"logs/TEST-{DESC}.txt"
log_file = open(LOG_FILE_NAME, "a")
sys.stdout = log_file

# DESC = "vit-b-16-dual-model-pretrained-both_1694963453.pt"

LOAD = False

if LOAD:
    ckpt = torch.load(f"./checkpoints/{DESC}")
    print("Checkpoint Loaded...")


# Main Evaluation
print("*" * 120)
print("Started at: ", CURR_TIME)


fine_model = VitImageClassificationSingleClassToken.from_pretrained(
    VIT_PRETRAINED_MODEL_2
)
fine_model.pre_forward_adjust(10)  # For cifar 10

broad_model = VitImageClassificationSingleClassToken.from_pretrained(
    VIT_PRETRAINED_MODEL_2
)
broad_model.pre_forward_adjust(2)  # For cifar 10 2 classes

model = VitDualModelBroadFine(fine_model, broad_model)

# model: VitDualModelBroadFine = ckpt["model"]
# model.model_fine.requires_grad_(False)

# print(model)
model = to_device(model, DEVICE)

optimizer = torch.optim.SGD(
    [
        {"params": model.model_fine.vit.parameters(), "lr": 1e-5},
        {"params": model.model_broad.vit.parameters(), "lr": 1e-5},
        {"params": model.model_fine.classifier.parameters(), "lr": 1e-3},
        {"params": model.model_broad.classifier.parameters(), "lr": 1e-3},
    ],
    weight_decay=WEIGHT_DECAY,
    momentum=0.9,
)
# optimizer.load_state_dict(ckpt["opt"][0])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    verbose=True,
    min_lr=MIN_LR,
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
    "best_score_key": "Acc@1_fine",  # Set it as per need
    "model_checkpoint_dir": WEIGHT_FOLDER_PATH,
    "description": DESC,
    # "uniq_desc": False,  # Uncomment if loading checkpoint
    # "best_train_score": ckpt["best_train_score"],
    # "best_test_score": ckpt["best_test_score"],
}
trainer = VitDualModelTrainer(**trainer_params)
run_kwargs = {
    "fine_class_CE": nn.CrossEntropyLoss(),
    # "broad_class_CE": nn.CrossEntropyLoss(),
}

trainer.run(**run_kwargs)


print(f"Final Model saved @ {trainer.ckpt_full_path}")