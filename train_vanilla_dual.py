import sys
from datetime import datetime

from torch import nn
from transformers.models.vit import ViTForImageClassification

from vish.model.tp.dual import TPDualVit
from vish.model.tp.tp_vit import TPVitImageClassification
from vish.trainer.dual import TPDualTrainer
from vish.utils import *

# NOTE: Overwrite for every train file
DESC = "tf-vit-dual-model-p-16-huggingface-pretrained-google-weights-broad-only"

# Set this flag to True if you want to just test the thing.
# For running complete experiments, set it to False
TEST = False

DATE_TIME_FORMAT = "%Y_%m_%d_%H_%M_%S"
CURR_TIME = datetime.now().strftime(DATE_TIME_FORMAT)

LOG_FILE_NAME = f"logs/EXP-{CURR_TIME}_{DESC}.txt"


if TEST:
    LOG_FILE_NAME = f"logs/TEST-{DESC}.txt"
    EPOCHS = 30


log_file = open(LOG_FILE_NAME, "a")
sys.stdout = log_file


LOAD = False

if LOAD:
    # DESC = "vit-b-16-dual-model-pretrained-both_1694963453.pt"
    ckpt = torch.load(f"./checkpoints/{DESC}")
    print("Checkpoint Loaded...")


# Main Evaluation
print("*" * 120)
print("Started at: ", CURR_TIME)

# Model Definition
model_params = {
    "img_height": 224,
    "img_width": 224,
    "img_in_channels": 3,
    "patch_dim": 16,  # reduced patch size
    "emb_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 3072,
    "num_classification_heads": 1,
    "mlp_outputs_list": (10,),  # Cifar 10 default
    "p_dropout": 0.1,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}

# MODEL CONFIGURATION
fine_model = TPVitImageClassification(**model_params)
broad_model = ViTForImageClassification.from_pretrained(VIT_PRETRAINED_MODEL_2)
broad_model.classifier = nn.Linear(broad_model.config.hidden_size, 2)

model = TPDualVit(fine_model, broad_model)

# model = ckpt["model"]
print(model)

model = to_device(model, DEVICE)

optimizer = torch.optim.SGD(
    [
        {"params": model.fine_model.parameters(), "lr": 1e-3},
        {"params": model.broad_model.vit.parameters(), "lr": 5e-6},
        {"params": model.broad_model.classifier.parameters(), "lr": 1e-3},
    ],
    lr=LEARNING_RATE,
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
    "best_score_key": "Acc@1_fine",  # Set it as per need - for dual specify broad or fine
    "model_checkpoint_dir": WEIGHT_FOLDER_PATH,
    "description": DESC,
    # "uniq_desc": False,  # Uncomment if loading checkpoint
    # "best_train_score": ckpt["best_train_score"],
    # "best_test_score": ckpt["best_test_score"],
}
trainer = TPDualTrainer(**trainer_params)
run_kwargs = {"fine_class_CE": nn.CrossEntropyLoss(), "broad_class_CE": nn.CrossEntropyLoss()}

trainer.run(**run_kwargs)

print(f"Final Model saved @ {trainer.ckpt_full_path}")
