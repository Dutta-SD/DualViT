import torch
from components.utils import *
from components.model.vit import ViTBasicForImageClassification
from constants import *
from datetime import datetime
import sys


MODE = all_modes.FINE_ONLY

EXPERIMENT_UNIQ_ID = datetime.now()
DESC = "hparams-omihub-autoaugment-on-out-proj-included-full-cifar-train"

EXPERIMENT_NAME = f"{EXPERIMENT_UNIQ_ID}-{DESC}"
log_file = open(f"log-vit.txt", "a")
sys.stdout = log_file

# Settings for directing output to a file
print("*" * 100)
print(f"Exp: {EXPERIMENT_NAME}")

model_params = {
    "img_height": 32,
    "img_width": 32,
    "img_in_channels": 3,
    "patch_dim": 2, # reduced patch size
    "emb_dim": 384,
    "num_layers": 7,
    "num_attention_heads": 12,
    "pwff_hidden_dim": 384,
    "num_classification_heads": 1,
    "mlp_outputs_list": (10,),  # Cifar 10 default
    "p_dropout": 0.0,
    "qkv_bias": True,
    "pwff_bias": True,
    "clf_head_bias": True,
    "conv_bias": True,
}


@torch.no_grad()
def evaluate(model: ViTBasicForImageClassification, val_loader, model_mode=MODE):
    model.eval()
    outputs = [model.validation_step(batch, model_mode) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# Experiment 3
model = ViTBasicForImageClassification(**model_params)
print(model)


# Parameters
LOAD_MODEL = False
MODEL_SAVE_NAME = (
    f"{WEIGHT_FOLDER_PATH}/ViT-CIFAR-10-{MODE}-ID-{EXPERIMENT_UNIQ_ID}-{DESC}.pt"
)
MODEL_SAVE_NAME_AFTER_TRAINING = MODEL_SAVE_NAME


if LOAD_MODEL:
    print("loading saved")
    model = torch.load(MODEL_SAVE_NAME)
    print(f"Loaded: {MODEL_SAVE_NAME}")

model = to_device(model, DEVICE)
print(f"Last Saved Model Validation Accuracy: {model.max_accuracy}")

train_dl, test_dl = DeviceDataLoader(train_dl, DEVICE), DeviceDataLoader(
    test_dl, DEVICE
)

evaluate(model, test_dl)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 5, eta_min=1e-5
)


def fit(
    num_epochs,
    model: ViTBasicForImageClassification,
    train_loader,
    val_loader,
    optimizer,
    model_mode,
    model_save_path: str,
):
    """
    epochs - number of epochs
    model - the model in required device
    train_loader - train dataloader
    val_loader - val dataloader
    optimizer - optimizer, with parameters and parameters set
    """
    history = []
    for epoch in range(num_epochs):
        # Training Phase - Fine always last
        model.train()
        # train_losses_broad = []
        train_losses_fine = []

        for batch in train_loader:
            optimizer.zero_grad()
            *_, loss_fine = model.training_step(batch, model_mode)

            # train_losses_broad.append(loss_broad)
            train_losses_fine.append(loss_fine)

            if (
                model_mode == all_modes.FINE_AND_BROAD
                or model_mode == all_modes.FINE_ONLY
            ):
                loss_fine.backward()
            elif model_mode == all_modes.BROAD_ONLY:
                # loss_broad.backward()
                pass

            optimizer.step()
        scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader, model_mode)

        result["train_loss_fine"] = torch.stack(train_losses_fine).mean().item()
        # result["train_loss_broad"] = torch.stack(train_losses_broad).mean().item()
        result["train_loss_broad"] = 0

        model.epoch_end(epoch, result, model_save_path, model_mode)
        history.append(result)
    return history


history = fit(
    EPOCHS,
    model,
    train_dl,
    test_dl,
    optimizer,
    MODE,
    MODEL_SAVE_NAME_AFTER_TRAINING,
)
