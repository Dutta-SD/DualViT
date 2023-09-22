import sys
from datetime import datetime

from vish.model.decomposed.decomposed import VitClassificationDecomposed
from vish.model.tree import LabelHierarchyTree
from vish.trainer.decomposed import VitDecomposedTrainer
from vish.utils import *

# Set this flag to True if you want to just test the thing.
# For running complete experiments, set it to False
TEST = False

DATE_TIME_FORMAT = "%Y_%m_%d_%H_%M_%S"
CURR_TIME = datetime.now().strftime(DATE_TIME_FORMAT)

LOG_FILE_NAME = f"logs/EXP-{CURR_TIME}_{DESC}.txt"


if TEST:
    LOG_FILE_NAME = f"logs/TEST-{DESC}.txt"
    EPOCHS = 20


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
    num_blocks_per_group=3,
)

# model = ckpt["model"]
print(model)

model = to_device(model, DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
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
    "best_score_key": "Acc@1",  # Set it as per need
    "model_checkpoint_dir": WEIGHT_FOLDER_PATH,
    "description": DESC,
    # "uniq_desc": False,  # Uncomment if loading checkpoint
    # "best_train_score": ckpt["best_train_score"],
    # "best_test_score": ckpt["best_test_score"],
}
trainer = VitDecomposedTrainer(**trainer_params)
run_kwargs = {}

trainer.run(**run_kwargs)


print(f"Final Model saved @ {trainer.ckpt_full_path}")
