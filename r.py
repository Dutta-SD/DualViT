import torch
from tp_model import TP_MODEL_MODIFIED_CIFAR10
from vish.lightning.modulev2 import BroadFineModelLM
from vish.utils import test_dl


if __name__ == "__main__":
    CKPT_PATH = "logs/cifar100/modified_dual_tpvit_fulldataset/lightning_logs/version_1/checkpoints/tpdualvitcifar100-epoch=71-val_acc_fine=0.864.ckpt"
    ck = torch.load(CKPT_PATH)['state_dict']
    print(*ck.keys(), sep = "\n")
