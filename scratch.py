# from vish.model.cifar_10_split import TransformerDecomposed
# import torch

# net = TransformerDecomposed()
# batch = torch.rand(32, 197, 768)

# net(batch)


# This is working

# import torch

# t_dict = {i : torch.rand(32, 1) for i in range(10)}

# print(
#     torch.cat(tuple(t_dict.values()), dim=-1).shape
# )

# import torch.nn.functional as F
# import torch

# output = torch.mean(torch.rand(32, 200, 768), dim=-2)

# print(output.shape)

# from vish.model.decomposed.segregator import Segregator
# from vish.model.tree import LabelHierarchyTree
# from vish.model.decomposed.entity import TransformerData
# import torch


# lt = LabelHierarchyTree("vish/data/cifar10.xml")
# sg = Segregator(lt)

# ip = {
#     "dog": TransformerData(
#         data=torch.randn(1, 120, 768), labels=1 * torch.ones(1, 120)
#     ),
#     "cat": TransformerData(
#         data=torch.randn(1, 0, 768), labels=2 * torch.ones(1, 0)
#     ),
#     "herbivore": TransformerData(
#         data=torch.randn(1, 120, 768), labels=3 * torch.ones(1, 120)
#     ),
# }

# op = sg.segregate(ip)

# for key, value in op.items():
#     print("Key: ", key)
#     print("Data shape: ", value.data.shape)
#     print("Label shape: ", value.labels.shape)
#     print("Label Mean:", value.labels.mean())

# for i in x.iter():
#     print(i.tag)

# import torch
# from vish.model.tree import LabelHierarchyTree

# lt = LabelHierarchyTree("vish/data/cifar10.xml")
# # print(lt.get_elements_at_depth(3))

# from vish.model.decomposed.decomposed import VitClassificationDecomposed

# model = VitClassificationDecomposed(
#     img_height=224,
#     img_width=224,
#     img_in_channels=3,
#     patch_dim=16,
#     emb_dim=768,
#     label_tree=lt,
# )
# b = 8
# ip = {
#     "pixel_values": torch.randn(b, 3, 224, 224),
#     "labels": torch.randint(0, 10, size=(b, 1)),
# }

# model(ip)

# from vish.trainer.alternate import BroadFineAlternateModifiedTrainer
# import constants

# for epoch in range(constants.EPOCHS):
#     print(epoch, BroadFineAlternateModifiedTrainer.get_curr_loss_idx(epoch))

# fine_model = VitImageClassificationSingleClassToken.from_pretrained(
#     VIT_PRETRAINED_MODEL_2
# )
# fine_model.pre_forward_adjust(10)  # For cifar 10

# fine_model.requires_grad_(False)


# broad_model = VitImageClassificationSingleClassToken.from_pretrained(
#     VIT_PRETRAINED_MODEL_2
# )
# broad_model.pre_forward_adjust(2)  # For cifar 10 2 classes

# model = VitDualModelBroadFine(fine_model, broad_model)


# print(*[name for name, p in model.named_parameters() if p.requires_grad], sep="\n")
# import torch
# from torch.nn import functional as F
#
# DESC = "vit-b-16-dual-model-pretrained-both_1694963453.pt.pt"
#
# ckpt = torch.load(f"./checkpoints/{DESC}")
# model: VitDualModelBroadFine = ckpt["model"]
# model = to_device(model, DEVICE)
# model.eval()
#
# with torch.no_grad():
#     for batch in train_dl:
#         x, fine_labels, broad_labels = batch
#         emb_fine, output_fine, emb_broad, output_broad = model(x)
#
#         print("Batch Size:", x.shape[0])
#
#         print(
#             "L2 diff between broad embedding and fine embedding:",
#             torch.linalg.norm(emb_fine - emb_broad).item(),
#         )
#         print(
#             "L1 diff between broad embedding and fine embedding:",
#             F.l1_loss(emb_fine, emb_broad).item(),
#         )
#         break

import torch

from vish.model.tp.dual import TPDualVit
from vish.utils import (
    DEVICE,
    accuracy,
    to_device,
    test_dl,
    train_dl,
    get_broad_label_cifar10,
)

# from vish.model.tp.blocks import TPMHA, TPTransformerBlock

# model = TPTransformerBlock()
# print(model)

# hidden_tensor = torch.randn(32, 197, 768)
# repr_tensor = torch.randn(32, 197, 768)

# print("Output Shape: ", model(hidden_tensor, repr_tensor).shape)

CKPTS = [
    {
        "name": "TP-Dual-BNF",
        "ckpt": "./checkpoints/tf-vit-dual-model-p-16-huggingface-pretrained-google-weights-broad-only-BNF-loss_1695638053.pt",
    },
    {
        "name": "TP-Dual-BNFC",
        "ckpt": "./checkpoints/tf-vit-dual-model-p-16-huggingface-pretrained-google-weights-broad-only-BNFCluster-loss_1695667259.pt",
    },
]

def accs(y_true, y_pred):
    n = (y_true == y_pred).sum()
    d = y_true.shape[0]
    return n / d

with torch.no_grad():
    for info in CKPTS:
        name = info["name"]
        path = info["ckpt"]

        model: TPDualVit = torch.load(path)["model"]
        model = to_device(model, DEVICE)
        model.eval()

        acc = []

        for batch in train_dl:
            x, f_l, b_l = batch
            [be, bl], _ = model(x)
            b_op = torch.argmax(model.fine_model.mlp_heads[0](be), dim=1)

            b_op = b_op.clone().detach().cpu()
            b_l = b_l.clone().detach().cpu()

            b_p = b_op.apply_(get_broad_label_cifar10)
            acc.append(accs(b_p, b_l))
            # break

        print(f"Model: {name}, Broad Accuracy(Train) is: {torch.mean(torch.stack(acc))}")

        acc = []

        for batch in test_dl:
            x, f_l, b_l = batch
            [be, bl], _ = model(x)
            b_op = torch.argmax(model.fine_model.mlp_heads[0](be), dim=1)

            b_op = b_op.clone().detach().cpu()
            b_l = b_l.clone().detach().cpu()

            b_p = b_op.apply_(get_broad_label_cifar10)
            acc.append(accs(b_p, b_l))
            # break

        print(f"Model: {name}, Broad Accuracy(Test) is: {torch.mean(torch.stack(acc))}")
