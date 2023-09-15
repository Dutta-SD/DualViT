# from components.model.cifar_10_split import TransformerDecomposed
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

# from components.model.decomposed.segregator import Segregator
# from components.model.tree import LabelHierarchyTree
# from components.model.decomposed.entity import TransformerData
# import torch


# lt = LabelHierarchyTree("components/data/cifar10.xml")
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
# from components.model.tree import LabelHierarchyTree

# lt = LabelHierarchyTree("components/data/cifar10.xml")
# # print(lt.get_elements_at_depth(3))

# from components.model.decomposed.decomposed import VitClassificationDecomposed

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

from components.trainer.alternate import BroadFineAlternateModifiedTrainer
import constants

for epoch in range(constants.EPOCHS):
    print(epoch, BroadFineAlternateModifiedTrainer.get_curr_loss_idx(epoch))
