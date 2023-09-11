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

from components.model.decomposed.segregator import Segregator
from components.model.tree import LabelHierarchyTree
from components.model.decomposed.entity import TransformerData
import torch


lt = LabelHierarchyTree("components/data/cifar10.xml")
sg = Segregator(lt)

ip = {
    "dog": TransformerData(
        data=torch.randn(1, 120, 768), labels=1 * torch.ones(1, 120)
    ),
    "cat": TransformerData(
        data=torch.randn(1, 0, 768), labels=2 * torch.ones(1, 0)
    ),
    "herbivore": TransformerData(
        data=torch.randn(1, 120, 768), labels=3 * torch.ones(1, 120)
    ),
}

op = sg.segregate(ip)

for key, value in op.items():
    print("Key: ", key)
    print("Data shape: ", value.data.shape)
    print("Label shape: ", value.labels.shape)
    print("Label Mean:", value.labels.mean())
