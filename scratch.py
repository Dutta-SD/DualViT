# from components.model.cifar_10_split import TransformerDecomposed
# import torch

# net = TransformerDecomposed()
# batch = torch.rand(32, 197, 768)

# net(batch)

"""
# This is working

import torch

t_dict = {i : torch.rand(32, 1) for i in range(10)}

print(
    torch.cat(tuple(t_dict.values()), dim=-1).shape
)"""

import torch.nn.functional as F
import torch

output = torch.mean(torch.rand(32, 200, 768), dim=-2)

print(output.shape)
