from components.model.cifar_10_split import TransformerDecomposed
import torch

net = TransformerDecomposed()
batch = torch.rand(32, 197, 768)

net(batch)
