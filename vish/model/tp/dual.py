from torch import nn
from transformers.models.vit import ViTForImageClassification


class TPDualVit(nn.Module):
    """
    Dual model using broad fine representations.
    Broad trains on Cross Entropy
    Fine trains on Cross Entropy as well as Representations from broad class
    """

    def __init__(self, fine_model, broad_model):
        super().__init__()
        # TODO: Inherit a separate model that gives intermediate outputs and use it as type hint
        self.fine_model: ViTForImageClassification = fine_model
        self.broad_model: ViTForImageClassification = broad_model

    def forward(self, x):
        pass
