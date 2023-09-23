from torch import nn as nn

from vish.model.common.pretrained import VitImageClassificationSingleClassToken


class VitDualModelBroadFine(nn.Module):
    def __init__(
        self,
        model_fine: VitImageClassificationSingleClassToken,
        model_broad: VitImageClassificationSingleClassToken,
    ):
        super().__init__()
        self.model_fine = model_fine
        self.model_broad = model_broad

    def forward(self, x):
        emb_fine, output_fine = self.model_fine(x)
        emb_broad, output_broad = self.model_broad(x)

        return emb_fine, output_fine, emb_broad, output_broad
