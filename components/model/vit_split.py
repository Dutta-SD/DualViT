from torch import nn
from components.model.vit_blocks import TransformerBlock
from components.model.split import segregate_samples
from components.model.pretrained import PreTrainedWordEmbeddings

LANG_MODEL_NAME = "bert-base-uncased"


class TransformerBlockGroupBase(nn.Module):
    def __init__(self, num_blocks=3) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict(
            {f"transformer_block_{i}": TransformerBlock() for i in range(num_blocks)}
        )

    def forward(self, x):
        for key in self.blocks.keys():
            x = self.blocks[key](x)
        return x


lang_model = PreTrainedWordEmbeddings(LANG_MODEL_NAME)

net_1 = TransformerBlockGroupBase()
net_2 = TransformerBlockGroupBase()
net_3 = TransformerBlockGroupBase()
net_4 = TransformerBlockGroupBase()


class SplitVitModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
