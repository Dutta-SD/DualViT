# Decomposed model via Cifar 10
# TODO: This is hardcoded. Parse hierarchichal tree and create blocks regarding the same


from torch import nn
from components.model.vit_blocks import TransformerBlockGroupBase
from components.model.split import segregate_samples
from components.model.pretrained import PreTrainedWordEmbeddings

LANG_MODEL_NAME = "bert-base-uncased"


lang_model = PreTrainedWordEmbeddings(LANG_MODEL_NAME)

net1 = TransformerBlockGroupBase(num_blocks=5)
net2 = TransformerBlockGroupBase(num_blocks=5)


