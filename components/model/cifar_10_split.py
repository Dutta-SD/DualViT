# Decomposed model via Cifar 10
# TODO: This is hardcoded. Parse hierarchichal tree and create blocks regarding the same


from torch import nn
from components.model.vit_blocks import TransformerBlockGroup
from components.model.split import segregate_samples
from components.model.pretrained import PreTrainedWordEmbeddings
from components.model.tree import LabelHierarchyTree

LANG_MODEL_NAME = "bert-base-uncased"


lang_model = PreTrainedWordEmbeddings(LANG_MODEL_NAME)
label_tree = LabelHierarchyTree("components/data/cifar10.xml")

net_l1 = TransformerBlockGroup(num_blocks=4)
net_l2 = {
    "animal": TransformerBlockGroup(num_blocks=4),
    "vehicle": TransformerBlockGroup(num_blocks=4),
}
net_l3 = {
    "heavy": TransformerBlockGroup(num_blocks=4),
    "light": TransformerBlockGroup(num_blocks=4),
    "domestic": TransformerBlockGroup(num_blocks=4),
    "herbivore": TransformerBlockGroup(num_blocks=4),
    "other_animal": TransformerBlockGroup(num_blocks=4),
}


class TransformerDecomposed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net_l1 = net_l1
        self.net_l2 = net_l2
        self.net_l3 = net_l3

    def forward(self, x, super_label_info):
        # 0 is the root for super_label_info
        outputs_l1 = self.net_l1(x)
        super_class_label_1 = super_label_info[1]
        
