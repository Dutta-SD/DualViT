import torch

from vish.model.common.tools import debug_dict
from vish.model.common.tree import LabelHierarchyTree
from vish.model.decomposed import VitClassificationDecomposed

lt = LabelHierarchyTree("vish/data/cifar10.xml")

model = VitClassificationDecomposed(
    img_height=224,
    img_width=224,
    img_in_channels=3,
    label_tree=lt,
    max_depth_before_clf=2,
    num_blocks_per_group=2,
)

ip = {
    "pixel_values": torch.randn(4, 3, 224, 224),
    "labels": torch.randint(0, 10, (4,)),
}


op = model(ip)
debug_dict(op)
