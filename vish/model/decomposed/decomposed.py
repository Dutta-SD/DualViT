import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vish.model.common.tree import LabelHierarchyTree
from vish.model.common.vit_blocks import PositionalEmbedding1D
from vish.model.decomposed.aggregator import Aggregator
from vish.model.decomposed.entity import TransformerData
from vish.model.decomposed.segregator import Segregator

ROOT_KEY = "class"
DATA_KEY = "pixel_values"
LABEL_KEY = "labels"


def debug_dict(d: dict[str, TransformerData]):
    for key, value in d.items():
        print("\tKey:", key)
        print(
            "\t\tValue -> Data Shape: ",
            value.data.shape,
            "DType:",
            value.data.dtype,
        )
        print(
            "\t\tValue -> Label Shape: ",
            value.labels.shape,
            "DType:",
            value.labels.dtype,
        )


class VitClassificationDecomposed(nn.Module):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        img_in_channels: int = 3,
        patch_dim: int = 16,
        emb_dim: int = 768,
        label_tree: LabelHierarchyTree = None,
        num_blocks_per_group: int = 4,
        max_depth_before_clf: int = 2,  # this is count not index, depth will be [0, max_depth_before_clf)
    ) -> None:
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.patch_dim = patch_dim
        self.emb_dim = emb_dim
        self.max_depth = max_depth_before_clf

        self.num_patch_width = img_width // patch_dim
        self.num_patch_height = img_height // patch_dim

        self.seq_len = self.num_patch_width * self.num_patch_height
        self.label_tree = label_tree
        self.segregator = Segregator(label_tree)
        self.output_leaves = self.label_tree.getall_leaves(ROOT_KEY)
        self.num_leaves = len(self.output_leaves)
        self._pretrained: bool = False

        self.aggregators = nn.ModuleList(
            [
                Aggregator(
                    num_blocks=num_blocks_per_group,
                    labels_list=label_tree.get_elements_at_depth(i),
                )
                for i in range(max_depth_before_clf)
            ]
        )

        self.embedding_layer = nn.Sequential(
            nn.Conv2d(
                img_in_channels,
                emb_dim,
                kernel_size=(patch_dim, patch_dim),
                stride=(patch_dim, patch_dim),
            ),
            Rearrange("b d ph pw -> b (ph pw) d"),
        )

        self.positional_embedding = PositionalEmbedding1D(self.seq_len, self.emb_dim)

        self.classifiers = nn.ModuleDict(
            {
                name: nn.Linear(self.emb_dim, self.num_leaves)
                for name in self.output_leaves
            }
        )

    def from_pretrained(self):
        self._pretrained = True

    def _init_transform(
        self,
        raw: dict[str, torch.FloatTensor],
    ) -> dict[str, TransformerData]:
        """Converts image into required embeddings

        We assume initial data is the form of a dictionary,
        with 3-dimensional image input as raw[`pixel_values`]
        and labels as raw[`labels`]

        We convert it to `TransformerData` class required for the model
        """
        data = raw[DATA_KEY]
        labels = raw[LABEL_KEY].unsqueeze(-1)

        op_data = self.embedding_layer(data)  # B N D
        op_data = self.positional_embedding(op_data)  # Add positional Embedding, B N D

        final_data = rearrange(op_data, "b n d -> 1 (b n) d")
        final_labels = repeat(labels, "b 1 -> 1 (b n)", n=self.seq_len)

        return {ROOT_KEY: TransformerData(data=final_data, labels=final_labels)}

    def forward(self, raw: dict[str, torch.FloatTensor]):
        ip = self._init_transform(raw)

        for curr_depth in range(self.max_depth):
            # print(f"Pre Aggregator @ Depth:  {curr_depth}")
            # self.debug_dict(ip)

            ip = self.aggregators[curr_depth](ip)

            # print(f"Post Aggregator @ Depth:  {curr_depth}")
            # self.debug_dict(ip)

            ip = self.segregator.segregate(ip)

            # print(f"Post Segregator @ Depth:  {curr_depth}")
            # self.debug_dict(ip)

        # Final Segregation to leaves
        ip = self.segregator.segregate(ip, bypass_to_leaf=True)

        # print("Final Segregation to Leaves")
        # self.debug_dict(ip)

        op: dict[str, TransformerData] = {}

        for leaf_key in ip.keys():
            data = ip[leaf_key].data.squeeze(0)  # 1 Z D -> Z D
            labels = ip[leaf_key].labels.T  # Z C

            op[leaf_key] = TransformerData(
                data=self.classifiers[leaf_key](data),  # Z D -> Z C
                labels=labels,  # Z C
            )

        # print("Final Output: ")
        # self.debug_dict(op)

        return op
