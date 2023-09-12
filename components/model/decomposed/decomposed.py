from einops import rearrange, repeat
import torch
import torch.nn as nn
from components.model.decomposed.aggregator import Aggregator
from components.model.decomposed.entity import TransformerData
from components.model.decomposed.segregator import Segregator
from einops.layers.torch import Rearrange

from components.model.tree import LabelHierarchyTree
from components.model.vit_blocks import PositionalEmbedding1D

ROOT_KEY = "class"
DATA_KEY = "pixel_values"
LABEL_KEY = "labels"


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

    def _init_transform(
        self,
        raw: dict[str, torch.FloatTensor],
    ) -> dict[str, TransformerData]:
        """Converts image into required embeddings

        We assume initial data is the form of a dictionary,
        with 3 dimensional image input as raw[`pixel_values`]
        and labels as raw[`labels`]

        We convert it to `TransformerData` class required for the model
        """
        data = raw[DATA_KEY]
        labels = raw[LABEL_KEY]
        op_data = self.embedding_layer(data)  # B N D
        op_data = self.positional_embedding(op_data)  # Add positional Embedding, B N D

        final_data = rearrange(op_data, "b n d -> 1 (b n) d")
        final_labels = repeat(labels, "b 1 -> 1 (b n)", n=self.seq_len)

        return {ROOT_KEY: TransformerData(data=final_data, labels=final_labels)}

    def debug_dict(self, d: dict[str, TransformerData]):
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

    def forward(self, raw: dict[str, torch.FloatTensor]):
        ip = self._init_transform(raw)
        for curr_depth in range(self.max_depth):
            print(f"Pre Aggregator @ Depth:  {curr_depth}")
            self.debug_dict(ip)

            ip = self.aggregators[curr_depth](ip)

            print(f"Post Aggregator @ Depth:  {curr_depth}")
            self.debug_dict(ip)

            ip = self.segregator.segregate(ip)

            print(f"Post Segregator @ Depth:  {curr_depth}")
            self.debug_dict(ip)
