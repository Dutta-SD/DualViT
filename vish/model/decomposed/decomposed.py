import numpy as np
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


class VitClassificationDecomposed(nn.Module):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        label_tree: LabelHierarchyTree = None,
        img_in_channels: int = 3,
        patch_dim: int = 16,
        emb_dim: int = 768,
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
                # +1 for not of that group
                name: nn.Linear(self.emb_dim, 1 + len(label_tree.getall_leaves(name)))
                for name in label_tree.get_elements_at_depth(max_depth_before_clf)
            }
        )

    def from_pretrained(self):
        self._pretrained = True

    @torch.no_grad()
    def get_predicted_outputs(self, outputs: dict[str, TransformerData]):
        # Predict is not like other models
        for key, data in outputs.items():
            label_map = self.make_label_map(key)

            logits = data.data
            labels = data.labels

            converted_labels = self.convert_labels(data, label_map, labels)

            converted_labels

    #         TODO finish

    def convert_labels(self, data, label_map, labels):
        node_labels = torch.tensor(
            np.fromiter(label_map.keys(), dtype=np.int8), dtype=labels.device
        )
        labels_mask = labels.isin(node_labels)
        labels = labels_mask * data.labels
        converted_labels = torch.tensor(
            [label_map[i] for i in labels], device=labels.device
        )
        return converted_labels

    def make_label_map(self, key):
        label_map = {
            int(e[1]["fine_label"]): int(e[1]["group_label"])
            for e in self.label_tree.getall_leaves(key, names=False)
        }
        # For not this group classes
        label_map = {**label_map, 0: 0}
        return label_map

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
        inputs = self._init_transform(raw)

        for curr_depth in range(self.max_depth):
            # print(f"Pre Aggregator @ Depth:  {curr_depth}")
            # self.debug_dict(inputs)

            inputs = self.aggregators[curr_depth](inputs)

            # print(f"Post Aggregator @ Depth:  {curr_depth}")
            # self.debug_dict(inputs)

            inputs = self.segregator.segregate(inputs)

            # print(f"Post Segregator @ Depth:  {curr_depth}")
            # self.debug_dict(inputs)

        # # Final Segregation to leaves -- DO NOT DO
        # inputs = self.segregator.segregate(inputs, bypass_to_leaf=True)

        # print("Final Segregation to Leaves")
        # self.debug_dict(inputs)

        outputs: dict[str, TransformerData] = {}

        for classifier_key in inputs.keys():
            data = inputs[classifier_key].data.squeeze(0)  # 1 Z D -> Z D
            labels = inputs[classifier_key].labels.T  # Z 1

            outputs[classifier_key] = TransformerData(
                data=self.classifiers[classifier_key](data),  # Z D -> Z C
                labels=labels.squeeze(1),  # Z
            )

        # print("Final Output: ")
        # self.debug_dict(outputs)

        return outputs
