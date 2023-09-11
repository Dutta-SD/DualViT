from torch import nn as nn
import torch

from components.model.decomposed.config import ViTDecomposedConfig
from components.model.decomposed.entity import TransformerData
from components.model.decomposed.split import segregate_samples_within_batch
from components.model.lang import PreTrainedWordEmbeddings
from components.model.tree import LabelHierarchyTree

_emb_generator = PreTrainedWordEmbeddings(ViTDecomposedConfig.LANG_MODEL_NAME)


class Segregator:
    """
    Segregates input into child embeddings based on pretrained language model embeddings
    """

    def __init__(self, label_tree: LabelHierarchyTree) -> None:
        self._emb_generator = _emb_generator
        self.segregate_samples = segregate_samples_within_batch
        self.label_tree = label_tree

    def get_embeddings(self, words: list[str]) -> torch.FloatTensor:
        embs = [self._emb_generator(word).unsqueeze(0) for word in words]
        return torch.cat(embs, dim=0).unsqueeze(0)

    def is_empty_embeddings(self, tfr_data: TransformerData):
        return tfr_data.data.shape[1] == 0

    def segregate(
        self,
        model_input: dict[str, TransformerData],
    ) -> dict[str, TransformerData]:
        # Assume data is (1, Z, D)
        seg_output = {}
        for parent_key in model_input.keys():
            tfr_data = model_input[parent_key]
            if self.label_tree.is_leaf(parent_key) or self.is_empty_embeddings(
                tfr_data
            ):
                parent_output = {parent_key: tfr_data}
            else:
                parent_output = self.get_output_non_leaf(parent_key, tfr_data)

            seg_output = {**seg_output, **parent_output}

        return seg_output

    def get_output_non_leaf(self, parent_key, tfr_data: TransformerData):
        parent_data = tfr_data.data
        parent_labels = tfr_data.labels

        all_child_keys = self.label_tree.get_immediate_children(
            parent_key, names_only=True
        )

        child_embs = self.get_embeddings(all_child_keys)

        class_div, interim_output = self.segregate_samples(
            child_embs, parent_data, parent_data
        )

        parent_output = {
            txt: TransformerData(
                data=interim_output[idx],
                labels=parent_labels[class_div == idx].unsqueeze(0),
            )
            for txt, idx in zip(all_child_keys, interim_output.keys())
        }

        return parent_output
