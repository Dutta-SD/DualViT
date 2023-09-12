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

    @torch.no_grad()
    def segregate(
        self,
        model_input: dict[str, TransformerData],
        bypass_to_leaf: bool = False,
    ) -> dict[str, TransformerData]:
        """Assume data is (1, Z, D), where Z is arbitrary"""
        seg_output = {}

        for parent_key in model_input.keys():
            tfr_data = model_input[parent_key]
            is_empty = self.is_empty_embeddings(tfr_data)

            if self.label_tree.is_leaf(parent_key) and (
                not bypass_to_leaf and is_empty
            ):
                parent_output = {parent_key: tfr_data}

            elif bypass_to_leaf and is_empty:
                parent_output = self.get_output_empty_bypass(
                    parent_key, tfr_data, bypass=True
                )
            else:
                parent_output = self.get_output_non_leaf(
                    parent_key, tfr_data, bypass=bypass_to_leaf
                )

            seg_output = {**seg_output, **parent_output}

        return seg_output

    def get_output_empty_bypass(self, parent_key: str, tfr_data: TransformerData):
        parent_data = tfr_data.data
        parent_labels = tfr_data.labels

        all_child_keys = self.label_tree.getall_leaves(parent_key, names=True)
        return {
            key: TransformerData(data=parent_data, labels=parent_labels)
            for key in all_child_keys
        }

    def get_output_non_leaf(
        self,
        parent_key: str,
        tfr_data: TransformerData,
        bypass=False,
    ):
        parent_data = tfr_data.data
        parent_labels = tfr_data.labels

        all_child_keys = (
            self.label_tree.get_immediate_children(parent_key, names=True)
            if not bypass
            else self.label_tree.getall_leaves(parent_key, names=True)
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
