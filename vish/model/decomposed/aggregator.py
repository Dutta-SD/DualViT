from torch import nn as nn
from transformers.models.vit import ViTForImageClassification

from vish.model.common.vit_blocks import TransformerBlockGroup
from vish.model.decomposed.entity import TransformerData


class Aggregator(nn.Module):
    """
    Contains a bunch of encoder blocks.
    Takes in multiple inputs and gives out multiple outputs
    Passes inputs via Encoder blocks and aggregates the results
    """

    def __init__(self, num_blocks: int, labels_list: list[str]):
        super().__init__()
        self.n_blocks = num_blocks
        self.n_inputs = len(labels_list)
        self.all_models = nn.ModuleDict(
            {
                labels_list[idx]: TransformerBlockGroup(self.n_blocks)
                for idx in range(self.n_inputs)
            }
        )
        self.curr_level_keys = labels_list

    def _extract_layers(self, pretrained_model: ViTForImageClassification, start):
        return [
            pretrained_model.vit.encoder.layer[idx]
            for idx in range(start, start + self.n_blocks)
        ]

    def from_pretrained(self, pretrained_model, start):
        """
        Extracts blocks from a pretrained model and uses pretrained weights
        Args:
            start: Layer to extract weights from; Weights from start to start + self.num_blocks is used
            pretrained_model: The model to extract weights from

        Returns:
            None
        """
        layers = self._extract_layers(pretrained_model, start)
        # TODO: Are they sharing weights or separate???
        self.all_models = nn.ModuleDict(
            {
                self.labels_list[idx]: TransformerBlockGroup(self.n_blocks).from_pretrained(layers)
                for idx in range(self.n_inputs)
            }
        )

    def forward(self, model_input: dict[str, TransformerData], check: bool = True):
        if check:
            self.alert_for_invalid_keys(model_input)

        output = {}

        for key in self.curr_level_keys:
            ip = model_input[key].data
            labels = model_input[key].labels
            op = self.all_models[key](ip)
            output[key] = TransformerData(data=op, labels=labels)

        return output

    def alert_for_invalid_keys(self, input_dict):
        own_keys = set(self.curr_level_keys)
        input_keys = set(input_dict.keys())

        if own_keys.difference(input_keys):
            # Non-empty set => Spurious Keys Present
            raise KeyError(
                f"Expected Keys: {own_keys} in input, Found Keys: {input_keys}"
            )
