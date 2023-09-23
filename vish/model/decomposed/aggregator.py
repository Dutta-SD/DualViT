from torch import nn as nn

from vish.model.decomposed.entity import TransformerData
from vish.model.common.vit_blocks import TransformerBlockGroup


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
            # Non empty set
            raise KeyError(
                f"Expected Keys: {own_keys} in input, Found Keys: {input_keys}"
            )
