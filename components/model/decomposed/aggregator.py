import torch
from components.model.vit_blocks import TransformerBlockGroup


class Aggregator:
    """
    Contains a bunch of encoder blocks.
    Takes in multiple inputs and gives out multiple outputs
    Passes inputs via Encoder blocks and aggregates the results
    """

    def __init__(self, num_blocks: int, labels_list: list[str]):
        super().__init__()
        self.n_inputs = self.n_blocks = num_blocks
        self.all_models = {
            labels_list[idx]: TransformerBlockGroup(self.n_blocks)
            for idx in range(self.n_inputs)
        }
        self.curr_level_keys = labels_list

    def aggregate(self, input_dict: dict[str, torch.FloatTensor], check: bool = True):
        if check:
            self.alert_for_invalid_keys(input_dict)

        output = {}

        for key in self.curr_level_keys:
            ip = input_dict[key]
            op = self.all_models[key](ip)
            output[key] = op

        return output

    def alert_for_invalid_keys(self, input_dict):
        own_keys = set(self.curr_level_keys)
        input_keys = set(input_dict.keys())

        if own_keys.difference(input_keys):
            # Non empty set
            raise KeyError(
                f"Expected Keys: {own_keys} in input, Found Keys: {input_keys}"
            )
