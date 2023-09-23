from torch import nn
from transformers.models.vit import ViTForImageClassification

from vish.model.tp.tp_vit import TPVitImageClassification


class TPDualVit(nn.Module):
    """
    Dual model using broad fine representations.
    Broad trains on Cross Entropy
    Fine trains on Cross Entropy as well as Representations from broad class
    """

    def __init__(self, fine_model: TPVitImageClassification, broad_model: ViTForImageClassification):
        super().__init__()
        # TODO: To modify broad gradients or not to modify broad gradients from fine?
        self.fine_model = fine_model
        self.broad_model = broad_model
        self.num_layers = fine_model.transformer_encoder.num_layers

    def get_broad_outputs(self, x):
        ops = self.broad_model(x, output_hidden_states=True)
        # From HuggingFace documentation
        return ops["hidden_states"], ops["logits"]

    def forward(self, x, start=None, stride=None):
        x_ext_list, broad_outputs = self.get_broad_outputs(x)
        if (start is not None) and (stride is not None):
            valid_indexes = list(range(start, self.num_layers, stride))
            x_ext_list = [
                x_ext_list[idx] if idx in valid_indexes else None
                for idx in range(len(x_ext_list))
            ]

        # Ensure Broad output and fine output are of appropriate dimensions
        # Not handled here
        fine_outputs = self.fine_model(x, x_ext_list)
        return broad_outputs, fine_outputs
