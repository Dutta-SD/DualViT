from torch import nn
from transformers.models.vit import ViTForImageClassification

from vish.model.tp.tp_vit import TPVitImageClassification


class TPDualVit(nn.Module):
    """
    Dual model using broad fine representations.
    Broad trains on Cross Entropy,
    Fine trains on Cross Entropy as well as Representations from broad class
    """

    def __init__(
        self,
        fine_model: TPVitImageClassification,
        broad_model: ViTForImageClassification,
    ):
        super().__init__()
        self.fine_model = fine_model
        self.broad_model = broad_model
        self.num_layers = fine_model.transformer_encoder.num_layers

    def get_broad_outputs(self, x):
        broad_outputs = self.broad_model(x, output_hidden_states=True)
        # From HuggingFace documentation
        return broad_outputs["hidden_states"], broad_outputs["logits"]

    def forward(self, x, start=None, stride=None):
        x_ext_list, broad_logits = self.get_broad_outputs(x)
        broad_embedding = x_ext_list[-1][:, 0, :]
        x_ext_list = self.filter_external_inputs(start, stride, x_ext_list)

        fine_embedding, fine_logits = self.fine_model(x, x_ext_list)
        return [broad_embedding, broad_logits], [fine_embedding, fine_logits]

    def filter_external_inputs(self, start, stride, x_ext_list):
        # If both None, use all from pretrained model
        if (start is not None) and (stride is not None):
            valid_indexes = list(range(start, self.num_layers, stride))
            x_ext_list = [
                x_ext_list[idx] if idx in valid_indexes else None
                for idx in range(len(x_ext_list))
            ]
        return x_ext_list


class TPDualVitAlternate(nn.Module):
    """
    TP based dual model with alternate training
    """
    def __init__(
        self,
        fine_model: TPVitImageClassification,
        broad_model: ViTForImageClassification,
    ):
        super().__init__()
        self.fine_model = fine_model
        self.broad_model = broad_model
        self.num_layers = fine_model.transformer_encoder.num_layers