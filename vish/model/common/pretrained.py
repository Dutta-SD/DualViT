from torch import nn as nn
from transformers import ViTForImageClassification, ViTConfig


class VitImageClassificationSingleClassToken(ViTForImageClassification):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.num_outputs = None

    def pre_forward_adjust(self, num_output: int):
        # Changes the classifier
        cfg = self.vit.config
        self.num_outputs = 1
        self.classifier = nn.Linear(cfg.hidden_size, num_output)

    def forward(self, x, *args, **kwargs):
        outputs = self.vit.layernorm(self.vit(x)[0])

        final_embedding = outputs[:, 0]
        final_logits = self.classifier(final_embedding)

        return final_embedding, final_logits
