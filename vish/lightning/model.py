from copy import deepcopy
import torch
from torch import nn
from transformers import ViTModel, ViTConfig
from itertools import chain


class SplitHierarchicalTPViT(ViTModel):
    def __init__(
        self,
        config: ViTConfig,
        num_broad_outputs: int = 2,
        num_fine_outputs: int = 10,
    ):
        super().__init__(config)
        # Make sure number of hidden layers is even number
        self.n_broad = config.num_hidden_layers // 2
        del self.pooler
        self.classifier_fine = nn.Linear(
            config.hidden_size, num_fine_outputs, bias=True
        )
        self.classifier_broad = nn.Linear(
            config.hidden_size, num_broad_outputs, bias=True
        )

    def forward(self, pixel_values: torch.Tensor):
        ip_fine = ip_broad = self.embeddings(pixel_values)

        # Last ones are Layers, not list of layers
        broad_modules, last_broad_module = (
            self.encoder.layer[: self.n_broad - 1],
            self.encoder.layer[self.n_broad - 1],
        )
        fine_modules, last_fine_module = (
            self.encoder.layer[self.n_broad : -1],
            self.encoder.layer[-1],
        )

        for broad_module, fine_module in zip(broad_modules, fine_modules):
            # self.n_broad - 1 and last(-1) not used in TP
            output_broad = broad_module(ip_broad)[0]
            output_fine = fine_module(ip_fine)[0]
            ip_fine = output_fine * output_broad
            ip_broad = output_broad

        # output from last encoder layer without TP
        ip_broad = last_broad_module(ip_broad)[0]
        ip_fine = last_fine_module(ip_fine)[0]

        broad_embedding = ip_broad[:, :1, :]
        fine_embedding = ip_fine[:, :1, :]

        fine_logits = self.classify_fine(fine_embedding.squeeze(1))
        broad_logits = self.classify_broad(broad_embedding.squeeze(1))

        # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
        return broad_embedding, fine_embedding, broad_logits, fine_logits

    def classify_fine(self, embeddings):
        return self.classifier_fine(embeddings)

    def classify_broad(self, embeddings):
        return self.classifier_broad(embeddings)


class SplitViTHierarchicalTPVitHalfPretrained(SplitHierarchicalTPViT):
    def __init__(
        self,
        config: ViTConfig,
        num_broad_outputs: int = 2,
        num_fine_outputs: int = 10,
    ):
        super().__init__(config, num_broad_outputs, num_fine_outputs)
        del self.classifier_broad

        self.classifier_fine = None
        self.fine_encoder = None
        self.broad_encoder = None

    def get_large_lr_params(self, lr):
        return [
            {"params": m.parameters(), "lr": lr}
            for m in [
                self.get_broad_layers(),
                self.broad_embeddings,
                self.classifier_fine,
            ]
        ]

    def get_small_lr_parameters(self, lr):
        return [
            {"params": m.parameters(), "lr": lr}
            for m in [self.get_fine_layers(), self.fine_embeddings]
        ]

    @staticmethod
    @torch.no_grad()
    def random_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.fill_(0.01)

    def init_broad_fine(self, num_fine_outputs):
        self.classifier_fine = nn.Linear(
            self.config.hidden_size, num_fine_outputs, True
        )
        self.fine_encoder = self.encoder.layer
        self.broad_encoder = deepcopy(self.encoder.layer)
        self.fine_embeddings = self.embeddings
        self.broad_embeddings = deepcopy(self.embeddings)

        self.broad_embeddings = self.broad_embeddings.apply(self.random_init)
        self.broad_encoder = self.broad_encoder.apply(self.random_init)

    def get_broad_layers(self):
        return self.broad_encoder

    def get_fine_layers(self):
        return self.fine_encoder

    def forward(
        self,
        pixel_values: torch.Tensor,
        clf_broad: bool = True,
        clf_fine: bool = True,
    ):
        """
        Forward or __call__ method
        Args:
            pixel_values: input image tensor
            clf_broad: To classify or not to classify broad
            clf_fine: To classify or not to classify fine

        Returns:

        """
        broad_embedding, fine_embedding = self.get_embeddings(pixel_values)

        fine_logits, broad_logits = None, None

        if clf_fine:
            fine_logits = self.classify_fine(fine_embedding.squeeze(1))

        if clf_broad:
            broad_logits = self.classify_broad(broad_embedding.squeeze(1))

        # [B, 1, D], [B, 1, D], [B, C_broad], [B, C_fine]
        return broad_embedding, fine_embedding, broad_logits, fine_logits

    def get_embeddings(self, pixel_values):
        ip_fine = self.fine_embeddings(pixel_values)
        ip_broad = self.broad_embeddings(pixel_values)

        # Last ones are Layers, not list of layers
        *broad_modules, last_broad_module = self.get_broad_layers()
        *fine_modules, last_fine_module = self.get_fine_layers()

        for broad_module, fine_module in zip(broad_modules, fine_modules):
            # self.n_broad - 1 and last(-1) not used in TP
            output_broad = broad_module(ip_broad)[0]
            output_fine = fine_module(ip_fine)[0]
            # Fine to broad Hadamard so that fine BP twice
            ip_broad = output_fine * output_broad
            ip_fine = output_fine

        # output from last encoder layer without TP
        ip_broad = last_broad_module(ip_broad)[0]
        ip_fine = last_fine_module(ip_fine)[0]

        broad_embedding = ip_broad[:, :1, :]
        fine_embedding = ip_fine[:, :1, :]
        return broad_embedding, fine_embedding
    

# class SplitHVit
