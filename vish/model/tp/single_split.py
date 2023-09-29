import torch
from torch import nn
from transformers import ViTModel, ViTConfig


class SplitHierarchicalTPViT(ViTModel):
    def __init__(self, config: ViTConfig, num_outputs: int = 10):
        super().__init__(config)
        self.n_broad = config.num_hidden_layers // 2
        del self.pooler
        self.classifier = nn.Linear(config.hidden_size, num_outputs, bias=True)

    def forward(self, pixel_values: torch.Tensor):
        ip_fine = ip_broad = self.embeddings(pixel_values)

        b_tensors = []

        for broad_module in self.encoder.layer[: self.n_broad]:
            ip_broad = broad_module(ip_broad)[0]
            b_tensors.append(ip_broad)

        for broad_int_output, fine_module in zip(
            b_tensors, self.encoder.layer[self.n_broad :]
        ):
            ip_fine = fine_module(ip_fine)[0]
            ip_fine = ip_fine * broad_int_output

        broad_embedding = b_tensors[-1][:, :1, :]
        fine_embedding = ip_fine[:, :1, :]

        fine_logits = self.classify(fine_embedding)

        return broad_embedding, fine_embedding, fine_logits

    def classify(self, embeddings):
        return self.classifier(embeddings)


if __name__ == "__main__":
    model = SplitHierarchicalTPViT.from_pretrained("google/vit-base-patch16-224")
    with torch.no_grad():
        rand_tensor = torch.rand(32, 3, 224, 224)
        op = model(rand_tensor)
        for t in op:
            print(t.shape)
