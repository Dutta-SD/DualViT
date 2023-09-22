import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from vish.utils import DEVICE


class PreTrainedWordEmbeddings(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, word: str):
        encoded_text = self.tokenizer(word, return_tensors="pt").to(DEVICE)
        output = self.model(**encoded_text)
        return output["last_hidden_state"][:, 0].squeeze(0)
