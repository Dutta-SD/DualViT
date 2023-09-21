import time
from torch import nn
import torch
from transformers import AutoModel, AutoTokenizer

from components.utils import to_device, DEVICE


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


if __name__ == "__main__":
    device = torch.device("cpu")
    tik = time.time()
    be_model = PreTrainedWordEmbeddings("distilbert-base-uncased").to(device)
    op = be_model("word")
    print(op.shape)
    tok = time.time()
    print(f"{device} run time = {tok - tik} s")
