import torch
from torch import nn as nn
from torch.nn import functional as F
from components.model.lang import PreTrainedWordEmbeddings
from components.model.decomposed.config import ViTDecomposedConfig
from components.model.decomposed.split import segregate_samples_within_batch

_emb_generator = PreTrainedWordEmbeddings(ViTDecomposedConfig.LANG_MODEL_NAME)


class Segregator:
    """
    Segregates input into child embeddings based on pretrained language model embeddings
    """

    def __init__(
        self,
        input_keys: list[str],
    ) -> None:
        self._emb_generator = _emb_generator
        self.input_keys = input_keys
        self.segregate_samples = segregate_samples_within_batch
