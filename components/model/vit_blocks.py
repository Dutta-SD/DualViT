import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np


class PositionalEmbedding1D(nn.Module):
    """
    Adds (optionally learned) positional embeddings to the inputs
    When using additional classification token, seq_len will be sequence length + 1

    The forward method expects the input to have the shape of (batch_dim, seq_len, emb_dim)

    seq_len: The number of tokens in the input sequence
    d_model: The embedding dimension of the model
    """

    def __init__(self, seq_len, emb_dim: int = 768):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, emb_dim))

    def forward(self, x, fixed=False):
        positional_embedding = (
            self.pos_embedding if not fixed else self._get_fixed_embedding(x)
        )
        return x + positional_embedding

    def _get_fixed_embedding(self, x):
        # for fixed positional embedding
        raise NotImplemented("Fixed positional embedding not implemented")


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention Block.
    Multi Head, so splits along last dimension(Embedding Dimension) and rejoins
    Takes in a tensor of shape (batch_dim, seq_len, emb_dim) and computes query, key, values
    """

    def __init__(
        self,
        emb_dim: int = 768,
        num_heads: int = 12,
        p_dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.query_layer = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.key_layer = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.value_layer = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.dropout = nn.Dropout(p_dropout)

        # Validations
        self._validate_head_and_emb_dim()

        # Additional Values
        self.norm_factor = np.sqrt(self.emb_dim // self.num_heads)

    def _validate_head_and_emb_dim(self):
        # We need to ensure that n_heads divides emb_dim otherwise split is awkward
        if self.emb_dim % self.num_heads != 0:
            raise ArithmeticError(f"{self.num_heads} is not a factor of {self.emb_dim}")

    def to_qkv(self, x):
        return self.query_layer(x), self.key_layer(x), self.value_layer(x)

    def _split_for_multi_head(self, queries, keys, values):
        # Split the last dimension to d_h such that d_h * n_heads = emb_dim
        return [
            einops.rearrange(mat, "b s (nh dh) -> b nh s dh", nh=self.num_heads)
            for mat in [queries, keys, values]
        ]

    def _add_mask(self, scores, mask):
        mask = mask[:, None, None, :].float()
        scores -= 10000.0 * (1.0 - mask)
        return scores

    def forward(self, x, mask=None):
        """
        x, query, key, value: All have shape (batch_size, seq_len, emb_dim)
        mask(optional): for masked image modelling (batch_size, seq_len)

        Multi Head Attention: Splits emb_dim into d_h dimensions such that
        d_h * n_heads = emb_dim holds

        query, key, value can be computed from here or externally injected.
        """
        query_multi_head, key_multi_head, value_multihead = self._split_for_multi_head(
            *self.to_qkv(x)
        )

        attn_mat_mh = (
            query_multi_head @ key_multi_head.transpose(-2, -1)
        ) / self.norm_factor

        if mask is not None:
            attn_mat_mh = self._add_mask(attn_mat_mh, mask)

        attn_mat_mh = self.dropout(F.softmax(attn_mat_mh, dim=-1))

        final_values = einops.rearrange(
            attn_mat_mh @ value_multihead, "b nh s dh -> b s nh dh"
        )
        # Concat
        final_values = einops.rearrange(final_values, "b s nh dh -> b s (nh dh)")
        return self.out_proj(final_values)


class PositionWiseFeedForward(nn.Module):
    """
    FeedForward Neural Networks for each
    element of the seuqence
    """

    def __init__(
        self,
        emb_dim: int = 768,
        feed_fwd_dim: int = 3072,
        p_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.pos_wise_feed_forward = nn.Sequential(
            nn.Linear(emb_dim, feed_fwd_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(feed_fwd_dim, emb_dim, bias=bias),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        return self.pos_wise_feed_forward(x)


class TransformerBlock(nn.Module):
    """
    Single Block of Transformer with Residual Connection

    """

    def __init__(
        self,
        emb_dim: int = 768,
        num_heads: int = 12,
        pos_wise_ff_dim: int = 3072,
        p_dropout: float = 0.0,
        qkv_bias: bool = True,
        pwff_bias: bool = True,
        eps: float = 1e-06,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(emb_dim, num_heads, p_dropout, qkv_bias)
        self.layer_norm_1 = nn.LayerNorm(emb_dim, eps)
        self.pos_wise_ff_layer = PositionWiseFeedForward(
            emb_dim, pos_wise_ff_dim, p_dropout, pwff_bias
        )
        self.layer_norm_2 = nn.LayerNorm(emb_dim, eps)
        self.dropout = nn.Dropout(p_dropout)

    def _get_mha_residue(self, ip, mask=None):
        res = self.layer_norm_1(ip)
        res = self.mha(res, mask)
        res = self.dropout(res)
        return res

    def _get_pwff_residue(self, ip):
        residue = self.layer_norm_2(ip)
        residue = self.pos_wise_ff_layer(residue)
        residue = self.dropout(residue)
        return residue

    def forward(self, x, mask=None):
        output = x
        output = output + self._get_mha_residue(output, mask)
        output = output + self._get_pwff_residue(output)
        return output


class TransformerEncoder(nn.Module):
    """
    Transformer with Self-Attention Blocks
    """

    def __init__(
        self,
        num_layers: int = 12,
        emb_dim: int = 768,
        num_heads: int = 12,
        pwff_dim: int = 3072,
        p_dropout: float = 0.0,
        qkv_bias: bool = True,
        pwff_bias: bool = True,
    ):
        super().__init__()
        self.tf_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim, num_heads, pwff_dim, p_dropout, qkv_bias, pwff_bias
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        for block in self.tf_blocks:
            x = block(x, mask)
        return x
