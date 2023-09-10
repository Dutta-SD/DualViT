import torch
from typing import Dict
from einops import repeat, rearrange


def segregate_samples_within_batch(
    seg_queries: torch.FloatTensor,
    keys: torch.FloatTensor,
    values: torch.FloatTensor,
) -> Dict[int, torch.FloatTensor]:
    """
    Segregates Values based on given Query Vectors

    Args:
        values: (torch.FloatTensor) Samples of shape [batch, seq, emb_dim]

        keys: (torch.FloatTensor) Samples of shape [batch, seq, emb_dim]

        seg_queries: (torch.FloatTensor) Queries of shape [batch, num_classes, emb_dim]

    Return:
        Dict[int, torch.FloatTensor] The segregated values based on keys
    """
    num_classes = seg_queries.shape[1]

    dot_product = seg_queries @ keys.transpose(-1, -2)
    class_division = torch.argmax(dot_product, dim=1)

    return class_division, {
        class_idx: values[class_division == class_idx].unsqueeze(0)
        for class_idx in range(num_classes)
    }


def segregate_samples_within_sample(
    seg_queries: torch.FloatTensor,
    keys: torch.FloatTensor,
    values: torch.FloatTensor,
) -> Dict[int, torch.FloatTensor]:
    batch_dim, num_classes, emb_dim = seg_queries.shape

    dot_product = seg_queries @ keys.transpose(-1, -2)
    print(dot_product.shape)

    class_division = torch.argmax(dot_product, dim=1)
    print("Class division shape is: ", class_division.shape)

    return class_division, {
        class_idx: torch.masked_select(values, dot_product)
        for class_idx in range(num_classes)
    }


if __name__ == "__main__":
    b = 32
    c = 3
    d = 768
    n = 197

    keys = torch.rand(b, n, d)
    values = torch.rand(b, n, d)
    # [B, C, D]
    seg_queries = repeat(torch.rand(c, d), "c d -> b c d", b=b)
    class_divisions, separated_values = segregate_samples_within_batch(
        seg_queries, keys, values
    )

    print("Class Div: ", class_divisions.shape)
    for idx in separated_values.keys():
        print("Shape for key", idx, separated_values[idx].shape)

    seg_queries = repeat(torch.rand(c, d), "c d -> b c d", b=b)
    class_divisions, separated_values = segregate_samples_within_sample(
        seg_queries, keys, values
    )

    print("Class Div: ", class_divisions.shape)
    for idx in separated_values.keys():
        print("Shape for key", idx, separated_values[idx].shape)
