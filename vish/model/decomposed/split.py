import torch
from torch import Tensor


def segregate_samples_within_batch(
    class_queries: torch.FloatTensor,
    keys: torch.FloatTensor,
    values: torch.FloatTensor,
) -> tuple[Tensor, dict]:
    """
    Segregates Values based on given Query Vectors

    Args:
        values: (torch.FloatTensor) Samples of shape [batch, seq, emb_dim]
        keys: (torch.FloatTensor) Samples of shape [batch, seq, emb_dim]
        class_queries: (torch.FloatTensor) Queries of shape [batch, num_classes, emb_dim]

    Return:
        tuple[Tensor, Dict[int, torch.FloatTensor]] The segregated values based on keys and
        the segregating class indexes
    """
    num_classes = class_queries.shape[1]

    dot_product = class_queries @ keys.transpose(-1, -2)
    class_division = torch.argmax(dot_product, dim=1)

    return class_division, {
        class_idx: values[class_division == class_idx].unsqueeze(0)
        for class_idx in range(num_classes)
    }


@DeprecationWarning
def segregate_samples_within_sample(
    seg_queries: torch.FloatTensor, keys: torch.FloatTensor, values: torch.FloatTensor
) -> tuple[Tensor, dict]:
    batch_dim, num_classes, emb_dim = seg_queries.shape

    dot_product = seg_queries @ keys.transpose(-1, -2)
    print(dot_product.shape)

    class_division = torch.argmax(dot_product, dim=1)
    print("Class division shape is: ", class_division.shape)

    return class_division, {
        class_idx: torch.masked_select(values, dot_product)
        for class_idx in range(num_classes)
    }
