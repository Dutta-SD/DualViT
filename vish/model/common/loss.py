from typing import Any

import torch


def bnf_embedding_loss(
    broad_outputs: list[torch.FloatTensor],
    fine_outputs: list[torch.FloatTensor],
    broad_labels: torch.IntTensor,
    fine_labels: torch.IntTensor,
    fine_criterion: Any,
    p=1.0,
):
    """
    Computes Broad and Fine Embedding Loss of Dual Models
    Args:
        fine_criterion: Criterion for Fine Loss
        fine_labels (torch.IntTensor): Fine Labels tensor
        broad_labels (torch.IntTensor): Broad Labels Tensor
        fine_outputs (list[torch.FloatTensor]): [fine_embedding, fine_logits],
        broad_outputs (torch.FloatTensor): [broad_embedding, broad_logits]
        p (float): [>= 1], the value of the norm to be used

    Returns:
        tuple(torch.float32, torch.float32): broad_loss, fine_loss

    """
    # Fine logits is list as adapted from multiple token model
    fine_embedding, [*_, fine_logits] = fine_outputs
    broad_embedding, broad_logits = broad_outputs

    fine_embedding_clone = make_tensor_clone(fine_embedding)

    num_broad_classes = broad_logits.shape[-1]
    emb_dim = fine_embedding.shape[-1]

    emb_losses = []

    for broad_idx in range(num_broad_classes):
        fine_indexes = torch.unique(fine_labels[broad_labels == broad_idx])

        broad_this_idx = _emz(broad_embedding[broad_labels == broad_idx])
        fine_this_idx = [
            _emuz(fine_embedding_clone[(fine_labels == fine_idx)])
            for fine_idx in fine_indexes
        ]

        broad_mean = broad_this_idx # only 1 index, D

        if len(fine_this_idx) == 0:
            fine_this_idx = [torch.zeros_like(broad_mean)]

        fine_mean = torch.mean(torch.cat(fine_this_idx, dim=0), 0)

        emb_losses.append(torch.nan_to_num(broad_criterion(broad_mean, fine_mean, p)))

    fine_loss = fine_criterion(fine_logits, fine_labels)
    broad_loss = torch.mean(torch.stack(emb_losses))

    return broad_loss, fine_loss


def _emuz(t: torch.FloatTensor):
    # Z 1 D -> 1 D
    return empty_if_problem(torch.mean(t, 0))


def _emz(t: torch.FloatTensor):
    # Z, 1, D -> D
    return empty_if_problem(torch.mean(t, 0).squeeze(0))


def make_tensor_clone(fine_embedding):
    fine_embedding_clone = fine_embedding.detach().clone()
    fine_embedding_clone.requires_grad = False
    return fine_embedding_clone


def is_problem_tensor(tensor):
    return torch.isnan(tensor).sum() > 0


def empty_if_problem(tensor: torch.IntTensor):
    _, *rest = tensor.shape
    if is_problem_tensor(tensor):
        return torch.empty((0, *rest), device=tensor.device, requires_grad=True)
    return tensor


def get_safe_zero(tensor: torch.Tensor):
    return torch.tensor(0.0, device=tensor.device, requires_grad=True)


def broad_criterion(x: torch.Tensor, y: torch.Tensor, p: float | str = 1):
    return torch.linalg.norm(x - y, p)
