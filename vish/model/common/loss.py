import sys
import warnings
from typing import Any, Callable

import torch
from torch import Tensor, nn

from vish.constants import CIFAR10_NUM_BROAD

MODE_0 = 0


def current_mode(curr_epoch, alt_freq, num_modes=2):
    val, _ = divmod(curr_epoch, alt_freq)
    return val % num_modes


def convert_fine_to_broad_logits(
    logits: Tensor,
    broad_labels: Tensor,
    fine_labels: Tensor,
    num_broad=CIFAR10_NUM_BROAD,
):
    """
    Converts fine logits to broad logits for CIFAR 10

    Args:
        logits: logits of shape [B, 1, C_fine] or [B, C_fine]
        broad_labels: Labels of shape [B, ]
        fine_labels: Labels of shape [B, ]

    Returns:

    """

    if len(logits.shape) == 3 and logits.shape[1] == 1:
        logits = logits.squeeze(1)

    batch, *_ = logits.shape
    b_logits = torch.empty((batch, 0), device=logits.device)

    for b_idx in range(num_broad):
        fine_indexes = torch.unique(fine_labels[broad_labels == b_idx])

        if torch.numel(fine_indexes) == 0:
            curr_idx_logits = torch.zeros(
                (batch,), device=logits.device, requires_grad=True
            )
        else:
            curr_idx_logits, _ = torch.max(logits[:, fine_indexes], dim=1)
            # curr_idx_logits = torch.mean(logits[:, fine_indexes], dim=1)
        curr_idx_logits = curr_idx_logits.unsqueeze(1)

        b_logits = torch.cat([b_logits, curr_idx_logits], dim=1)

    return b_logits


def bnf_alternate_loss(
    broad_outputs: list[torch.FloatTensor],
    fine_outputs: list[torch.FloatTensor],
    broad_labels: torch.IntTensor,
    fine_labels: torch.IntTensor,
    curr_epoch: int,
    classifier: Callable[[Tensor], Tensor],
    f2b_filter: Callable[
        [Tensor, Tensor, Tensor], Tensor
    ] = convert_fine_to_broad_logits,
    scale_factor: float = 100.0,
    alt_freq: int = 10,
    p: int = 1,
):
    """
    Computes Broad and Fine Embedding Loss of Dual Models
    Args:
        scale_factor:
        f2b_filter: Fine to broad Label filter
        classifier: The Fine classifier, input is [B, 1, D]
        curr_epoch: Current epoch
        alt_freq: Alternating frequency
        fine_labels (torch.IntTensor): Fine Labels tensor
        broad_labels (torch.IntTensor): Broad Labels Tensor
        fine_outputs (list[torch.FloatTensor]): [fine_embedding, fine_logits],
        broad_outputs (torch.FloatTensor): [broad_embedding, broad_logits]
        p (int): [>= 1], the value of the norm to be used

    Returns:
        tuple(torch.float32, torch.float32): broad_loss, fine_loss

    Todo:
        1. Do not use dynamic number of broad class calculations. This gives error
        If a batch has only 1 type of samples, num_unique_broad = 1 and empty is possible

    """
    # Fine logits is list as adapted from a multiple token model
    ce_loss_criterion = nn.CrossEntropyLoss()
    fine_embedding, [*_, fine_logits] = fine_outputs
    broad_embedding, *_ = broad_outputs

    mode = current_mode(curr_epoch, alt_freq)

    if mode == MODE_0:
        broad_embedding_clone = make_frozen_clone(broad_embedding)
        emb_losses = _loss_bnf(
            broad_embedding_clone, broad_labels, fine_embedding, fine_labels, p
        )
        broad_emb_from_fine = classifier(broad_embedding)

        if isinstance(broad_emb_from_fine, (list, tuple)):
            broad_emb_from_fine = broad_emb_from_fine[-1]

        broad_predictions = f2b_filter(
            broad_emb_from_fine,
            broad_labels,
            fine_labels,
        )

        loss_embedding = torch.mean(torch.stack(emb_losses))
        loss_ce = ce_loss_criterion(broad_predictions, broad_labels)

    else:
        fine_embedding_clone = make_frozen_clone(fine_embedding)
        emb_losses = _loss_bnf(
            broad_embedding, broad_labels, fine_embedding_clone, fine_labels, p
        )

        loss_embedding = torch.mean(torch.stack(emb_losses))
        loss_ce = ce_loss_criterion(fine_logits, fine_labels)

    if scale_factor:
        loss_embedding = loss_embedding / scale_factor

    return loss_embedding, loss_ce


def _loss_bnf(
    broad_embedding,
    broad_labels,
    fine_embedding,
    fine_labels,
    p=1,
) -> list[Tensor]:
    """
    Computes L1 distance based Broad Fine loss for the minibatch
    Args:
        broad_embedding: Broad Embeddings of shape [B, N, D]
        broad_labels: Broad Labels of shape [B, C_Broad]
        fine_embedding: Fine Embeddings of shape [B, N, D]
        fine_labels: Fine Labels of shape [B, C_Fine]
        p: The Norm value, default = 1

    Returns:
        list[Tensor], a list of 0-D tensor of losses for each broad label

    """
    broad_indexes = torch.unique(broad_labels).tolist()

    emb_losses = []
    for broad_idx in broad_indexes:
        fine_indexes = torch.unique(fine_labels[broad_labels == broad_idx])

        if len(fine_indexes) == 0:
            # No fine classes found results in NaN for the batch
            # This can arise during data loading and shuffling
            warnings.warn(f"Found fine Indexes {fine_indexes} for label {broad_idx}")
            continue

        broad_mean = mean_sqz_empty(
            broad_embedding[broad_labels == broad_idx]
        )  # only one index, dim: [D]
        fine_this_idx = [
            mean_sqz_empty(fine_embedding[(fine_labels == fine_idx)])
            for fine_idx in fine_indexes
        ]

        fine_mean = torch.mean(torch.cat(fine_this_idx, dim=0), 0)

        emb_losses.append(torch.nan_to_num(broad_criterion(broad_mean, fine_mean, p)))
    return emb_losses


def bnf_embedding_cluster_loss(
    broad_outputs: list[torch.FloatTensor],
    fine_outputs: list[torch.FloatTensor],
    broad_labels: torch.IntTensor,
    fine_labels: torch.IntTensor,
    fine_criterion: Any,
    p=1.0,
):
    """
    Computes Broad and Fine Embedding Loss of Dual Models, using KMeans like loss
    Args:
        fine_criterion: Criterion for Fine Loss
        fine_labels (torch.IntTensor): Fine Labels tensor
        broad_labels (torch.IntTensor): Broad Labels Tensor
        fine_outputs (list[torch.FloatTensor]): [fine_embedding, fine_logits],
        broad_outputs (list[torch.FloatTensor]): [broad_embedding, broad_logits]
        p (float): [>= 1], the value of the norm to be used

    Returns:
        tuple(torch.float32, torch.float32): broad_loss, fine_loss

    """
    # Fine logits is list as adapted from a multiple token model
    fine_embedding, [*_, fine_logits] = fine_outputs
    broad_embedding, broad_logits = broad_outputs

    fine_embedding_clone = make_frozen_clone(fine_embedding)

    num_broad_classes = broad_logits.shape[-1]
    emb_losses = []

    for broad_idx in range(num_broad_classes):
        fine_indexes = torch.unique(fine_labels[broad_labels == broad_idx])

        if len(fine_indexes) == 0:
            continue

        broad_this_idx = mean_sqz_empty(broad_embedding[broad_labels == broad_idx])
        fine_e = [
            mean_sqz_empty(fine_embedding_clone[(fine_labels == fine_idx)])
            for fine_idx in fine_indexes
        ]

        if len(fine_e) == 0:
            continue

        fine_this_idx = torch.stack(fine_e if len(fine_e) > 0 else [torch.zeros])

        emb_losses.append(
            torch.nan_to_num(broad_criterion(broad_this_idx, fine_this_idx, p))
        )

    fine_loss = fine_criterion(fine_logits, fine_labels)
    broad_loss = torch.mean(torch.stack(emb_losses))

    return broad_loss, fine_loss


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
    # Fine logits is list as adapted from a multiple token model
    fine_embedding, [*_, fine_logits] = fine_outputs
    broad_embedding, broad_logits = broad_outputs

    fine_embedding_clone = make_frozen_clone(fine_embedding)

    num_broad_classes = broad_logits.shape[-1]

    emb_losses = []

    for broad_idx in range(num_broad_classes):
        fine_indexes = torch.unique(fine_labels[broad_labels == broad_idx])

        broad_this_idx = mean_sqz_empty(broad_embedding[broad_labels == broad_idx])
        fine_this_idx = [
            mean_sqz_empty(fine_embedding_clone[(fine_labels == fine_idx)])
            for fine_idx in fine_indexes
        ]

        broad_mean = broad_this_idx  # only one index, D

        if len(fine_this_idx) == 0:
            fine_this_idx = [torch.zeros_like(broad_mean)]

        fine_mean = torch.mean(torch.cat(fine_this_idx, dim=0), 0)

        emb_losses.append(torch.nan_to_num(broad_criterion(broad_mean, fine_mean, p)))

    fine_loss = fine_criterion(fine_logits, fine_labels)
    broad_loss = torch.mean(torch.stack(emb_losses))

    return broad_loss, fine_loss


def mean_empty(t: torch.FloatTensor):
    # Z 1 D -> 1 D
    return empty_if_problem(torch.mean(t, 0))


def mean_sqz_empty(t: torch.FloatTensor):
    # Z, 1, D -> D
    return empty_if_problem(torch.mean(t, 0).squeeze(0))


def make_frozen_clone(fine_embedding):
    fine_embedding_clone = fine_embedding.detach().clone()
    fine_embedding_clone.requires_grad = False
    return fine_embedding_clone


def is_problem_tensor(tensor):
    return torch.isnan(tensor).sum() > 0


def empty_if_problem(tensor: torch.Tensor):
    _, *rest = tensor.shape
    if is_problem_tensor(tensor):
        return torch.empty((0, *rest), device=tensor.device, requires_grad=True)
    return tensor


def get_safe_zero(tensor: torch.Tensor):
    return torch.tensor(0.0, device=tensor.device, requires_grad=True)


def broad_criterion(x: torch.Tensor, y: torch.Tensor, p: float | str = 1):
    return torch.linalg.norm(x - y, p)
