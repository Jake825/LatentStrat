"""Type-aware group-balanced match-breakdown reconstruction losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from latentstrat.world_model.match_breakdown.vectorize import EncodedField


def grouped_type_aware_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    encoded_fields: tuple[EncodedField, ...] | list[EncodedField],
) -> Tensor:
    """Average active features within groups, then weight active groups equally."""

    group_losses = []
    groups = sorted({field.group for field in encoded_fields})
    for group in groups:
        feature_losses = []
        for index, field in enumerate(encoded_fields):
            if field.group != group:
                continue
            active = mask[:, index] > 0
            if not torch.any(active):
                continue
            if field.kind == "numeric":
                loss = F.smooth_l1_loss(pred[active, index], target[active, index])
            else:
                loss = F.binary_cross_entropy_with_logits(
                    pred[active, index], target[active, index]
                )
            feature_losses.append(loss)
        if feature_losses:
            group_losses.append(torch.stack(feature_losses).mean())
    if not group_losses:
        return pred.sum() * 0
    return torch.stack(group_losses).mean()


def grouped_type_aware_report(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    encoded_fields: tuple[EncodedField, ...] | list[EncodedField],
) -> dict[str, float]:
    """Return detached group losses for validation diagnostics."""

    report = {}
    for group in sorted({field.group for field in encoded_fields}):
        feature_losses = []
        for index, field in enumerate(encoded_fields):
            if field.group != group:
                continue
            active = mask[:, index] > 0
            if not torch.any(active):
                continue
            if field.kind == "numeric":
                loss = F.smooth_l1_loss(pred[active, index], target[active, index])
            else:
                loss = F.binary_cross_entropy_with_logits(
                    pred[active, index], target[active, index]
                )
            feature_losses.append(loss)
        if feature_losses:
            report[group] = float(torch.stack(feature_losses).mean().detach().cpu())
    return report
