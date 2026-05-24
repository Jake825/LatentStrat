"""Transductive V5.6 prior distiller."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from latentstrat.config import PriorOpts
from latentstrat.pretrain_features import CULTURE_TARGET_COLUMNS, NORM_EPA_TARGET_COLUMNS


@dataclass(frozen=True)
class PriorLosses:
    total: Tensor
    openai_mse: Tensor
    norm_epa_mse: Tensor
    culture_mse: Tensor
    norm_epa_axis_mse: Tensor
    culture_axis_mse: Tensor


class TeamPriorDistiller(nn.Module):
    """Learn a fixed team-number dictionary for text, EPA trajectory, and culture targets."""

    def __init__(self, opts: PriorOpts | None = None) -> None:
        super().__init__()
        opts = opts or PriorOpts()
        self.opts = opts
        self.team_embedding = nn.Embedding(
            opts.max_team_number + 1,
            opts.latent_dim,
        )
        self.decoder = nn.Sequential(
            nn.Linear(opts.latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.openai_head = nn.Linear(512, opts.llm_dim)
        self.norm_epa_head = nn.Linear(512, len(NORM_EPA_TARGET_COLUMNS))
        self.culture_head = nn.Linear(512, len(CULTURE_TARGET_COLUMNS))
        self.log_var_openai = nn.Parameter(torch.zeros(()))
        self.log_var_epa = nn.Parameter(torch.zeros(()))
        self.log_var_culture = nn.Parameter(torch.zeros(len(CULTURE_TARGET_COLUMNS)))
        nn.init.normal_(self.team_embedding.weight, mean=0.0, std=0.02)

    def forward(self, team_number: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z_latent = self.team_embedding(team_number.long())
        hidden = self.decoder(z_latent)
        return (
            self.openai_head(hidden),
            self.norm_epa_head(hidden),
            self.culture_head(hidden),
            z_latent,
        )

    def embedding_table(self) -> Tensor:
        return self.team_embedding.weight.detach().clone()


def _zero_like_target(target_embedding: Tensor) -> Tensor:
    return torch.zeros((), dtype=target_embedding.dtype, device=target_embedding.device)


def _axis_masked_mse(
    prediction: Tensor,
    target: Tensor,
    observed: Tensor,
) -> tuple[Tensor, Tensor]:
    losses = []
    active = []
    for axis in range(prediction.shape[1]):
        finite = observed[:, axis].bool() & torch.isfinite(target[:, axis])
        if torch.any(finite):
            losses.append(F.mse_loss(prediction[finite, axis], target[finite, axis]))
            active.append(True)
        else:
            losses.append(
                torch.full(
                    (),
                    float("nan"),
                    dtype=prediction.dtype,
                    device=prediction.device,
                )
            )
            active.append(False)
    return torch.stack(losses), torch.as_tensor(active, dtype=torch.bool, device=prediction.device)


def _homoscedastic_axis_loss(axis_mse: Tensor, active: Tensor, log_vars: Tensor) -> Tensor:
    if not torch.any(active):
        return torch.zeros((), dtype=axis_mse.dtype, device=axis_mse.device)
    mse = axis_mse[active]
    log_var = log_vars[active].to(dtype=axis_mse.dtype, device=axis_mse.device)
    return torch.sum(torch.exp(-log_var) * mse + log_var)


def _trajectory_masked_mse(
    prediction: Tensor,
    target: Tensor,
    observed: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    mask = observed.bool() & torch.isfinite(target)
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    sq_err = (prediction - safe_target) ** 2
    masked_sq_err = torch.where(mask, sq_err, torch.zeros_like(prediction))
    valid_count = mask.sum()
    if int(valid_count.detach().cpu()) == 0:
        zero = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
        axis_mse = torch.full(
            (prediction.shape[1],),
            torch.nan,
            dtype=prediction.dtype,
            device=prediction.device,
        )
        active = torch.zeros(prediction.shape[1], dtype=torch.bool, device=prediction.device)
        return zero, axis_mse, active
    mse = (masked_sq_err.sum() / valid_count.clamp(min=1)) * prediction.shape[1]
    axis_mse, active = _axis_masked_mse(prediction, target, mask)
    return mse, axis_mse, active


def prior_distillation_losses(
    model: TeamPriorDistiller,
    team_number: Tensor,
    target_embedding: Tensor,
    target_norm_epa: Tensor,
    norm_epa_observed: Tensor,
    target_culture: Tensor,
) -> PriorLosses:
    valid = team_number.long() >= 0
    if not torch.any(valid):
        zero = _zero_like_target(target_embedding)
        norm_epa_axis = torch.full(
            (len(NORM_EPA_TARGET_COLUMNS),),
            torch.nan,
            dtype=target_embedding.dtype,
            device=target_embedding.device,
        )
        culture_axis = torch.full(
            (len(CULTURE_TARGET_COLUMNS),),
            torch.nan,
            dtype=target_embedding.dtype,
            device=target_embedding.device,
        )
        return PriorLosses(zero, zero, zero, zero, norm_epa_axis, culture_axis)

    prediction, pred_norm_epa, pred_culture, _ = model(team_number[valid])
    target_embedding = target_embedding[valid].to(
        dtype=prediction.dtype,
        device=prediction.device,
    )
    target_norm_epa = target_norm_epa[valid].to(
        dtype=prediction.dtype,
        device=prediction.device,
    )
    norm_epa_observed = norm_epa_observed[valid].to(device=prediction.device)
    target_culture = target_culture[valid].to(dtype=prediction.dtype, device=prediction.device)

    openai_mse = torch.mean(torch.sum((prediction - target_embedding) ** 2, dim=1))
    norm_epa_mse, norm_epa_axis_mse, norm_epa_active = _trajectory_masked_mse(
        pred_norm_epa,
        target_norm_epa,
        norm_epa_observed,
    )
    culture_observed = torch.ones_like(target_culture, dtype=torch.bool)
    culture_axis_mse, culture_active = _axis_masked_mse(
        pred_culture,
        target_culture,
        culture_observed,
    )
    norm_epa_term = (
        torch.exp(-model.log_var_epa) * norm_epa_mse + model.log_var_epa
        if torch.any(norm_epa_active)
        else torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    )
    culture_term = _homoscedastic_axis_loss(
        culture_axis_mse,
        culture_active,
        model.log_var_culture,
    )
    culture_mse = torch.nanmean(culture_axis_mse)
    total = (
        torch.exp(-model.log_var_openai) * openai_mse
        + model.log_var_openai
        + norm_epa_term
        + culture_term
    )
    return PriorLosses(
        total=total,
        openai_mse=openai_mse,
        norm_epa_mse=norm_epa_mse,
        culture_mse=culture_mse,
        norm_epa_axis_mse=norm_epa_axis_mse,
        culture_axis_mse=culture_axis_mse,
    )


def distillation_loss(
    model: TeamPriorDistiller,
    team_number: Tensor,
    target_embedding: Tensor,
    target_norm_epa: Tensor,
    norm_epa_observed: Tensor,
    target_culture: Tensor,
) -> Tensor:
    return prior_distillation_losses(
        model,
        team_number,
        target_embedding,
        target_norm_epa,
        norm_epa_observed,
        target_culture,
    ).total
