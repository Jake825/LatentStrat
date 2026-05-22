"""Transductive V5.6 prior distiller."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from latentstrat.config import PriorOpts


class TeamPriorDistiller(nn.Module):
    """Learn a fixed team-number dictionary for text and EPA targets."""

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
        self.epa_head = nn.Linear(512, 1)
        self.log_var_openai = nn.Parameter(torch.zeros(()))
        self.log_var_epa = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.team_embedding.weight, mean=0.0, std=0.02)

    def forward(self, team_number: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_latent = self.team_embedding(team_number.long())
        hidden = self.decoder(z_latent)
        return self.openai_head(hidden), self.epa_head(hidden), z_latent

    def embedding_table(self) -> Tensor:
        return self.team_embedding.weight.detach().clone()


def prior_distillation_losses(
    model: TeamPriorDistiller,
    team_number: Tensor,
    target_embedding: Tensor,
    target_epa: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    valid = team_number.long() >= 0
    if not torch.any(valid):
        zero = torch.zeros((), dtype=target_embedding.dtype, device=target_embedding.device)
        return zero, zero, zero
    prediction, pred_epa, _ = model(team_number[valid])
    target_epa = target_epa[valid].to(dtype=prediction.dtype, device=prediction.device)
    target_epa = target_epa.view_as(pred_epa)
    openai_mse = F.mse_loss(prediction, target_embedding[valid])
    epa_mse = F.mse_loss(pred_epa, target_epa)
    total = (
        torch.exp(-model.log_var_openai) * openai_mse
        + model.log_var_openai
        + torch.exp(-model.log_var_epa) * epa_mse
        + model.log_var_epa
    )
    return total, openai_mse, epa_mse


def distillation_loss(
    model: TeamPriorDistiller,
    team_number: Tensor,
    target_embedding: Tensor,
    target_epa: Tensor,
) -> Tensor:
    return prior_distillation_losses(model, team_number, target_embedding, target_epa)[0]
