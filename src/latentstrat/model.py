"""PyTorch implementation of the V4 cross-alliance Set Transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from latentstrat.config import LatentStratOptions, default_options

try:  # pragma: no cover - exercised when torch is installed
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


def require_torch() -> Any:
    if torch is None or nn is None:
        raise ModuleNotFoundError("PyTorch is required for latentstrat.model.")
    return torch


@dataclass
class ForwardOutput:
    z_red: Any
    z_blue: Any
    z_match: Any
    red_pma_weights: Any
    blue_pma_weights: Any
    cont_z: Any
    bin_logits: Any


class MABBlock(nn.Module if nn is not None else object):
    def __init__(self, latent_dim: int, ffn_dim: int, epsilon: float) -> None:
        require_torch()
        super().__init__()
        projection_scale = math.sqrt(2 / (latent_dim + latent_dim))
        ffn_scale = math.sqrt(2 / latent_dim)
        output_scale = math.sqrt(2 / ffn_dim)
        self.epsilon = epsilon
        self.w_q = nn.Parameter(torch.randn(latent_dim, latent_dim) * projection_scale)
        self.w_k = nn.Parameter(torch.randn(latent_dim, latent_dim) * projection_scale)
        self.w_v = nn.Parameter(torch.randn(latent_dim, latent_dim) * projection_scale)
        self.w_o = nn.Parameter(torch.randn(latent_dim, latent_dim) * projection_scale)
        self.w_f1 = nn.Parameter(torch.randn(ffn_dim, latent_dim) * ffn_scale)
        self.b_f1 = nn.Parameter(torch.zeros(ffn_dim))
        self.w_f2 = nn.Parameter(torch.randn(latent_dim, ffn_dim) * output_scale)
        self.b_f2 = nn.Parameter(torch.zeros(latent_dim))
        self.g1 = nn.Parameter(torch.ones(latent_dim))
        self.b1 = nn.Parameter(torch.zeros(latent_dim))
        self.g2 = nn.Parameter(torch.ones(latent_dim))
        self.b2 = nn.Parameter(torch.zeros(latent_dim))

    @staticmethod
    def project(weights: Any, x: Any) -> Any:
        return x @ weights.t()

    def layer_norm(self, x: Any, gamma: Any, beta: Any) -> Any:
        mu = x.mean(dim=-1, keepdim=True)
        variance = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
        normalized = (x - mu) / torch.sqrt(variance + self.epsilon)
        return gamma.view(1, 1, -1) * normalized + beta.view(1, 1, -1)

    def forward(self, x: Any, y: Any) -> tuple[Any, Any]:
        latent_dim = x.shape[-1]
        q = self.project(self.w_q, x)
        k = self.project(self.w_k, y)
        v = self.project(self.w_v, y)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(latent_dim)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attention = torch.exp(scores)
        attention = attention / attention.sum(dim=-1, keepdim=True)
        context = torch.matmul(attention, v)
        out = self.project(self.w_o, context)
        h = self.layer_norm(x + out, self.g1, self.b1)
        ffn_hidden = torch.relu(self.project(self.w_f1, h) + self.b_f1.view(1, 1, -1))
        ffn_out = self.project(self.w_f2, ffn_hidden) + self.b_f2.view(1, 1, -1)
        z = self.layer_norm(h + ffn_out, self.g2, self.b2)
        return z, attention

    def l2_fields(self) -> list[Any]:
        return [self.w_q, self.w_k, self.w_v, self.w_o, self.w_f1, self.w_f2, self.g1, self.g2]


class SetTransformerModel(nn.Module if nn is not None else object):
    def __init__(
        self,
        num_teams: int,
        latent_dim: int,
        num_cont_targets: int,
        num_bin_targets: int,
        opts: LatentStratOptions | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        opts = opts or default_options()
        self.opts = opts
        self.latent_dim = latent_dim
        self.team_embedding = nn.Embedding(num_teams, latent_dim)
        nn.init.normal_(self.team_embedding.weight, mean=0.0, std=0.01)
        self.sab = MABBlock(latent_dim, opts.set_ffn_dim, opts.set_layer_norm_epsilon)
        self.cross = MABBlock(latent_dim, opts.set_ffn_dim, opts.set_layer_norm_epsilon)
        self.pma = MABBlock(latent_dim, opts.set_ffn_dim, opts.set_layer_norm_epsilon)
        self.pma_seed = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.w_cont = nn.Parameter(torch.randn(num_cont_targets, 5 * latent_dim) * 0.01)
        self.b_cont = nn.Parameter(torch.zeros(num_cont_targets))
        self.w_bin = nn.Parameter(torch.randn(num_bin_targets, 5 * latent_dim) * 0.01)
        self.b_bin = nn.Parameter(torch.zeros(num_bin_targets))

    def team_set(self, team_idx: Any, zero_slot: int = 0) -> Any:
        if team_idx.dtype != torch.long:
            team_idx = team_idx.long()
        x = self.team_embedding(team_idx)
        if zero_slot >= 1 and zero_slot <= 3:
            x = x.clone()
            x[:, zero_slot - 1, :] = 0
        return x

    def pma_pool(self, h: Any, pma_mode: str = "learned") -> tuple[Any, Any]:
        batch_size = h.shape[0]
        if pma_mode == "uniform":
            weights = torch.ones(batch_size, 1, h.shape[1], dtype=h.dtype, device=h.device) / h.shape[1]
            return h.mean(dim=1), weights
        if pma_mode != "learned":
            raise ValueError('pma_mode must be "learned" or "uniform".')
        seed = self.pma_seed.view(1, 1, -1).expand(batch_size, 1, -1)
        z3d, weights = self.pma(seed, h)
        return z3d.reshape(batch_size, self.latent_dim), weights

    def forward(
        self,
        red_team_idx: Any,
        blue_team_idx: Any,
        *,
        pma_mode: str = "learned",
        red_zero_slot: int = 0,
        blue_zero_slot: int = 0,
    ) -> ForwardOutput:
        red_set = self.team_set(red_team_idx, red_zero_slot)
        blue_set = self.team_set(blue_team_idx, blue_zero_slot)
        red_context, _ = self.sab(red_set, red_set)
        blue_context, _ = self.sab(blue_set, blue_set)
        red_interacted, _ = self.cross(red_context, blue_context)
        blue_interacted, _ = self.cross(blue_context, red_context)
        z_red, red_weights = self.pma_pool(red_interacted, pma_mode)
        z_blue, blue_weights = self.pma_pool(blue_interacted, pma_mode)
        diff = z_red - z_blue
        z_match = torch.cat([z_red, z_blue, diff, torch.abs(diff), z_red * z_blue], dim=1)
        cont_z = z_match @ self.w_cont.t() + self.b_cont
        bin_logits = z_match @ self.w_bin.t() + self.b_bin
        return ForwardOutput(
            z_red=z_red,
            z_blue=z_blue,
            z_match=z_match,
            red_pma_weights=red_weights,
            blue_pma_weights=blue_weights,
            cont_z=cont_z,
            bin_logits=bin_logits,
        )

    def set_l2_parameters(self) -> list[Any]:
        return self.sab.l2_fields() + self.cross.l2_fields() + self.pma.l2_fields() + [self.pma_seed]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def init_model(
    num_teams: int,
    latent_dim: int,
    num_cont_targets: int,
    num_bin_targets: int,
    opts: LatentStratOptions | None = None,
) -> SetTransformerModel:
    return SetTransformerModel(num_teams, latent_dim, num_cont_targets, num_bin_targets, opts)
