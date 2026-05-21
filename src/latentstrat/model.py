"""Native PyTorch model for LatentStrat."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from latentstrat.config import LatentStratOptions, default_options


@dataclass
class ForwardOutput:
    z_red: Tensor
    z_blue: Tensor
    z_match: Tensor
    red_pma_weights: Tensor
    blue_pma_weights: Tensor
    cont_z: Tensor
    bin_logits: Tensor


class SetAttentionBlock(nn.Module):
    """Transformer-style attention block for unordered team sets."""

    def __init__(
        self,
        latent_dim: int,
        ffn_dim: int,
        *,
        num_heads: int = 1,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by attention_heads.")
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(latent_dim, eps=layer_norm_epsilon)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_dim, latent_dim),
            nn.Dropout(ffn_dropout),
        )
        self.norm2 = nn.LayerNorm(latent_dim, eps=layer_norm_epsilon)

    def forward(self, query: Tensor, key_value: Tensor) -> tuple[Tensor, Tensor]:
        attn_out, attn_weights = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=True,
        )
        h = self.norm1(query + attn_out)
        z = self.norm2(h + self.ffn(h))
        return z, attn_weights


class SetTransformerModel(nn.Module):
    """Cross-alliance Set Transformer for FRC match prediction."""

    def __init__(
        self,
        num_teams: int,
        latent_dim: int,
        num_cont_targets: int,
        num_bin_targets: int,
        opts: LatentStratOptions | None = None,
    ) -> None:
        super().__init__()
        opts = opts or default_options()
        self.opts = opts
        self.latent_dim = latent_dim
        self.team_embedding = nn.Embedding(num_teams, latent_dim)
        nn.init.normal_(self.team_embedding.weight, mean=0.0, std=0.02)

        block_args = {
            "num_heads": opts.attention_heads,
            "attention_dropout": opts.attention_dropout,
            "ffn_dropout": opts.ffn_dropout,
            "layer_norm_epsilon": opts.set_layer_norm_epsilon,
        }
        self.sab = SetAttentionBlock(latent_dim, opts.set_ffn_dim, **block_args)
        self.cross = SetAttentionBlock(latent_dim, opts.set_ffn_dim, **block_args)
        self.pma = SetAttentionBlock(latent_dim, opts.set_ffn_dim, **block_args)
        self.pma_seed = nn.Parameter(torch.empty(latent_dim))
        nn.init.normal_(self.pma_seed, mean=0.0, std=0.02)

        head_dim = 5 * latent_dim
        self.cont_head = nn.Linear(head_dim, num_cont_targets)
        self.bin_head = nn.Linear(head_dim, num_bin_targets)

    def team_set(self, team_idx: Tensor, zero_slot: int = 0) -> Tensor:
        x = self.team_embedding(team_idx.long())
        if 1 <= zero_slot <= x.shape[1]:
            x = x.clone()
            x[:, zero_slot - 1, :] = 0
        return x

    def pma_pool(self, h: Tensor, pma_mode: str = "learned") -> tuple[Tensor, Tensor]:
        if pma_mode == "uniform":
            weights = torch.full(
                (h.shape[0], 1, h.shape[1]),
                1.0 / h.shape[1],
                dtype=h.dtype,
                device=h.device,
            )
            return h.mean(dim=1), weights
        if pma_mode != "learned":
            raise ValueError('pma_mode must be "learned" or "uniform".')

        seed = self.pma_seed.view(1, 1, -1).expand(h.shape[0], 1, -1)
        pooled, weights = self.pma(seed, h)
        return pooled.squeeze(1), weights

    def forward(
        self,
        red_team_idx: Tensor,
        blue_team_idx: Tensor,
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
        z_match = torch.cat([z_red, z_blue, diff, diff.abs(), z_red * z_blue], dim=1)

        return ForwardOutput(
            z_red=z_red,
            z_blue=z_blue,
            z_match=z_match,
            red_pma_weights=red_weights,
            blue_pma_weights=blue_weights,
            cont_z=self.cont_head(z_match),
            bin_logits=self.bin_head(z_match),
        )

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


def optimizer_parameter_groups(model: SetTransformerModel, opts: LatentStratOptions) -> list[dict]:
    """Build AdamW groups with decay only on standard linear/attention weights."""

    set_decay = []
    head_decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_bias = name.endswith(".bias") or name.endswith("_bias")
        is_norm = ".norm" in name or "norm" in name
        is_embedding = name.startswith("team_embedding")
        is_seed = name == "pma_seed"
        if is_bias or is_norm or is_embedding or is_seed:
            no_decay.append(parameter)
        elif name.startswith(("sab.", "cross.", "pma.")):
            set_decay.append(parameter)
        elif name.startswith(("cont_head.", "bin_head.")):
            head_decay.append(parameter)
        else:
            no_decay.append(parameter)

    groups = []
    if set_decay:
        groups.append({"params": set_decay, "weight_decay": opts.l2_set})
    if head_decay:
        groups.append({"params": head_decay, "weight_decay": opts.l2_heads})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups
