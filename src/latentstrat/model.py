"""Native PyTorch V5 model for LatentStrat."""

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
    atomic_z: Tensor
    foul_z: Tensor
    bonus_logits: Tensor
    special_logits: Tensor
    endgame_logits: Tensor
    award_logits: Tensor
    red_missing_mask: Tensor
    blue_missing_mask: Tensor
    red_dropout_mask: Tensor
    blue_dropout_mask: Tensor


class MAB(nn.Module):
    """Multihead attention block with residual feed-forward refinement."""

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

    @staticmethod
    def safe_key_padding_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        safe = mask.bool().clone()
        all_masked = safe.all(dim=1)
        if torch.any(all_masked):
            safe[all_masked] = False
        return safe

    def forward(
        self, query: Tensor, key_value: Tensor, key_padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        safe_mask = self.safe_key_padding_mask(key_padding_mask)
        attn_out, attn_weights = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=safe_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        h = self.norm1(query + attn_out)
        z = self.norm2(h + self.ffn(h))
        if key_padding_mask is not None and attn_weights.shape[-1] == key_padding_mask.shape[-1]:
            weights = attn_weights.masked_fill(key_padding_mask[:, None, :].bool(), 0.0)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
            weights = weights / denom
            all_masked = key_padding_mask.bool().all(dim=1)
            if torch.any(all_masked):
                weights[all_masked] = 0.0
            attn_weights = weights
        return z, attn_weights


class SAB(MAB):
    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return super().forward(x, x, key_padding_mask)


class CROSS(MAB):
    pass


class PMA(nn.Module):
    def __init__(self, latent_dim: int, ffn_dim: int, **block_args) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty(latent_dim))
        nn.init.normal_(self.seed, mean=0.0, std=0.02)
        self.block = MAB(latent_dim, ffn_dim, **block_args)

    def forward(
        self, h: Tensor, key_padding_mask: Tensor | None = None, mode: str = "learned"
    ) -> tuple[Tensor, Tensor]:
        if mode == "uniform":
            weights = torch.ones(
                (h.shape[0], 1, h.shape[1]), dtype=h.dtype, device=h.device
            )
            if key_padding_mask is not None:
                weights = weights.masked_fill(key_padding_mask[:, None, :].bool(), 0.0)
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(h.dtype).eps)
            weights = weights / denom
            pooled = torch.bmm(weights, h).squeeze(1)
            return pooled, weights
        if mode != "learned":
            raise ValueError('pma_mode must be "learned" or "uniform".')
        seed = self.seed.view(1, 1, -1).expand(h.shape[0], 1, -1)
        pooled, weights = self.block(seed, h, key_padding_mask)
        return pooled.squeeze(1), weights


class SiameseWinHead(nn.Module):
    """Anti-symmetric win head: swapping alliances negates the logit."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), nn.Sigmoid())
        self.score = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, z_red: Tensor, z_blue: Tensor) -> Tensor:
        diff = z_red - z_blue
        interaction = torch.cat([diff.abs(), z_red * z_blue], dim=-1)
        return self.score(diff * self.gate(interaction))


class OrdinalEndgameHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(latent_dim, max(num_classes - 1, 1))

    def forward(self, slot_context: Tensor) -> Tensor:
        logits = self.linear(slot_context)
        return logits[..., : max(self.num_classes - 1, 0)]


class JudgesRoomHead(nn.Module):
    def __init__(self, latent_dim: int, num_awards: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, num_awards)

    def forward(self, slot_context: Tensor) -> Tensor:
        return self.linear(slot_context)


class TeamValueHead(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, z_team: Tensor) -> Tensor:
        return self.linear(z_team).squeeze(-1)


class AllianceValueHead(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, z_alliance: Tensor) -> Tensor:
        return self.linear(z_alliance).squeeze(-1)


class HomoscedasticTaskBalancer(nn.Module):
    def __init__(self, task_names: tuple[str, ...]) -> None:
        super().__init__()
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in task_names}
        )

    def forward(self, task_name: str, loss: Tensor, active: bool) -> Tensor:
        if not active:
            return torch.zeros((), dtype=loss.dtype, device=loss.device)
        log_var = self.log_vars[task_name]
        return torch.exp(-log_var) * loss + log_var

    def precision(self, task_name: str) -> Tensor:
        return torch.exp(-self.log_vars[task_name])


class DeltaIntegrationGate(nn.Module):
    """Offline gate for folding event deltas into durable base embeddings."""

    def __init__(self, latent_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or max(4 * latent_dim, 8)
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_base: Tensor, z_event: Tensor, delta_weeks: Tensor) -> Tensor:
        if delta_weeks.ndim == 1:
            delta_weeks = delta_weeks.unsqueeze(-1)
        delta_weeks = delta_weeks.to(dtype=z_base.dtype, device=z_base.device)
        return self.net(torch.cat([z_base, z_event, delta_weeks], dim=-1))

    def integrate(self, z_base: Tensor, z_event: Tensor, delta_weeks: Tensor) -> Tensor:
        return z_base + self(z_base, z_event, delta_weeks) * z_event


class SetTransformerModel(nn.Module):
    """V5 cross-alliance Set Transformer for FRC match and auxiliary targets."""

    def __init__(
        self,
        num_teams: int,
        latent_dim: int,
        num_cont_targets: int,
        num_bin_targets: int,
        opts: LatentStratOptions | None = None,
        *,
        num_event_teams: int | None = None,
        num_endgame_classes: int | None = None,
        num_awards: int | None = None,
    ) -> None:
        super().__init__()
        opts = opts or default_options()
        self.opts = opts
        self.latent_dim = latent_dim
        self.num_cont_targets = num_cont_targets
        self.num_bin_targets = num_bin_targets
        self.num_endgame_classes = num_endgame_classes or len(opts.endgame_class_order)
        self.num_awards = num_awards or len(opts.award_targets)
        self.num_atomic_targets = 2 * len(opts.atomic_count_targets)
        self.num_foul_targets = 2 * len(opts.foul_targets)
        self.num_bonus_targets = 2 * len(opts.bonus_binary_targets)
        self.num_special_targets = 2 * len(opts.special_binary_targets)

        self.Z_base = nn.Embedding(max(num_teams, 1), latent_dim)
        self.Z_event = nn.Embedding(max(num_event_teams or num_teams, 1), latent_dim, padding_idx=0)
        self.team_embedding = self.Z_base
        nn.init.normal_(self.Z_base.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.Z_event.weight)

        block_args = {
            "num_heads": opts.attention_heads,
            "attention_dropout": opts.attention_dropout,
            "ffn_dropout": opts.ffn_dropout,
            "layer_norm_epsilon": opts.set_layer_norm_epsilon,
        }
        self.sab = SAB(latent_dim, opts.set_ffn_dim, **block_args)
        self.cross = CROSS(latent_dim, opts.set_ffn_dim, **block_args)
        self.pma = PMA(latent_dim, opts.set_ffn_dim, **block_args)

        self.cont_head = nn.Linear(latent_dim, max(num_cont_targets // 2, 1))
        self.atomic_head = nn.Linear(latent_dim, max(len(opts.atomic_count_targets), 1))
        self.foul_head = nn.Linear(latent_dim, max(len(opts.foul_targets), 1))
        self.bonus_head = nn.Linear(latent_dim, max(len(opts.bonus_binary_targets), 1))
        self.special_head = nn.Linear(latent_dim, max(len(opts.special_binary_targets), 1))
        self.bin_head = SiameseWinHead(latent_dim)
        self.endgame_head = OrdinalEndgameHead(latent_dim, self.num_endgame_classes)
        self.award_head = JudgesRoomHead(latent_dim, self.num_awards)
        self.team_value_head = TeamValueHead(latent_dim)
        self.alliance_value_head = AllianceValueHead(latent_dim)
        self.delta_integration_gate = DeltaIntegrationGate(latent_dim)
        self.loss_balancer = HomoscedasticTaskBalancer(
            (
                "continuous",
                "win",
                "endgame",
                "awards",
                "atomic",
                "foul",
                "bonus",
                "special",
                "rank",
                "playoff",
                "selection",
            )
        )

    def balance_loss(self, task_name: str, loss: Tensor, active: bool) -> Tensor:
        return self.loss_balancer(task_name, loss, active)

    def team_latent(
        self, team_base_idx: Tensor, team_event_idx: Tensor | None = None
    ) -> Tensor:
        base_idx = team_base_idx.long().clamp_min(0)
        event_idx = torch.zeros_like(base_idx) if team_event_idx is None else team_event_idx.long()
        return self.Z_base(base_idx) + self.Z_event(event_idx.clamp_min(0))

    def team_value(self, team_base_idx: Tensor, team_event_idx: Tensor | None = None) -> Tensor:
        return self.team_value_head(self.team_latent(team_base_idx, team_event_idx))

    def alliance_value(
        self,
        team_base_idx: Tensor,
        team_event_idx: Tensor | None = None,
        *,
        pma_mode: str = "learned",
    ) -> Tensor:
        h, _, _ = self.team_set(team_base_idx, team_event_idx)
        attention_mask = torch.zeros(
            team_base_idx.shape, dtype=torch.bool, device=team_base_idx.device
        )
        context, _ = self.sab(h, attention_mask)
        z_alliance, _ = self.pma_pool(context, attention_mask, pma_mode)
        return self.alliance_value_head(z_alliance)

    def _random_dropout_mask(self, missing_mask: Tensor) -> Tensor:
        rate = float(self.opts.team_dropout_rate)
        if not self.training or rate <= 0:
            return torch.zeros_like(missing_mask, dtype=torch.bool)
        healthy = ~missing_mask.bool()
        dropout = (torch.rand(missing_mask.shape, device=missing_mask.device) < rate) & healthy
        effective = missing_mask.bool() | dropout
        all_masked = effective.all(dim=1)
        if torch.any(all_masked):
            healthy_rows = healthy[all_masked]
            row_positions = torch.nonzero(all_masked, as_tuple=False).flatten()
            first_healthy = torch.argmax(healthy_rows.to(torch.int64), dim=1)
            dropout[row_positions, first_healthy] = False
        return dropout

    def team_set(
        self,
        team_base_idx: Tensor,
        team_event_idx: Tensor | None = None,
        missing_mask: Tensor | None = None,
        *,
        zero_slot: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        base_idx = team_base_idx.long()
        event_idx = torch.zeros_like(base_idx) if team_event_idx is None else team_event_idx.long()
        explicit_missing = base_idx.eq(0) if missing_mask is None else missing_mask.bool()
        if 1 <= zero_slot <= base_idx.shape[1]:
            explicit_missing = explicit_missing.clone()
            explicit_missing[:, zero_slot - 1] = True
        dropout_mask = self._random_dropout_mask(explicit_missing)
        effective_missing = explicit_missing | dropout_mask
        routed_base_idx = torch.where(effective_missing, torch.zeros_like(base_idx), base_idx)
        routed_event_idx = torch.where(effective_missing, torch.zeros_like(event_idx), event_idx)
        x = self.Z_base(routed_base_idx.clamp_min(0)) + self.Z_event(
            routed_event_idx.clamp_min(0)
        )
        return x, effective_missing, dropout_mask

    def pma_pool(
        self, h: Tensor, missing_mask: Tensor | None = None, pma_mode: str = "learned"
    ) -> tuple[Tensor, Tensor]:
        return self.pma(h, missing_mask, pma_mode)

    def forward(
        self,
        red_team_idx: Tensor,
        blue_team_idx: Tensor,
        *,
        red_event_idx: Tensor | None = None,
        blue_event_idx: Tensor | None = None,
        red_missing_mask: Tensor | None = None,
        blue_missing_mask: Tensor | None = None,
        pma_mode: str = "learned",
        red_zero_slot: int = 0,
        blue_zero_slot: int = 0,
    ) -> ForwardOutput:
        red_set, red_effective_missing, red_dropout = self.team_set(
            red_team_idx, red_event_idx, red_missing_mask, zero_slot=red_zero_slot
        )
        blue_set, blue_effective_missing, blue_dropout = self.team_set(
            blue_team_idx, blue_event_idx, blue_missing_mask, zero_slot=blue_zero_slot
        )
        red_attention_mask = torch.zeros_like(red_effective_missing, dtype=torch.bool)
        blue_attention_mask = torch.zeros_like(blue_effective_missing, dtype=torch.bool)

        red_context, _ = self.sab(red_set, red_attention_mask)
        blue_context, _ = self.sab(blue_set, blue_attention_mask)
        red_interacted, _ = self.cross(red_context, blue_context, blue_attention_mask)
        blue_interacted, _ = self.cross(blue_context, red_context, red_attention_mask)

        z_red, red_weights = self.pma_pool(red_interacted, red_attention_mask, pma_mode)
        z_blue, blue_weights = self.pma_pool(blue_interacted, blue_attention_mask, pma_mode)
        diff = z_red - z_blue
        z_match = torch.cat([z_red, z_blue, diff, diff.abs(), z_red * z_blue], dim=1)
        cont = torch.cat([self.cont_head(z_red), self.cont_head(z_blue)], dim=1)
        if cont.shape[1] > self.num_cont_targets:
            cont = cont[:, : self.num_cont_targets]
        atomic = torch.cat([self.atomic_head(z_red), self.atomic_head(z_blue)], dim=1)
        foul = torch.cat([self.foul_head(z_red), self.foul_head(z_blue)], dim=1)
        bonus = torch.cat([self.bonus_head(z_red), self.bonus_head(z_blue)], dim=1)
        special = torch.cat([self.special_head(z_red), self.special_head(z_blue)], dim=1)

        slot_context = torch.cat([red_interacted, blue_interacted], dim=1)
        return ForwardOutput(
            z_red=z_red,
            z_blue=z_blue,
            z_match=z_match,
            red_pma_weights=red_weights,
            blue_pma_weights=blue_weights,
            cont_z=cont,
            bin_logits=self.bin_head(z_red, z_blue),
            atomic_z=atomic[:, : self.num_atomic_targets],
            foul_z=foul[:, : self.num_foul_targets],
            bonus_logits=bonus[:, : self.num_bonus_targets],
            special_logits=special[:, : self.num_special_targets],
            endgame_logits=self.endgame_head(slot_context),
            award_logits=self.award_head(slot_context),
            red_missing_mask=red_effective_missing,
            blue_missing_mask=blue_effective_missing,
            red_dropout_mask=red_dropout,
            blue_dropout_mask=blue_dropout,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def init_model(
    num_teams: int,
    latent_dim: int,
    num_cont_targets: int,
    num_bin_targets: int,
    opts: LatentStratOptions | None = None,
    *,
    num_event_teams: int | None = None,
    num_endgame_classes: int | None = None,
    num_awards: int | None = None,
) -> SetTransformerModel:
    return SetTransformerModel(
        num_teams,
        latent_dim,
        num_cont_targets,
        num_bin_targets,
        opts,
        num_event_teams=num_event_teams,
        num_endgame_classes=num_endgame_classes,
        num_awards=num_awards,
    )


def optimizer_parameter_groups(model: SetTransformerModel, opts: LatentStratOptions) -> list[dict]:
    """Build AdamW groups while keeping embeddings on active-row custom L2."""

    set_decay = []
    head_decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_bias = name.endswith(".bias") or name.endswith("_bias")
        is_norm = ".norm" in name or "norm" in name
        is_embedding = name.startswith(("Z_base", "Z_event", "team_embedding"))
        is_seed_or_token = name == "pma.seed"
        is_log_var = name.startswith("loss_balancer.log_vars")
        if is_bias or is_norm or is_embedding or is_seed_or_token or is_log_var:
            no_decay.append(parameter)
        elif name.startswith(("sab.", "cross.", "pma.")):
            set_decay.append(parameter)
        elif name.startswith(
            (
                "cont_head.",
                "atomic_head.",
                "foul_head.",
                "bonus_head.",
                "special_head.",
                "bin_head.",
                "endgame_head.",
                "award_head.",
                "team_value_head.",
                "alliance_value_head.",
                "delta_integration_gate.",
            )
        ):
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
