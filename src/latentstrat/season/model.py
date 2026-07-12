"""Native PyTorch V5 model for LatentStrat."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from latentstrat.config import LatentStratOptions, default_options
from latentstrat.experimental.frozen_targets import WorldModelOptions


@dataclass
class MatchRepresentations:
    z_red: Tensor
    z_blue: Tensor
    z_match: Tensor
    red_pma_weights: Tensor
    blue_pma_weights: Tensor


@dataclass
class TaskPredictions:
    values: dict[str, Tensor]

    def __getitem__(self, name: str) -> Tensor:
        return self.values[name]


@dataclass
class SlotMasks:
    red_missing_mask: Tensor
    blue_missing_mask: Tensor
    red_dropout_mask: Tensor
    blue_dropout_mask: Tensor


@dataclass
class ForwardOutput:
    representations: MatchRepresentations
    predictions: TaskPredictions
    masks: SlotMasks

    @property
    def z_red(self) -> Tensor:
        return self.representations.z_red

    @property
    def z_blue(self) -> Tensor:
        return self.representations.z_blue

    @property
    def z_match(self) -> Tensor:
        return self.representations.z_match

    @property
    def red_pma_weights(self) -> Tensor:
        return self.representations.red_pma_weights

    @property
    def blue_pma_weights(self) -> Tensor:
        return self.representations.blue_pma_weights

    @property
    def red_missing_mask(self) -> Tensor:
        return self.masks.red_missing_mask

    @property
    def blue_missing_mask(self) -> Tensor:
        return self.masks.blue_missing_mask

    @property
    def red_dropout_mask(self) -> Tensor:
        return self.masks.red_dropout_mask

    @property
    def blue_dropout_mask(self) -> Tensor:
        return self.masks.blue_dropout_mask

    @property
    def cont_z(self) -> Tensor:
        return self.predictions["continuous"]

    @property
    def bin_logits(self) -> Tensor:
        return self.predictions["win"]

    @property
    def atomic_z(self) -> Tensor:
        return self.predictions["atomic"]

    @property
    def foul_z(self) -> Tensor:
        return self.predictions["foul"]

    @property
    def bonus_logits(self) -> Tensor:
        return self.predictions["bonus"]

    @property
    def special_logits(self) -> Tensor:
        return self.predictions["special"]

    @property
    def endgame_logits(self) -> Tensor:
        return self.predictions["endgame"]

    @property
    def auto_logits(self) -> Tensor:
        return self.predictions["auto"]

    @property
    def award_logits(self) -> Tensor:
        return self.predictions["award"]

    @property
    def wm_award_embedding(self) -> Tensor:
        return self.predictions["wm_award"]


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
            weights = torch.ones((h.shape[0], 1, h.shape[1]), dtype=h.dtype, device=h.device)
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


class AdditiveWinHead(nn.Module):
    """Strictly additive anti-symmetric win head."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, z_red: Tensor, z_blue: Tensor) -> Tensor:
        return self.score(z_red - z_blue)


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

    def clamp_(self, min_value: float, max_value: float) -> None:
        with torch.no_grad():
            for parameter in self.log_vars.values():
                parameter.clamp_(min=min_value, max=max_value)


class FixedTaskBalancer(nn.Module):
    """Parameter-free task weights for controlled diagnostic studies."""

    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = dict(weights)
        self.log_vars = nn.ParameterDict()

    def forward(self, task_name: str, loss: Tensor, active: bool) -> Tensor:
        if not active:
            return torch.zeros((), dtype=loss.dtype, device=loss.device)
        return float(self.weights.get(task_name, 0.0)) * loss

    def precision(self, task_name: str) -> Tensor:
        return torch.as_tensor(float(self.weights.get(task_name, 0.0)))

    def clamp_(self, min_value: float, max_value: float) -> None:
        del min_value, max_value


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
        world_model_opts: WorldModelOptions | None = None,
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
        self.world_model_opts = world_model_opts or WorldModelOptions()
        self.world_model_opts.active_spaces()

        self.Z_base = nn.Embedding(max(num_teams, 1), latent_dim)
        if opts.state_model == "base-plus-event":
            self.Z_event = nn.Embedding(
                max(num_event_teams or num_teams, 1), latent_dim, padding_idx=0
            )
        self.team_embedding = self.Z_base
        nn.init.normal_(self.Z_base.weight, mean=0.0, std=0.02)
        if hasattr(self, "Z_event"):
            nn.init.zeros_(self.Z_event.weight)

        block_args = {
            "num_heads": opts.attention_heads,
            "attention_dropout": opts.attention_dropout,
            "ffn_dropout": opts.ffn_dropout,
            "layer_norm_epsilon": opts.set_layer_norm_epsilon,
        }
        if opts.match_architecture in {"teammate-set", "full-match"}:
            self.sab = SAB(latent_dim, opts.set_ffn_dim, **block_args)
            self.pma = PMA(latent_dim, opts.set_ffn_dim, **block_args)
        if opts.match_architecture == "full-match":
            self.cross = CROSS(latent_dim, opts.set_ffn_dim, **block_args)

        self.cont_head = nn.Linear(latent_dim, max(num_cont_targets // 2, 1))
        self.atomic_head = (
            None
            if opts.core_objective
            else nn.Linear(latent_dim, max(len(opts.atomic_count_targets), 1))
        )
        self.foul_head = (
            None if opts.core_objective else nn.Linear(latent_dim, max(len(opts.foul_targets), 1))
        )
        self.bonus_head = (
            None
            if opts.core_objective
            else nn.Linear(latent_dim, max(len(opts.bonus_binary_targets), 1))
        )
        self.special_head = (
            None
            if opts.core_objective
            else nn.Linear(latent_dim, max(len(opts.special_binary_targets), 1))
        )
        self.bin_head = (
            AdditiveWinHead(latent_dim)
            if opts.match_architecture == "additive"
            else SiameseWinHead(latent_dim)
        )
        self.endgame_head = (
            None
            if opts.core_objective
            else OrdinalEndgameHead(latent_dim, self.num_endgame_classes)
        )
        if opts.state_model == "static-z-base" and not opts.core_objective:
            self.auto_head = OrdinalEndgameHead(latent_dim, len(opts.auto_class_order))
        self.award_head = (
            None if opts.core_objective else JudgesRoomHead(latent_dim, self.num_awards)
        )
        self.team_value_head = None if opts.core_objective else TeamValueHead(latent_dim)
        self.alliance_value_head = None if opts.core_objective else AllianceValueHead(latent_dim)
        if opts.state_model == "base-plus-event":
            self.delta_integration_gate = DeltaIntegrationGate(latent_dim)
        self.award_prototype_predictor = (
            None
            if opts.core_objective
            else nn.Linear(latent_dim, self.world_model_opts.award_embedding.width)
        )
        self.rank_outcome_predictor = (
            None
            if opts.core_objective
            else nn.Linear(latent_dim, self.world_model_opts.rank_embedding.width)
        )
        self.selection_embedding_predictor = (
            None
            if opts.core_objective
            else nn.Linear(2 * latent_dim, self.world_model_opts.pick_embedding.width)
        )
        task_names = [
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
            "wm_award",
            "wm_rank",
            "wm_pick",
        ]
        if opts.state_model == "static-z-base":
            task_names.insert(3, "auto")
        self.loss_balancer = (
            FixedTaskBalancer({"continuous": opts.score_loss_weight, "win": opts.win_loss_weight})
            if opts.core_objective
            else HomoscedasticTaskBalancer(tuple(task_names))
        )

    def balance_loss(self, task_name: str, loss: Tensor, active: bool) -> Tensor:
        return self.loss_balancer(task_name, loss, active)

    def team_latent(self, team_base_idx: Tensor, team_event_idx: Tensor | None = None) -> Tensor:
        base_idx = team_base_idx.long().clamp_min(0)
        z_base = self.Z_base(base_idx)
        if self.opts.state_model == "static-z-base":
            return z_base
        event_idx = torch.zeros_like(base_idx) if team_event_idx is None else team_event_idx.long()
        return z_base + self.Z_event(event_idx.clamp_min(0))

    def team_value(self, team_base_idx: Tensor, team_event_idx: Tensor | None = None) -> Tensor:
        return self.team_value_head(self.team_latent(team_base_idx, team_event_idx))

    def predict_rank_embedding(
        self, team_base_idx: Tensor, team_event_idx: Tensor | None = None
    ) -> Tensor:
        return self.rank_outcome_predictor(self.team_latent(team_base_idx, team_event_idx))

    def predict_selection_embedding(
        self,
        captain_base_idx: Tensor,
        candidate_base_idx: Tensor,
        captain_event_idx: Tensor | None = None,
        candidate_event_idx: Tensor | None = None,
    ) -> Tensor:
        captain = self.team_latent(captain_base_idx, captain_event_idx)
        candidate = self.team_latent(candidate_base_idx, candidate_event_idx)
        return self.selection_embedding_predictor(torch.cat([captain, candidate], dim=-1))

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
        if self.opts.match_architecture == "additive":
            z_alliance = h.sum(dim=1)
        else:
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
        x = self.Z_base(routed_base_idx.clamp_min(0))
        if self.opts.state_model == "base-plus-event":
            x = x + self.Z_event(routed_event_idx.clamp_min(0))
        return x, effective_missing, dropout_mask

    def pma_pool(
        self, h: Tensor, missing_mask: Tensor | None = None, pma_mode: str = "learned"
    ) -> tuple[Tensor, Tensor]:
        return self.pma(h, missing_mask, pma_mode)

    @staticmethod
    def _additive_pool(h: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.full(
            (h.shape[0], 1, h.shape[1]),
            1.0 / max(h.shape[1], 1),
            dtype=h.dtype,
            device=h.device,
        )
        return h.sum(dim=1), weights

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

        if self.opts.match_architecture == "additive":
            red_interacted = red_set
            blue_interacted = blue_set
            z_red, red_weights = self._additive_pool(red_interacted)
            z_blue, blue_weights = self._additive_pool(blue_interacted)
        else:
            red_context, _ = self.sab(red_set, red_attention_mask)
            blue_context, _ = self.sab(blue_set, blue_attention_mask)
            if self.opts.match_architecture == "full-match":
                red_interacted, _ = self.cross(red_context, blue_context, blue_attention_mask)
                blue_interacted, _ = self.cross(blue_context, red_context, red_attention_mask)
            else:
                red_interacted, blue_interacted = red_context, blue_context
            z_red, red_weights = self.pma_pool(red_interacted, red_attention_mask, pma_mode)
            z_blue, blue_weights = self.pma_pool(blue_interacted, blue_attention_mask, pma_mode)
        diff = z_red - z_blue
        z_match = torch.cat([z_red, z_blue, diff, diff.abs(), z_red * z_blue], dim=1)
        cont = torch.cat([self.cont_head(z_red), self.cont_head(z_blue)], dim=1)
        if cont.shape[1] > self.num_cont_targets:
            cont = cont[:, : self.num_cont_targets]
        empty_match = cont.new_empty((cont.shape[0], 0))
        atomic = (
            torch.cat([self.atomic_head(z_red), self.atomic_head(z_blue)], dim=1)
            if self.atomic_head is not None
            else empty_match
        )
        foul = (
            torch.cat([self.foul_head(z_red), self.foul_head(z_blue)], dim=1)
            if self.foul_head is not None
            else empty_match
        )
        bonus = (
            torch.cat([self.bonus_head(z_red), self.bonus_head(z_blue)], dim=1)
            if self.bonus_head is not None
            else empty_match
        )
        special = (
            torch.cat([self.special_head(z_red), self.special_head(z_blue)], dim=1)
            if self.special_head is not None
            else empty_match
        )

        slot_context = torch.cat([red_interacted, blue_interacted], dim=1)
        return ForwardOutput(
            representations=MatchRepresentations(
                z_red=z_red,
                z_blue=z_blue,
                z_match=z_match,
                red_pma_weights=red_weights,
                blue_pma_weights=blue_weights,
            ),
            predictions=TaskPredictions(
                {
                    "continuous": cont,
                    "win": self.bin_head(z_red, z_blue),
                    "atomic": atomic[:, : self.num_atomic_targets],
                    "foul": foul[:, : self.num_foul_targets],
                    "bonus": bonus[:, : self.num_bonus_targets],
                    "special": special[:, : self.num_special_targets],
                    "endgame": (
                        self.endgame_head(slot_context)
                        if self.endgame_head is not None
                        else slot_context.new_empty((slot_context.shape[0], 6, 0))
                    ),
                    "auto": (
                        self.auto_head(slot_context)
                        if hasattr(self, "auto_head")
                        else slot_context.new_empty((slot_context.shape[0], 6, 0))
                    ),
                    "award": (
                        self.award_head(slot_context)
                        if self.award_head is not None
                        else slot_context.new_empty((slot_context.shape[0], 6, 0))
                    ),
                    "wm_award": (
                        self.award_prototype_predictor(torch.cat([red_set, blue_set], dim=1))
                        if self.award_prototype_predictor is not None
                        else slot_context.new_empty((slot_context.shape[0], 6, 0))
                    ),
                }
            ),
            masks=SlotMasks(
                red_missing_mask=red_effective_missing,
                blue_missing_mask=blue_effective_missing,
                red_dropout_mask=red_dropout,
                blue_dropout_mask=blue_dropout,
            ),
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
    world_model_opts: WorldModelOptions | None = None,
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
        world_model_opts=world_model_opts,
    )


def optimizer_parameter_groups(model: SetTransformerModel, opts: LatentStratOptions) -> list[dict]:
    """Build AdamW groups while keeping embeddings on active-row custom L2."""

    def parameter_ids(*modules: nn.Module | None) -> set[int]:
        return {
            id(parameter)
            for module in modules
            if module is not None
            for parameter in module.parameters()
        }

    set_ids = parameter_ids(
        getattr(model, "sab", None),
        getattr(model, "cross", None),
        getattr(model, "pma", None),
    )
    head_ids = parameter_ids(
        model.cont_head,
        model.atomic_head,
        model.foul_head,
        model.bonus_head,
        model.special_head,
        model.bin_head,
        model.endgame_head,
        getattr(model, "auto_head", None),
        model.award_head,
        model.team_value_head,
        model.alliance_value_head,
        getattr(model, "delta_integration_gate", None),
        model.award_prototype_predictor,
        model.rank_outcome_predictor,
        model.selection_embedding_predictor,
    )
    no_decay_ids = parameter_ids(model.Z_base, getattr(model, "Z_event", None), model.loss_balancer)
    if hasattr(model, "pma"):
        no_decay_ids.add(id(model.pma.seed))
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            no_decay_ids.update(id(parameter) for parameter in module.parameters())
        bias = getattr(module, "bias", None)
        if isinstance(bias, nn.Parameter):
            no_decay_ids.add(id(bias))

    set_decay = []
    head_decay = []
    no_decay = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise RuntimeError("Trainable parameter was routed to more than one optimizer group.")
        seen.add(parameter_id)
        if parameter_id in no_decay_ids:
            no_decay.append(parameter)
        elif parameter_id in set_ids:
            set_decay.append(parameter)
        elif parameter_id in head_ids:
            head_decay.append(parameter)
        else:
            raise RuntimeError("Trainable parameter is not owned by an optimizer group.")

    groups = []
    if set_decay:
        groups.append({"params": set_decay, "weight_decay": opts.l2_set})
    if head_decay:
        groups.append({"params": head_decay, "weight_decay": opts.l2_heads})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups
