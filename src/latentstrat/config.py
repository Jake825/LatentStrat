"""LatentStrat configuration defaults."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TargetMapping(BaseModel):
    target_name: str
    tba_field: str


class LatentStratOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    season: int = 2026
    smoke_event_key: str = "2026ilch"
    latent_dim: int = 4
    attention_heads: int = 1
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0
    learning_rate: float = 1e-3
    l2: float = 1e-4
    set_ffn_dim: int = 16
    set_layer_norm_epsilon: float = 1e-5
    l2_embedding: float = 1e-4
    l2_heads: float = 1e-5
    l2_set: float = 1e-5
    epochs: int = 200
    mini_batch_size: int = 256
    smoke_mini_batch_size: int = 64
    dataloader_num_workers: int = 0
    device: str = "auto"
    use_early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    restore_best_validation_model: bool = True
    validation_fraction: float = 0.20
    split_policy: str = "chronological-holdout"
    full_season_split_policy: str = "chronological-holdout"
    fallback_full_season_split_policy: str = "chronological-holdout"
    random_seed: int = 2026
    evidence_seeds: tuple[int, ...] = (2026, 2027, 2028)
    bin_positive_weights: str | list[float] = "auto"
    target_map: tuple[TargetMapping, ...] = (
        TargetMapping(target_name="red_total_score", tba_field="redTotalScore"),
        TargetMapping(target_name="blue_total_score", tba_field="blueTotalScore"),
        TargetMapping(target_name="win_margin", tba_field="winMargin"),
        TargetMapping(target_name="fouls_drawn", tba_field="foulsDrawn"),
    )
    alliance_target_map: tuple[TargetMapping, ...] = (
        TargetMapping(target_name="total_score", tba_field="totalPoints"),
        TargetMapping(target_name="auto_pts", tba_field="totalAutoPoints"),
        TargetMapping(target_name="teleop_pts", tba_field="totalTeleopPoints"),
        TargetMapping(target_name="total_tower_pts", tba_field="totalTowerPoints"),
        TargetMapping(target_name="auto_tower_pts", tba_field="autoTowerPoints"),
        TargetMapping(target_name="endgame_tower_pts", tba_field="endGameTowerPoints"),
    )
    binary_targets: tuple[str, ...] = ("red_win",)
    diagnostic_targets: tuple[str, ...] = ("foul_pts", "major_foul_count", "minor_foul_count")


def default_options() -> LatentStratOptions:
    return LatentStratOptions()
