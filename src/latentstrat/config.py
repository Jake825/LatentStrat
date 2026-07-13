"""LatentStrat configuration defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from latentstrat.paths import OPENAI_EMBEDDING_CACHE_PATH

SEASON_CONFIG_ROOT = Path("configs/season")


class TargetMapping(BaseModel):
    target_name: str
    tba_field: str


class LatentStratOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    season: int = 2026
    smoke_event_key: str = "2026ilch"
    latent_dim: int = 16
    state_model: Literal["base-plus-event", "static-z-base"] = "base-plus-event"
    match_architecture: Literal["additive", "teammate-set", "full-match"] = "full-match"
    core_objective: bool = False
    study_arm: Literal[
        "none", "primary-only", "dense-breakdown", "physics-consistent", "full-structured"
    ] = "none"
    score_target_mode: Literal["legacy", "phase-core", "official-total-core"] = "legacy"
    score_loss_weight: float = 1.0
    win_loss_weight: float = 1.0
    embedding_loss_weight: float = 1.0
    breakdown_loss_weight: float = 0.5
    winner_primary_loss_weight: float = 0.25
    score_consistency_loss_weight: float = 0.10
    score_ordering_loss_weight: float = 0.10
    winner_consistency_loss_weight: float = 0.05
    competition_probe_loss_weight: float = 0.10
    award_probe_loss_weight: float = 0.10
    physics_score_logit_alpha: float = 1.0
    physics_score_logit_sigma: float = 1.0
    log_optimizer_steps: bool = True
    attention_heads: int = 1
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0
    learning_rate: float = 1e-3
    l2: float = 1e-4
    set_ffn_dim: int = 16
    set_layer_norm_epsilon: float = 1e-5
    l2_embedding: float = 1e-4
    event_delta_l2: float = 1e-1
    l2_heads: float = 1e-4
    l2_set: float = 1e-4
    use_lr_scheduler: bool = True
    scheduler: str = "cosine"
    lr_eta_min: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    one_cycle_pct_start: float = 0.3
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: float = 10_000.0
    deterministic_algorithms: bool = False
    checkpoint_every_epochs: int = 1
    checkpoint_milestone_epochs: tuple[int, ...] = ()
    optimizer_log_interval: int = 10
    loss_log_var_min: float = -5.0
    loss_log_var_max: float = 5.0
    team_dropout_rate: float = 0.03
    epochs: int = 200
    mini_batch_size: int = 256
    smoke_mini_batch_size: int = 64
    eval_batch_size: int = 1024
    dataloader_num_workers: int = 0
    device: str = "cpu"
    use_amp: bool | str = False
    compile_model: bool | str = False
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
    continuous_targets: tuple[str, ...] = (
        "red_auto_pts",
        "red_teleop_pts",
        "blue_auto_pts",
        "blue_teleop_pts",
    )
    endgame_class_order: tuple[str, ...] = ("None", "Level1", "Level2", "Level3")
    auto_class_order: tuple[str, ...] = ("None", "Level1", "Level2", "Level3")
    award_targets: tuple[str, ...] = (
        "impact",
        "ei",
        "auto",
        "quality",
        "design",
        "control",
        "excellence",
    )
    target_map: tuple[TargetMapping, ...] = (
        TargetMapping(target_name="red_auto_pts", tba_field="redAutoPoints"),
        TargetMapping(target_name="red_teleop_pts", tba_field="redTeleopPoints"),
        TargetMapping(target_name="blue_auto_pts", tba_field="blueAutoPoints"),
        TargetMapping(target_name="blue_teleop_pts", tba_field="blueTeleopPoints"),
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
    atomic_count_targets: tuple[str, ...] = (
        "atomic_auto_count",
        "atomic_transition_count",
        "atomic_shift1_count",
        "atomic_shift2_count",
        "atomic_shift3_count",
        "atomic_shift4_count",
        "atomic_endgame_count",
    )
    foul_targets: tuple[str, ...] = (
        "committed_foul_pts",
        "committed_minor_foul_count",
        "committed_major_foul_count",
    )
    bonus_binary_targets: tuple[str, ...] = (
        "bonus_energized",
        "bonus_supercharged",
        "bonus_traversal",
    )
    special_binary_targets: tuple[str, ...] = ("special_g206_penalty",)
    rank_margin: float = 0.1
    playoff_margin: float = 0.1
    selection_triplet_margin: float = 0.5


class PriorOpts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_dim: int = 256
    latent_dim: int = 16
    embedding_model: str = "text-embedding-3-small"
    epochs: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 256
    embedding_batch_size: int = 128
    max_embedding_retries: int = 6
    embedding_retry_initial_delay: float = 1.0
    embedding_retry_multiplier: float = 2.0
    embedding_retry_max_delay: float = 60.0
    embedding_retry_jitter: float = 0.25
    max_team_number: int = 12_500
    future_start_number: int = 11_000
    future_baseline_team: int = 10_900
    future_growth_per_year: int = 800
    epa_source_year: int | None = None
    cache_path: str = str(OPENAI_EMBEDDING_CACHE_PATH)
    device: str = "cpu"
    random_seed: int = 2026
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    lr_eta_min: float = 1e-5
    one_cycle_pct_start: float = 0.3
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: float = 10_000.0
    deterministic_algorithms: bool = False
    checkpoint_every_epochs: int = 1
    optimizer_log_interval: int = 10


def default_options(season: int = 2026) -> LatentStratOptions:
    """Load tracked season defaults while preserving model-level legacy fallbacks."""

    config_path = SEASON_CONFIG_ROOT / f"{season}.yaml"
    payload = (
        yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    )
    values = dict(payload or {})
    values["season"] = season
    return LatentStratOptions.model_validate(values)
