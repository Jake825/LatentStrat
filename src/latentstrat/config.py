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
    latent_dim: int = 16
    attention_heads: int = 1
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0
    learning_rate: float = 1e-3
    l2: float = 1e-4
    set_ffn_dim: int = 16
    set_layer_norm_epsilon: float = 1e-5
    l2_embedding: float = 1e-4
    event_delta_l2: float = 1e-1
    l2_heads: float = 1e-5
    l2_set: float = 1e-5
    team_dropout_rate: float = 0.03
    epochs: int = 200
    mini_batch_size: int = 256
    smoke_mini_batch_size: int = 64
    eval_batch_size: int = 1024
    dataloader_num_workers: int = 0
    device: str = "auto"
    use_amp: bool | str = "auto"
    compile_model: bool | str = "auto"
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


class PriorOpts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_dim: int = 256
    latent_dim: int = 16
    dropout_rate: float = 0.3
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
    cache_path: str = "data/prior_cache/openai_embeddings.sqlite"
    device: str = "auto"
    random_seed: int = 2026


def default_options() -> LatentStratOptions:
    return LatentStratOptions()
