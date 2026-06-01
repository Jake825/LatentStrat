"""Canonical local storage paths for LatentStrat-generated files."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("data")
CACHE_ROOT = DATA_ROOT / "cache"
SCOUTING_ROOT = DATA_ROOT / "scouting"
EMBEDDING_ROOT = DATA_ROOT / "embeddings"
FEATURE_ROOT = DATA_ROOT / "features"
PRIOR_FEATURE_ROOT = FEATURE_ROOT / "prior"
SEASON_FEATURE_ROOT = FEATURE_ROOT / "season"
EVENT_FEATURE_ROOT = FEATURE_ROOT / "event"
SIDECAR_ROOT = DATA_ROOT / "sidecars"
WORLD_MODEL_DATA_ROOT = DATA_ROOT / "world_model"
WORLD_MODEL_FEATURE_ROOT = FEATURE_ROOT / "world_model"

ARTIFACT_ROOT = Path("artifacts")
PRIOR_ARTIFACT_ROOT = ARTIFACT_ROOT / "prior"
PRIOR_GRID_ARTIFACT_ROOT = ARTIFACT_ROOT / "prior-grid"
SEASON_ARTIFACT_ROOT = ARTIFACT_ROOT / "season"
WALK_FORWARD_ARTIFACT_ROOT = ARTIFACT_ROOT / "walk-forward"
SMOKE_ARTIFACT_ROOT = ARTIFACT_ROOT / "smoke"
EVIDENCE_ARTIFACT_ROOT = ARTIFACT_ROOT / "evidence"
INSPECTION_ARTIFACT_ROOT = ARTIFACT_ROOT / "inspection"
WORLD_MODEL_ARTIFACT_ROOT = ARTIFACT_ROOT / "world_model"
MATCH_BREAKDOWN_ARTIFACT_ROOT = WORLD_MODEL_ARTIFACT_ROOT / "match_breakdown"

RUNS_ROOT = Path("runs")

TBA_CACHE_BASE = CACHE_ROOT / "tba"
STATBOTICS_CACHE_PATH = CACHE_ROOT / "statbotics.sqlite"
OPENAI_EMBEDDING_CACHE_PATH = CACHE_ROOT / "openai_embeddings.sqlite"
SCOUTING_DB_PATH = SCOUTING_ROOT / "scouting.db"
EMBEDDING_DB_PATH = EMBEDDING_ROOT / "latentstrat_embeddings.sqlite"
MATCH_BREAKDOWN_CORPUS_PATH = WORLD_MODEL_DATA_ROOT / "match_breakdowns.sqlite"


def prior_features_path(season: int) -> Path:
    return PRIOR_FEATURE_ROOT / f"prior_features_{season}.parquet"


def season_features_path(season: int) -> Path:
    return SEASON_FEATURE_ROOT / f"features_{season}.parquet"


def event_features_path(event_key: str) -> Path:
    return EVENT_FEATURE_ROOT / f"features_{event_key}.parquet"


def sidecar_dir(version: str, season: int) -> Path:
    return SIDECAR_ROOT / f"{version}_{season}"


def match_breakdown_features_path(start_season: int, end_season: int) -> Path:
    return (
        WORLD_MODEL_FEATURE_ROOT
        / f"match_breakdown_alliances_{start_season}_{end_season}.parquet"
    )


def match_breakdown_artifact_dir(start_season: int, end_season: int) -> Path:
    return MATCH_BREAKDOWN_ARTIFACT_ROOT / f"v1_{start_season}_{end_season}"
