"""Canonical local storage paths for LatentStrat-generated files."""

from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("data")
CACHE_ROOT = DATA_ROOT / "cache"
SCOUTING_ROOT = DATA_ROOT / "scouting"
EMBEDDING_ROOT = DATA_ROOT / "embeddings"
FEATURE_ROOT = DATA_ROOT / "features"
PRETRAINING_DATA_ROOT = DATA_ROOT / "pretraining"
MATCH_BREAKDOWN_DATA_ROOT = PRETRAINING_DATA_ROOT / "match-breakdown"
PRETRAINING_FEATURE_ROOT = FEATURE_ROOT / "pretraining"
PRIOR_FEATURE_ROOT = PRETRAINING_FEATURE_ROOT / "prior"
MATCH_BREAKDOWN_FEATURE_ROOT = PRETRAINING_FEATURE_ROOT / "match-breakdown"
SEASON_FEATURE_ROOT = FEATURE_ROOT / "season"
EVENT_FEATURE_ROOT = FEATURE_ROOT / "event"
SIDECAR_ROOT = DATA_ROOT / "sidecars"

ARTIFACT_ROOT = Path("artifacts")
PRETRAINING_ARTIFACT_ROOT = ARTIFACT_ROOT / "pretraining"
PRIOR_ARTIFACT_ROOT = PRETRAINING_ARTIFACT_ROOT / "prior"
PRIOR_GRID_ARTIFACT_ROOT = PRETRAINING_ARTIFACT_ROOT / "prior-grid"
MATCH_BREAKDOWN_ARTIFACT_ROOT = PRETRAINING_ARTIFACT_ROOT / "match-breakdown"
EXPERIMENTAL_ARTIFACT_ROOT = ARTIFACT_ROOT / "experimental"
EXPERIMENTAL_FROZEN_TARGET_ARTIFACT_ROOT = EXPERIMENTAL_ARTIFACT_ROOT / "frozen-targets"
SEASON_ARTIFACT_ROOT = ARTIFACT_ROOT / "season"
WALK_FORWARD_ARTIFACT_ROOT = ARTIFACT_ROOT / "walk-forward"
SMOKE_ARTIFACT_ROOT = ARTIFACT_ROOT / "smoke"
EVIDENCE_ARTIFACT_ROOT = ARTIFACT_ROOT / "evidence"
INSPECTION_ARTIFACT_ROOT = ARTIFACT_ROOT / "inspection"

RUNS_ROOT = Path("runs")

TBA_CACHE_BASE = CACHE_ROOT / "tba"
STATBOTICS_CACHE_PATH = CACHE_ROOT / "statbotics.sqlite"
OPENAI_EMBEDDING_CACHE_PATH = CACHE_ROOT / "openai_embeddings.sqlite"
SCOUTING_DB_PATH = SCOUTING_ROOT / "scouting.db"
EMBEDDING_DB_PATH = EMBEDDING_ROOT / "latentstrat_embeddings.sqlite"
MATCH_BREAKDOWN_CORPUS_PATH = MATCH_BREAKDOWN_DATA_ROOT / "corpus.sqlite"


def prior_features_path(season: int) -> Path:
    return PRIOR_FEATURE_ROOT / f"prior_features_{season}.parquet"


def season_features_path(season: int) -> Path:
    return SEASON_FEATURE_ROOT / f"features_{season}.parquet"


def event_features_path(event_key: str) -> Path:
    return EVENT_FEATURE_ROOT / f"features_{event_key}.parquet"


def match_breakdown_features_path(start_season: int, end_season: int) -> Path:
    return (
        MATCH_BREAKDOWN_FEATURE_ROOT
        / f"match_breakdown_alliances_{start_season}_{end_season}.parquet"
    )


def match_breakdown_artifact_dir(
    start_season: int, end_season: int, artifact_version: str = "v1"
) -> Path:
    return MATCH_BREAKDOWN_ARTIFACT_ROOT / f"{artifact_version}_{start_season}_{end_season}"
