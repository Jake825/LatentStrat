"""Tracked manifest generation for ignored empirical baseline archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latentstrat.artifacts import sha256_file

REQUIRED_ARCHIVE_FILES = (
    "commit.txt",
    "prior_checkpoint.pt",
    "full_season_checkpoint.pt",
    "inputs/features_v58_2026.parquet",
    "inputs/rankings_2026.parquet",
    "inputs/selections_2026.parquet",
    "inputs/playoffs_2026.parquet",
    "walk-forward/config.json",
    "walk-forward/walk_forward_metrics.csv",
    "walk-forward/walk_forward_history.csv",
    "walk-forward/walk_forward_predictions.parquet",
)


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary_metrics(archive: Path) -> dict[str, Any]:
    metrics = pd.read_csv(archive / "walk-forward" / "walk_forward_metrics.csv")
    average = metrics.loc[metrics["fold_number"].astype(str) == "AVERAGE"]
    if len(average) != 1:
        raise ValueError("Walk-forward metrics must contain exactly one AVERAGE row.")
    return {column: _json_scalar(value) for column, value in average.iloc[0].items()}


def _archive_hashes(archive: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(archive.rglob("*")):
        if not path.is_file() or path.suffix in {".log", ".pid"}:
            continue
        hashes[path.relative_to(archive).as_posix()] = sha256_file(path)
    return hashes


def build_baseline_manifest(
    archive_dir: str | Path,
    *,
    tag: str,
    commit_sha: str,
    regeneration_commands: list[str],
) -> dict[str, Any]:
    archive = Path(archive_dir)
    missing = [
        relative for relative in REQUIRED_ARCHIVE_FILES if not (archive / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError("Baseline archive is incomplete: " + ", ".join(missing))
    fold_checkpoints = sorted((archive / "walk-forward" / "fold_checkpoints").glob("*.pt"))
    if not fold_checkpoints:
        raise FileNotFoundError("Baseline archive has no saved walk-forward fold checkpoints.")
    return {
        "schema_version": 1,
        "baseline_ref": tag,
        "commit_sha": commit_sha,
        "archive_root": archive.as_posix(),
        "file_sha256": _archive_hashes(archive),
        "summary_metrics": _summary_metrics(archive),
        "regeneration_commands": regeneration_commands,
    }


def write_baseline_manifest(
    archive_dir: str | Path,
    output_path: str | Path,
    *,
    tag: str,
    commit_sha: str,
    regeneration_commands: list[str],
) -> Path:
    payload = build_baseline_manifest(
        archive_dir,
        tag=tag,
        commit_sha=commit_sha,
        regeneration_commands=regeneration_commands,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
