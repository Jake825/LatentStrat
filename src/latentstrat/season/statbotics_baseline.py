"""Leakage-safe Statbotics pre-match prediction artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latentstrat.artifacts import sha256_file

STATBOTICS_PREDICTION_COLUMNS = (
    "season",
    "event_key",
    "match_key",
    "statbotics_pred_red_score",
    "statbotics_pred_blue_score",
    "statbotics_pred_red_win_probability",
)


def _prediction_row(match: dict[str, Any], season: int) -> dict[str, Any]:
    pred = match.get("pred")
    if not isinstance(pred, dict):
        raise ValueError(
            f"Statbotics match {match.get('key')!r} has no pre-match pred mapping; "
            f"top-level keys={sorted(match)}"
        )
    required = {"red_score", "blue_score", "red_win_prob"}
    missing = sorted(required - set(pred))
    if missing:
        raise ValueError(
            f"Statbotics match {match.get('key')!r} pred is missing {missing}; "
            f"pred keys={sorted(pred)}"
        )
    row_season = int(match.get("year", str(match.get("key", ""))[:4]))
    if row_season != int(season):
        raise ValueError(
            f"Statbotics match {match.get('key')!r} has season {row_season}, expected {season}."
        )
    return {
        "season": row_season,
        "event_key": str(match.get("event") or ""),
        "match_key": str(match.get("key") or ""),
        "statbotics_pred_red_score": float(pred["red_score"]),
        "statbotics_pred_blue_score": float(pred["blue_score"]),
        "statbotics_pred_red_win_probability": float(pred["red_win_prob"]),
    }


def validate_statbotics_prediction_table(table: pd.DataFrame, season: int) -> pd.DataFrame:
    missing = sorted(set(STATBOTICS_PREDICTION_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Statbotics prediction table is missing columns: {', '.join(missing)}")
    out = table[list(STATBOTICS_PREDICTION_COLUMNS)].copy()
    if out.empty:
        raise ValueError("Statbotics prediction table is empty.")
    if out["match_key"].isna().any() or (out["match_key"].astype(str).str.len() == 0).any():
        raise ValueError("Statbotics prediction table contains empty match keys.")
    duplicates = int(out["match_key"].duplicated(keep=False).sum())
    if duplicates:
        raise ValueError(
            f"Statbotics prediction table contains {duplicates} duplicate match-key rows."
        )
    seasons = set(pd.to_numeric(out["season"], errors="raise").astype(int))
    if seasons != {int(season)}:
        raise ValueError(f"Statbotics prediction seasons {sorted(seasons)} do not equal {season}.")
    score_columns = ["statbotics_pred_red_score", "statbotics_pred_blue_score"]
    scores = out[score_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Statbotics score predictions must all be finite.")
    probabilities = pd.to_numeric(
        out["statbotics_pred_red_win_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Statbotics win probabilities must be finite and within [0, 1].")
    return out.sort_values("match_key", kind="mergesort").reset_index(drop=True)


def collect_statbotics_predictions(
    provider: Any,
    season: int,
    *,
    page_size: int = 10_000,
) -> pd.DataFrame:
    if page_size <= 0 or page_size > 10_000:
        raise ValueError("page_size must be between 1 and 10000.")
    rows: list[dict[str, Any]] = []
    seen_page_signatures: set[tuple[str, ...]] = set()
    offset = 0
    while True:
        page = provider.get_matches(year=int(season), limit=page_size, offset=offset)
        if not isinstance(page, list):
            raise TypeError("Statbotics get_matches must return a list.")
        signature = tuple(str(match.get("key") or "") for match in page)
        if page and signature in seen_page_signatures:
            raise RuntimeError("Statbotics pagination repeated a page before completion.")
        seen_page_signatures.add(signature)
        rows.extend(_prediction_row(match, int(season)) for match in page)
        if len(page) < page_size:
            break
        offset += page_size
    return validate_statbotics_prediction_table(pd.DataFrame(rows), int(season))


def write_statbotics_prediction_artifact(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    season: int,
    cache_path: str | Path,
    retrieved_at_utc: str | None = None,
) -> tuple[Path, Path]:
    validated = validate_statbotics_prediction_table(table, season)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(output, engine="pyarrow", index=False)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "workflow": "season.statbotics-baseline",
        "prediction_kind": "pre-match",
        "season": int(season),
        "rows": len(validated),
        "response_fields": ["pred.red_score", "pred.blue_score", "pred.red_win_prob"],
        "statbotics_package_version": version("statbotics"),
        "retrieved_at_utc": retrieved_at_utc or datetime.now(UTC).isoformat(),
        "cache_path": Path(cache_path).as_posix(),
        "output": {
            "path": output.as_posix(),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output, manifest_path


def build_statbotics_prediction_artifact(
    provider: Any,
    season: int,
    output_path: str | Path,
    *,
    page_size: int = 10_000,
) -> tuple[Path, Path]:
    table = collect_statbotics_predictions(provider, season, page_size=page_size)
    return write_statbotics_prediction_artifact(
        table,
        output_path,
        season=season,
        cache_path=provider.cache_path,
    )


__all__ = [
    "STATBOTICS_PREDICTION_COLUMNS",
    "build_statbotics_prediction_artifact",
    "collect_statbotics_predictions",
    "validate_statbotics_prediction_table",
    "write_statbotics_prediction_artifact",
]
