"""Audited differentiable 2026 scoring utilities."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

HUB_PERIODS = ("auto", "transition", "shift1", "shift2", "shift3", "shift4", "endgame")
ALLIANCE_COLORS = ("red", "blue")
AUTO_LEVEL_POINTS = (0.0, 15.0)
ENDGAME_LEVEL_POINTS = (0.0, 10.0, 20.0, 30.0)


def physics_target_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for color in ALLIANCE_COLORS:
        columns.extend(f"{color}_hub_{period}_count" for period in HUB_PERIODS)
        columns.extend(f"{color}_team_{slot}_auto_status" for slot in (1, 2, 3))
        columns.extend(f"{color}_team_{slot}_endgame_status" for slot in (1, 2, 3))
        columns.extend(
            (
                f"{color}_committed_minor_foul_count",
                f"{color}_committed_major_foul_count",
                f"{color}_bonus_energized",
                f"{color}_bonus_supercharged",
                f"{color}_official_score",
                f"{color}_adjust_points",
            )
        )
    return tuple(columns)


def ordered_atomic_columns(*, standardized: bool = False) -> list[str]:
    suffix = "_z" if standardized else ""
    return [
        f"{color}_atomic_{period}_count{suffix}"
        for color in ALLIANCE_COLORS
        for period in HUB_PERIODS
    ]


def ordered_foul_columns(*, standardized: bool = False) -> list[str]:
    suffix = "_z" if standardized else ""
    return [
        f"{color}_committed_{severity}_foul_count{suffix}"
        for color in ALLIANCE_COLORS
        for severity in ("minor", "major")
    ]


def ordered_bonus_columns() -> list[str]:
    return [
        f"{color}_bonus_{bonus}"
        for color in ALLIANCE_COLORS
        for bonus in ("energized", "supercharged")
    ]


def _alliance_targets(breakdown: dict[str, Any], color: str) -> dict[str, Any]:
    hub = breakdown.get("hubScore") or {}
    row: dict[str, Any] = {}
    for period in HUB_PERIODS:
        source = "autoCount" if period == "auto" else f"{period}Count"
        row[f"{color}_hub_{period}_count"] = hub.get(source)
    for slot in (1, 2, 3):
        row[f"{color}_team_{slot}_auto_status"] = breakdown.get(f"autoTowerRobot{slot}")
        row[f"{color}_team_{slot}_endgame_status"] = breakdown.get(f"endGameTowerRobot{slot}")
    row[f"{color}_committed_minor_foul_count"] = breakdown.get("minorFoulCount")
    row[f"{color}_committed_major_foul_count"] = breakdown.get("majorFoulCount")
    row[f"{color}_bonus_energized"] = breakdown.get("energizedAchieved")
    row[f"{color}_bonus_supercharged"] = breakdown.get("superchargedAchieved")
    row[f"{color}_official_score"] = breakdown.get("totalPoints")
    row[f"{color}_adjust_points"] = breakdown.get("adjustPoints")
    return row


def build_physics_target_table(
    corpus_path: str | Path,
    *,
    season: int = 2026,
) -> pd.DataFrame:
    """Extract one audited physics-target row per played cached match."""

    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Match-breakdown corpus does not exist: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        payloads = connection.execute(
            "SELECT match_key, event_key, raw_json FROM matches "
            "WHERE season = ? AND played = 1 AND has_breakdown = 1 ORDER BY match_key",
            (int(season),),
        ).fetchall()
    finally:
        connection.close()
    rows: list[dict[str, Any]] = []
    for match_key, event_key, raw_json in payloads:
        payload = json.loads(raw_json)
        score_breakdown = payload.get("score_breakdown") or {}
        if not all(isinstance(score_breakdown.get(color), dict) for color in ALLIANCE_COLORS):
            raise ValueError(f"Cached match {match_key} lacks both alliance breakdowns.")
        alliances = payload.get("alliances") or {}
        row: dict[str, Any] = {
            "season": int(season),
            "match_key": str(match_key),
            "event_key": str(event_key),
            "has_dq": any(
                bool((alliances.get(color) or {}).get("dq_team_keys")) for color in ALLIANCE_COLORS
            ),
        }
        for color in ALLIANCE_COLORS:
            row.update(_alliance_targets(score_breakdown[color], color))
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No played {season} match breakdowns exist in {path}.")
    if result["match_key"].duplicated().any():
        raise ValueError("Physics target table contains duplicate match keys.")
    missing = set(physics_target_columns()) - set(result)
    if missing:
        raise ValueError("Physics target table is missing: " + ", ".join(sorted(missing)))
    validate_hard_composer(result)
    return result.sort_values("match_key").reset_index(drop=True)


def _status_points(value: object, *, auto: bool) -> float:
    mapping = {"None": 0.0, "Level1": 15.0 if auto else 10.0}
    if not auto:
        mapping.update({"Level2": 20.0, "Level3": 30.0})
    if value not in mapping:
        raise ValueError(f"Unsupported 2026 tower status: {value!r}")
    return mapping[str(value)]


def hard_composed_scores(table: pd.DataFrame, *, include_adjustment: bool) -> np.ndarray:
    """Compose exact official red/blue scores from observed independent fields."""

    scores = np.zeros((len(table), 2), dtype=np.float64)
    for color_index, color in enumerate(ALLIANCE_COLORS):
        opponent = "blue" if color == "red" else "red"
        hub = sum(
            pd.to_numeric(table[f"{color}_hub_{period}_count"], errors="coerce").to_numpy(float)
            for period in HUB_PERIODS
        )
        tower = np.zeros(len(table), dtype=np.float64)
        for slot in (1, 2, 3):
            tower += np.asarray(
                [
                    _status_points(value, auto=True)
                    for value in table[f"{color}_team_{slot}_auto_status"]
                ],
                dtype=float,
            )
            tower += np.asarray(
                [
                    _status_points(value, auto=False)
                    for value in table[f"{color}_team_{slot}_endgame_status"]
                ],
                dtype=float,
            )
        fouls = 5.0 * pd.to_numeric(
            table[f"{opponent}_committed_minor_foul_count"], errors="coerce"
        ).to_numpy(float) + 15.0 * pd.to_numeric(
            table[f"{opponent}_committed_major_foul_count"], errors="coerce"
        ).to_numpy(float)
        adjustment = (
            pd.to_numeric(table[f"{color}_adjust_points"], errors="coerce").to_numpy(float)
            if include_adjustment
            else 0.0
        )
        scores[:, color_index] = hub + tower + fouls + adjustment
    return scores


def validate_hard_composer(table: pd.DataFrame) -> dict[str, int | float]:
    """Fail unless the independent cached quantities reproduce official scores exactly."""

    composed = hard_composed_scores(table, include_adjustment=True)
    actual = table[["red_official_score", "blue_official_score"]].to_numpy(float)
    finite = np.isfinite(composed) & np.isfinite(actual)
    if not finite.all():
        raise ValueError("Physics composer audit encountered missing or non-finite fields.")
    error = composed - actual
    if not np.array_equal(error, np.zeros_like(error)):
        index = np.argwhere(error != 0)[0]
        raise ValueError(
            "2026 hard composer does not reproduce official score for "
            f"{table.iloc[int(index[0])]['match_key']} {ALLIANCE_COLORS[int(index[1])]}."
        )
    nonzero_adjustment = int(
        sum(
            pd.to_numeric(table[f"{color}_adjust_points"], errors="coerce").ne(0).sum()
            for color in ALLIANCE_COLORS
        )
    )
    adjustment_without_dq = int(
        sum(
            (
                pd.to_numeric(table[f"{color}_adjust_points"], errors="coerce").ne(0)
                & ~table["has_dq"].astype(bool)
            ).sum()
            for color in ALLIANCE_COLORS
        )
    )
    if adjustment_without_dq:
        raise ValueError("A non-DQ 2026 row contains nonzero adjustment points.")
    return {
        "match_rows": int(len(table)),
        "alliance_rows": int(2 * len(table)),
        "max_absolute_error": float(np.abs(error).max(initial=0.0)),
        "nonzero_adjustment_alliances": nonzero_adjustment,
        "adjustment_without_dq": adjustment_without_dq,
    }


def compose_predicted_scores(
    atomic_raw: Tensor,
    foul_raw: Tensor,
    auto_logits: Tensor,
    endgame_logits: Tensor,
) -> Tensor:
    """Compose differentiable expected scores from independent prediction heads."""

    if atomic_raw.ndim != 2 or atomic_raw.shape[1] != 14:
        raise ValueError("Atomic predictions must have shape [batch, 14].")
    if foul_raw.ndim != 2 or foul_raw.shape[1] != 4:
        raise ValueError("Foul predictions must have shape [batch, 4].")
    if auto_logits.shape[1:] != (6, 2) or endgame_logits.shape[1:] != (6, 4):
        raise ValueError("Tower logits must have shapes [batch,6,2] and [batch,6,4].")
    atomic = torch.nn.functional.softplus(atomic_raw)
    fouls = torch.nn.functional.softplus(foul_raw)
    auto_probability = torch.softmax(auto_logits, dim=-1)
    endgame_probability = torch.softmax(endgame_logits, dim=-1)
    auto_points = torch.as_tensor(AUTO_LEVEL_POINTS, dtype=atomic.dtype, device=atomic.device)
    endgame_points = torch.as_tensor(ENDGAME_LEVEL_POINTS, dtype=atomic.dtype, device=atomic.device)
    tower = (auto_probability * auto_points).sum(dim=-1) + (
        endgame_probability * endgame_points
    ).sum(dim=-1)
    red_hub = atomic[:, :7].sum(dim=1)
    blue_hub = atomic[:, 7:].sum(dim=1)
    red_tower = tower[:, :3].sum(dim=1)
    blue_tower = tower[:, 3:].sum(dim=1)
    red_received_fouls = 5.0 * fouls[:, 2] + 15.0 * fouls[:, 3]
    blue_received_fouls = 5.0 * fouls[:, 0] + 15.0 * fouls[:, 1]
    return torch.stack(
        [red_hub + red_tower + red_received_fouls, blue_hub + blue_tower + blue_received_fouls],
        dim=1,
    )


def fit_positive_score_logit_scale(table: pd.DataFrame) -> dict[str, float | int | str]:
    """Fit a finite positive zero-intercept score-difference winner scale.

    Official winner labels are perfectly separated by the sign of official score
    differential. A raw logistic maximum-likelihood slope therefore diverges. Five-percent
    label smoothing is used only for this consistency-scale fit; the supervised winner BCE
    remains unsmoothed.
    """

    red = pd.to_numeric(table["red_total_score"], errors="coerce").to_numpy(float)
    blue = pd.to_numeric(table["blue_total_score"], errors="coerce").to_numpy(float)
    outcome = pd.to_numeric(table["red_win"], errors="coerce").to_numpy(float)
    valid = np.isfinite(red) & np.isfinite(blue) & np.isfinite(outcome) & (red != blue)
    difference = (red[valid] - blue[valid]).reshape(-1, 1)
    labels = outcome[valid].astype(int)
    if len(labels) < 2 or len(np.unique(labels)) != 2:
        raise ValueError("Score-logit scale requires both winner classes.")
    label_smoothing = 0.05
    smoothed = labels * (1 - 2 * label_smoothing) + label_smoothing
    values = difference[:, 0]
    alpha = 0.01
    for _ in range(100):
        logits = np.clip(alpha * values, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = float(np.mean((probabilities - smoothed) * values))
        hessian = float(np.mean(probabilities * (1 - probabilities) * values**2))
        if not math.isfinite(hessian) or hessian <= 1e-12:
            raise ValueError("Score-logit scale fit has invalid curvature.")
        updated = max(alpha - gradient / hessian, 1e-8)
        if abs(updated - alpha) <= 1e-12:
            alpha = updated
            break
        alpha = updated
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"Resolved score-logit scale is not positive: {alpha}")
    logits = alpha * difference[:, 0]
    sigma = float(np.std(logits))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Resolved score-logit standard deviation is invalid.")
    return {
        "alpha": alpha,
        "sigma_logit": sigma,
        "row_count": int(len(labels)),
        "fit_method": "zero-intercept-logistic-newton",
        "label_smoothing": label_smoothing,
    }
