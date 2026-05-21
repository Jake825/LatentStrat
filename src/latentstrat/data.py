"""Pandas data builders and split/normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from frc.datastore import FRCDataStore
from frc.models import Match, MatchAlliance
from latentstrat.config import LatentStratOptions, TargetMapping, default_options


@dataclass
class Split:
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray
    policy: str
    seed: int | None = None
    fallback_reason: str | None = None


@dataclass
class TargetStats:
    target_names: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    is_constant: np.ndarray


def comp_level_ordinal(comp_level: str) -> int:
    return {"qm": 1, "ef": 2, "qf": 3, "sf": 4, "f": 5}.get(str(comp_level), 99)


def _get_raw(data: dict[str, Any], field: str, default: Any = None) -> Any:
    return data.get(field, default)


def _posix_to_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or pd.isna(value):
        return pd.NaT
    return pd.to_datetime(float(value), unit="s", utc=True)


def _camel_to_snake(name: str) -> str:
    out = []
    for char in name:
        if char.isupper() and out:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _breakdown_value(breakdown: dict[str, Any], tba_field: str, role: str, color: str) -> float:
    snake = _camel_to_snake(tba_field)
    if tba_field in breakdown:
        return float(breakdown[tba_field])
    if snake in breakdown:
        return float(breakdown[snake])
    raise KeyError(f"Missing {role} breakdown field {tba_field!r} for {color} alliance.")


def _event_week_map(event_metadata: pd.DataFrame | None) -> dict[str, float]:
    if event_metadata is None or event_metadata.empty:
        return {}
    names = set(event_metadata.columns)
    key_col = next((c for c in ("key", "event_key", "EventKey") if c in names), None)
    week_col = next((c for c in ("week", "event_week", "EventWeek", "Week") if c in names), None)
    if key_col is None or week_col is None:
        return {}
    valid = event_metadata[[key_col, week_col]].dropna()
    return {str(row[key_col]): float(row[week_col]) for _, row in valid.iterrows()}


def _match_event_key(match: Match) -> str:
    return match.event_key or match.match_key.split("_", 1)[0]


def _extract_season(event_key: str, match_key: str) -> int:
    source = event_key or match_key
    return int(str(source)[:4])


def _alliance(match: Match, color: str) -> MatchAlliance:
    return match.red_alliance if color == "red" else match.blue_alliance


def _raw_alliance(match: Match, color: str) -> dict[str, Any]:
    alliances = _get_raw(match.tba_data, "alliances", {}) or {}
    return alliances.get(color, {}) or {}


def _raw_score(match: Match, color: str) -> float:
    raw = _raw_alliance(match, color)
    if "score" in raw:
        return float(raw["score"])
    return float(_alliance(match, color).score)


def _score_breakdown(match: Match, color: str) -> dict[str, Any]:
    breakdown = _get_raw(match.tba_data, "score_breakdown", None)
    if not breakdown:
        return {}
    return breakdown.get(color, {}) or {}


def _time_fields(
    tba_data: dict[str, Any],
    event_order: int,
    comp_level: str,
    set_number: float,
    match_number: float,
) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, float]:
    scheduled = _get_raw(tba_data, "time", np.nan)
    actual = _get_raw(tba_data, "actual_time", np.nan)
    post = _get_raw(tba_data, "post_result_time", np.nan)
    scheduled_time = _posix_to_timestamp(scheduled)
    actual_time = _posix_to_timestamp(actual)
    post_result_time = _posix_to_timestamp(post)
    sort_time = actual
    if sort_time is None or pd.isna(sort_time):
        sort_time = post
    if sort_time is None or pd.isna(sort_time):
        sort_time = scheduled
    if sort_time is None or pd.isna(sort_time):
        season_start = pd.Timestamp("2026-01-01", tz="UTC").timestamp()
        sort_time = (
            season_start
            + event_order * 7 * 86400
            + comp_level_ordinal(comp_level) * 10000
            + float(set_number or 0) * 100
            + float(match_number or 0)
        )
    return scheduled_time, actual_time, post_result_time, float(sort_time)


def _make_alliance_row(
    match: Match,
    color: str,
    opponent_color: str,
    event_order: int,
    event_weeks: dict[str, float],
    target_map: tuple[TargetMapping, ...],
    binary_targets: tuple[str, ...],
    diagnostic_targets: tuple[str, ...],
) -> dict[str, Any] | None:
    raw_score = _raw_score(match, color)
    if raw_score == -1:
        return None
    breakdown = _score_breakdown(match, color)
    if not breakdown:
        return None

    event_key = _match_event_key(match)
    tba_data = match.tba_data
    comp_level = str(_get_raw(tba_data, "comp_level", ""))
    set_number = float(_get_raw(tba_data, "set_number", np.nan))
    match_number = float(_get_raw(tba_data, "match_number", np.nan))
    scheduled, actual, post, sort_time = _time_fields(
        tba_data, event_order, comp_level, set_number, match_number
    )
    alliance = _alliance(match, color)
    raw_alliance = _raw_alliance(match, color)
    team_keys = list(alliance.team_keys[:3]) + [""] * max(0, 3 - len(alliance.team_keys))
    row: dict[str, Any] = {
        "season": _extract_season(event_key, match.match_key),
        "event_key": event_key,
        "match_key": match.match_key,
        "alliance_color": color,
        "opponent_alliance_color": opponent_color,
        "comp_level": comp_level,
        "set_number": set_number,
        "match_number": match_number,
        "team_1_key": team_keys[0],
        "team_2_key": team_keys[1],
        "team_3_key": team_keys[2],
        "score_raw": raw_score,
        "has_breakdown": True,
        "has_surrogate": bool(raw_alliance.get("surrogate_team_keys") or alliance.surrogate_team_keys),
        "has_dq": bool(raw_alliance.get("dq_team_keys") or alliance.dq_team_keys),
        "scheduled_time": scheduled,
        "actual_time": actual,
        "post_result_time": post,
        "sort_time": sort_time,
        "sort_ordinal": np.nan,
        "event_week": event_weeks.get(event_key, np.nan),
        "comp_ordinal": comp_level_ordinal(comp_level),
        "alliance_order": 2 if color == "blue" else 1,
    }
    for mapping in target_map:
        row[mapping.target_name] = _breakdown_value(breakdown, mapping.tba_field, "target", color)
    binary_map = {
        "energized": "energizedAchieved",
        "supercharged": "superchargedAchieved",
        "traversal": "traversalAchieved",
    }
    for target in binary_targets:
        row[target] = bool(_breakdown_value(breakdown, binary_map[target], "binary", color))
    diagnostic_map = {
        "foul_pts": "foulPoints",
        "major_foul_count": "majorFoulCount",
        "minor_foul_count": "minorFoulCount",
    }
    for target in diagnostic_targets:
        row[target] = _breakdown_value(breakdown, diagnostic_map[target], "diagnostic", color)
    return row


def build_season_alliance_table(
    store: FRCDataStore,
    opts: LatentStratOptions | None = None,
    *,
    event_metadata: pd.DataFrame | None = None,
    target_map: tuple[TargetMapping, ...] | None = None,
    binary_targets: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    opts = opts or default_options()
    target_map = target_map if target_map is not None else opts.alliance_target_map
    binary_targets = binary_targets if binary_targets is not None else ()
    matches = list(store.matches.values())
    if not matches:
        return pd.DataFrame()
    event_keys = sorted({_match_event_key(match) for match in matches})
    event_order = {key: idx + 1 for idx, key in enumerate(event_keys)}
    event_weeks = _event_week_map(event_metadata)
    rows = []
    for match in matches:
        event_key = _match_event_key(match)
        for color, opponent in (("red", "blue"), ("blue", "red")):
            row = _make_alliance_row(
                match,
                color,
                opponent,
                event_order[event_key],
                event_weeks,
                target_map,
                binary_targets,
                opts.diagnostic_targets,
            )
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["sort_time", "comp_ordinal", "set_number", "match_number", "alliance_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    table["sort_ordinal"] = np.arange(1, len(table) + 1)
    return table.drop(columns=["comp_ordinal", "alliance_order"])


def _make_match_row(red: pd.Series, blue: pd.Series) -> dict[str, Any]:
    row = {
        "season": int(red["season"]),
        "event_key": str(red["event_key"]),
        "match_key": str(red["match_key"]),
        "comp_level": str(red["comp_level"]),
        "set_number": red["set_number"],
        "match_number": red["match_number"],
        "red_team_1_key": str(red["team_1_key"]),
        "red_team_2_key": str(red["team_2_key"]),
        "red_team_3_key": str(red["team_3_key"]),
        "blue_team_1_key": str(blue["team_1_key"]),
        "blue_team_2_key": str(blue["team_2_key"]),
        "blue_team_3_key": str(blue["team_3_key"]),
        "red_score_raw": red["score_raw"],
        "blue_score_raw": blue["score_raw"],
        "red_has_surrogate": bool(red["has_surrogate"]),
        "blue_has_surrogate": bool(blue["has_surrogate"]),
        "red_has_dq": bool(red["has_dq"]),
        "blue_has_dq": bool(blue["has_dq"]),
        "scheduled_time": red["scheduled_time"],
        "actual_time": red["actual_time"],
        "post_result_time": red["post_result_time"],
        "sort_time": red["sort_time"],
        "sort_ordinal": np.nan,
        "event_week": red["event_week"],
        "comp_ordinal": comp_level_ordinal(str(red["comp_level"])),
        "red_total_score": red["total_score"],
        "blue_total_score": blue["total_score"],
        "red_foul_pts": red["foul_pts"],
        "blue_foul_pts": blue["foul_pts"],
    }
    row["win_margin"] = row["red_total_score"] - row["blue_total_score"]
    row["fouls_drawn"] = row["red_foul_pts"] - row["blue_foul_pts"]
    row["red_win"] = row["win_margin"] > 0
    return row


def build_season_match_table(
    store: FRCDataStore,
    opts: LatentStratOptions | None = None,
    *,
    event_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    opts = opts or default_options()
    alliance_rows = build_season_alliance_table(
        store,
        opts,
        event_metadata=event_metadata,
        target_map=opts.alliance_target_map,
        binary_targets=(),
    )
    if alliance_rows.empty:
        return pd.DataFrame()
    rows = []
    for match_key in sorted(alliance_rows["match_key"].unique()):
        match_rows = alliance_rows[alliance_rows["match_key"] == match_key]
        red_rows = match_rows[match_rows["alliance_color"] == "red"]
        blue_rows = match_rows[match_rows["alliance_color"] == "blue"]
        if len(red_rows) == 1 and len(blue_rows) == 1:
            rows.append(_make_match_row(red_rows.iloc[0], blue_rows.iloc[0]))
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["sort_time", "comp_ordinal", "set_number", "match_number"], kind="mergesort"
    ).reset_index(drop=True)
    table["sort_ordinal"] = np.arange(1, len(table) + 1)
    return table.drop(columns=["comp_ordinal"])


def _team_key_columns(table: pd.DataFrame) -> list[str]:
    red_blue = [
        "red_team_1_key",
        "red_team_2_key",
        "red_team_3_key",
        "blue_team_1_key",
        "blue_team_2_key",
        "blue_team_3_key",
    ]
    if all(column in table.columns for column in red_blue):
        return red_blue
    return ["team_1_key", "team_2_key", "team_3_key"]


def make_team_index_map(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    key_columns = _team_key_columns(table)
    missing = [column for column in key_columns if column not in table.columns]
    if missing:
        raise KeyError(f"Data table is missing team key columns: {', '.join(missing)}")
    keys = sorted(
        {
            str(value)
            for column in key_columns
            for value in table[column].tolist()
            if pd.notna(value) and str(value)
        }
    )
    index_map = {key: idx for idx, key in enumerate(keys)}
    out = table.copy()
    for column in key_columns:
        idx_column = column.replace("_key", "_idx")
        values = [index_map.get(str(value), pd.NA) if pd.notna(value) and str(value) else pd.NA for value in out[column]]
        out[idx_column] = pd.Series(values, dtype="Int64")
    return out, index_map


def bounded_holdout_count(num_items: int, fraction: float) -> int:
    count = max(1, int(round(num_items * fraction)))
    return min(count, num_items - 1)


def make_split(
    table: pd.DataFrame,
    opts: LatentStratOptions | None = None,
    *,
    policy: str | None = None,
    validation_fraction: float | None = None,
    random_seed: int | None = None,
) -> Split:
    opts = opts or default_options()
    policy = policy or opts.split_policy
    validation_fraction = opts.validation_fraction if validation_fraction is None else validation_fraction
    random_seed = opts.random_seed if random_seed is None else random_seed
    if validation_fraction < 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be >= 0 and < 1")
    n_rows = len(table)
    if n_rows < 2:
        raise ValueError("At least two rows are required to make a split.")
    validation_mask = np.zeros(n_rows, dtype=bool)
    if policy == "chronological-holdout":
        holdout = bounded_holdout_count(n_rows, validation_fraction)
        order = np.argsort(table["sort_ordinal"].to_numpy())
        validation_mask[order[-holdout:]] = True
    elif policy == "week-held-out":
        if "event_week" not in table.columns or table["event_week"].isna().any():
            raise KeyError("week-held-out split requires nonmissing event_week values.")
        weeks = np.sort(table["event_week"].dropna().unique())
        holdout = bounded_holdout_count(len(weeks), validation_fraction)
        validation_mask = table["event_week"].isin(weeks[-holdout:]).to_numpy()
    elif policy == "event-held-out":
        events = np.array(sorted(table["event_key"].astype(str).unique()))
        holdout = bounded_holdout_count(len(events), validation_fraction)
        rng = np.random.default_rng(random_seed)
        held_out = events[rng.permutation(len(events))[:holdout]]
        validation_mask = table["event_key"].astype(str).isin(held_out).to_numpy()
    else:
        raise ValueError(f"Unknown split policy {policy!r}.")
    train_mask = ~validation_mask
    return Split(
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=np.zeros(n_rows, dtype=bool),
        policy=policy,
    )


def target_matrix(table: pd.DataFrame, target_names: list[str] | tuple[str, ...]) -> np.ndarray:
    values = np.zeros((len(table), len(target_names)), dtype=float)
    for idx, name in enumerate(target_names):
        if name and name in table.columns:
            values[:, idx] = table[name].astype(float).to_numpy()
    return values


def fit_target_stats(
    table: pd.DataFrame, train_mask: np.ndarray, opts: LatentStratOptions | None = None
) -> TargetStats:
    opts = opts or default_options()
    target_names = [mapping.target_name for mapping in opts.target_map]
    if len(train_mask) != len(table):
        raise ValueError("train_mask must have one element per table row.")
    if not np.any(train_mask):
        raise ValueError("At least one training row is required.")
    values = target_matrix(table, target_names)
    train = values[train_mask]
    mu = np.nanmean(train, axis=0)
    sigma = np.nanstd(train, axis=0, ddof=1)
    is_constant = (sigma == 0) | np.isnan(sigma)
    sigma[is_constant] = 1
    return TargetStats(target_names=target_names, mu=mu, sigma=sigma, is_constant=is_constant)


def apply_target_stats(table: pd.DataFrame, stats: TargetStats) -> pd.DataFrame:
    out = table.copy()
    for idx, target in enumerate(stats.target_names):
        out[f"{target}_z"] = (out[target].astype(float) - stats.mu[idx]) / stats.sigma[idx]
    return out
