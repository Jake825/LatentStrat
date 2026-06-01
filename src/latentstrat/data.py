"""Pandas data builders and split/normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from frc.datastore import FRCDataStore
from frc.models import Match, MatchAlliance
from latentstrat.config import LatentStratOptions, TargetMapping, default_options

V56_MAX_TEAM_NUMBER = 12_500


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


def _breakdown_optional(breakdown: dict[str, Any], tba_field: str, default: Any = np.nan) -> Any:
    snake = _camel_to_snake(tba_field)
    if tba_field in breakdown:
        return breakdown[tba_field]
    if snake in breakdown:
        return breakdown[snake]
    return default


def _as_float_or_nan(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _as_binary_or_nan(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(bool(value))


def _v57_score_targets_2026(breakdown: dict[str, Any]) -> dict[str, float]:
    """Map the 2026 score_breakdown payload into generic V5.7 target names."""

    if not breakdown:
        return {
            "atomic_auto_count": float("nan"),
            "atomic_transition_count": float("nan"),
            "atomic_shift1_count": float("nan"),
            "atomic_shift2_count": float("nan"),
            "atomic_shift3_count": float("nan"),
            "atomic_shift4_count": float("nan"),
            "atomic_endgame_count": float("nan"),
            "bonus_energized": float("nan"),
            "bonus_supercharged": float("nan"),
            "bonus_traversal": float("nan"),
            "special_g206_penalty": float("nan"),
        }
    hub_score = breakdown.get("hubScore") or {}
    return {
        "atomic_auto_count": _as_float_or_nan(hub_score.get("autoCount")),
        "atomic_transition_count": _as_float_or_nan(hub_score.get("transitionCount")),
        "atomic_shift1_count": _as_float_or_nan(hub_score.get("shift1Count")),
        "atomic_shift2_count": _as_float_or_nan(hub_score.get("shift2Count")),
        "atomic_shift3_count": _as_float_or_nan(hub_score.get("shift3Count")),
        "atomic_shift4_count": _as_float_or_nan(hub_score.get("shift4Count")),
        "atomic_endgame_count": _as_float_or_nan(hub_score.get("endgameCount")),
        "bonus_energized": _as_binary_or_nan(breakdown.get("energizedAchieved")),
        "bonus_supercharged": _as_binary_or_nan(breakdown.get("superchargedAchieved")),
        "bonus_traversal": _as_binary_or_nan(breakdown.get("traversalAchieved")),
        "special_g206_penalty": _as_binary_or_nan(breakdown.get("g206Penalty")),
    }


def v57_score_targets_for_breakdown(
    season: int, breakdown: dict[str, Any]
) -> dict[str, float]:
    """Return generic V5.7 match-spine targets for a season-specific breakdown."""

    if int(season) == 2026:
        return _v57_score_targets_2026(breakdown)
    raise ValueError(f"No V5.7 score-breakdown mapper is registered for season {season}.")


def _alliance_columns(targets: tuple[str, ...] | list[str]) -> list[str]:
    return [f"{color}_{target}" for color in ("red", "blue") for target in targets]


def v57_continuous_target_names(opts: LatentStratOptions | None = None) -> list[str]:
    opts = opts or default_options()
    return _alliance_columns(tuple(opts.atomic_count_targets) + tuple(opts.foul_targets))


def v57_binary_target_names(opts: LatentStratOptions | None = None) -> list[str]:
    opts = opts or default_options()
    return _alliance_columns(
        tuple(opts.bonus_binary_targets) + tuple(opts.special_binary_targets)
    )


def _normalize_endgame_status(value: Any) -> str:
    if value is None or pd.isna(value):
        return "None"
    text = str(value).strip()
    if not text or text.lower() == "none":
        return "None"
    normalized = text.replace(" ", "")
    aliases = {
        "Level1": "Level1",
        "Level2": "Level2",
        "Level3": "Level3",
        "L1": "Level1",
        "L2": "Level2",
        "L3": "Level3",
    }
    return aliases.get(normalized, normalized)


def _event_key_column(event_metadata: pd.DataFrame) -> str | None:
    names = set(event_metadata.columns)
    return next((c for c in ("key", "event_key", "EventKey") if c in names), None)


def _event_date_column(event_metadata: pd.DataFrame) -> str | None:
    names = set(event_metadata.columns)
    return next(
        (
            c
            for c in (
                "start_date",
                "StartDate",
                "startDate",
                "sort_time",
                "scheduled_time",
                "end_date",
            )
            if c in names
        ),
        None,
    )


def canonical_event_week_table(event_metadata: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["event_key", "raw_event_week", "event_week"]
    if event_metadata is None or event_metadata.empty:
        return pd.DataFrame(columns=columns)
    key_col = _event_key_column(event_metadata)
    if key_col is None:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({"event_key": event_metadata[key_col].astype(str)})
    names = set(event_metadata.columns)
    week_col = next(
        (c for c in ("week", "raw_event_week", "EventWeek", "Week") if c in names), None
    )
    raw_week = (
        pd.to_numeric(event_metadata[week_col], errors="coerce")
        if week_col is not None
        else pd.Series(np.nan, index=event_metadata.index)
    )
    out["raw_event_week"] = raw_week.to_numpy(dtype=float)
    out["event_week"] = raw_week + 1
    date_col = _event_date_column(event_metadata)
    if date_col is not None:
        dates = pd.to_datetime(event_metadata[date_col], errors="coerce", utc=True)
    else:
        dates = pd.Series(pd.NaT, index=event_metadata.index, dtype="datetime64[ns, UTC]")
    normalized_dates = dates.dt.normalize()
    date_frame = pd.DataFrame({"date": normalized_dates, "event_key": out["event_key"]})
    valid_dates = date_frame.dropna(subset=["date"]).sort_values(
        ["date", "event_key"], kind="mergesort"
    )
    inferred: dict[str, int] = {}
    if not valid_dates.empty:
        origin = valid_dates["date"].min()
        inferred = {
            str(row.event_key): int((row.date - origin).days // 7) + 1
            for row in valid_dates.itertuples()
        }
    missing_week = ~np.isfinite(out["event_week"].to_numpy(dtype=float))
    if missing_week.any():
        fallback_values = [
            inferred.get(str(event_key), idx + 1)
            for idx, event_key in enumerate(out.loc[missing_week, "event_key"])
        ]
        out.loc[missing_week, "event_week"] = fallback_values
    out["event_week"] = pd.to_numeric(out["event_week"], errors="coerce")
    finite = np.isfinite(out["event_week"].to_numpy(dtype=float))
    out.loc[finite, "event_week"] = out.loc[finite, "event_week"].astype(int).clip(lower=1)
    return out[columns]


def _event_week_map(event_metadata: pd.DataFrame | None) -> dict[str, float]:
    weeks = canonical_event_week_table(event_metadata)
    if weeks.empty:
        return {}
    valid = weeks.dropna(subset=["event_week"])
    return {str(row.event_key): float(row.event_week) for row in valid.itertuples()}


def _raw_event_week_map(event_metadata: pd.DataFrame | None) -> dict[str, float]:
    if event_metadata is None or event_metadata.empty:
        return {}
    weeks = canonical_event_week_table(event_metadata)
    valid = weeks.dropna(subset=["raw_event_week"])
    return {str(row.event_key): float(row.raw_event_week) for row in valid.itertuples()}


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
    raw_event_weeks: dict[str, float],
    target_map: tuple[TargetMapping, ...],
    binary_targets: tuple[str, ...],
    diagnostic_targets: tuple[str, ...],
) -> dict[str, Any] | None:
    raw_score = _raw_score(match, color)
    if raw_score == -1:
        return None
    breakdown = _score_breakdown(match, color)

    event_key = _match_event_key(match)
    season = _extract_season(event_key, match.match_key)
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
        "has_breakdown": bool(breakdown),
        "has_surrogate": bool(
            raw_alliance.get("surrogate_team_keys") or alliance.surrogate_team_keys
        ),
        "has_dq": bool(raw_alliance.get("dq_team_keys") or alliance.dq_team_keys),
        "scheduled_time": scheduled,
        "actual_time": actual,
        "post_result_time": post,
        "sort_time": sort_time,
        "sort_ordinal": np.nan,
        "raw_event_week": raw_event_weeks.get(event_key, np.nan),
        "event_week": event_weeks.get(event_key, np.nan),
        "comp_ordinal": comp_level_ordinal(comp_level),
        "alliance_order": 2 if color == "blue" else 1,
    }
    for mapping in target_map:
        if breakdown:
            row[mapping.target_name] = _breakdown_value(
                breakdown, mapping.tba_field, "target", color
            )
        else:
            row[mapping.target_name] = np.nan
    for slot in (1, 2, 3):
        row[f"auto_tower_robot_{slot}"] = _normalize_endgame_status(
            _breakdown_optional(breakdown, f"autoTowerRobot{slot}", "None")
        )
        row[f"endgame_tower_robot_{slot}"] = _normalize_endgame_status(
            _breakdown_optional(breakdown, f"endGameTowerRobot{slot}", "None")
        )
    binary_map = {
        "energized": "energizedAchieved",
        "supercharged": "superchargedAchieved",
        "traversal": "traversalAchieved",
    }
    for target in binary_targets:
        row[target] = (
            bool(_breakdown_value(breakdown, binary_map[target], "binary", color))
            if breakdown
            else np.nan
        )
    diagnostic_map = {
        "foul_pts": "foulPoints",
        "major_foul_count": "majorFoulCount",
        "minor_foul_count": "minorFoulCount",
    }
    for target in diagnostic_targets:
        row[target] = (
            _breakdown_value(breakdown, diagnostic_map[target], "diagnostic", color)
            if breakdown
            else np.nan
        )
    row.update(v57_score_targets_for_breakdown(season, breakdown))
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
    raw_event_weeks = _raw_event_week_map(event_metadata)
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
                raw_event_weeks,
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
        "raw_event_week": red.get("raw_event_week", np.nan),
        "event_week": red["event_week"],
        "comp_ordinal": comp_level_ordinal(str(red["comp_level"])),
        "red_total_score": red["total_score"],
        "blue_total_score": blue["total_score"],
        "red_auto_pts": red["auto_pts"],
        "red_teleop_pts": red["teleop_pts"],
        "blue_auto_pts": blue["auto_pts"],
        "blue_teleop_pts": blue["teleop_pts"],
        "red_foul_pts": red["foul_pts"],
        "blue_foul_pts": blue["foul_pts"],
    }
    opts = default_options()
    for target in opts.atomic_count_targets:
        row[f"red_{target}"] = red.get(target, np.nan)
        row[f"blue_{target}"] = blue.get(target, np.nan)
    row["red_committed_foul_pts"] = blue.get("foul_pts", np.nan)
    row["blue_committed_foul_pts"] = red.get("foul_pts", np.nan)
    row["red_committed_minor_foul_count"] = blue.get("minor_foul_count", np.nan)
    row["blue_committed_minor_foul_count"] = red.get("minor_foul_count", np.nan)
    row["red_committed_major_foul_count"] = blue.get("major_foul_count", np.nan)
    row["blue_committed_major_foul_count"] = red.get("major_foul_count", np.nan)
    for target in opts.bonus_binary_targets + opts.special_binary_targets:
        row[f"red_{target}"] = red.get(target, np.nan)
        row[f"blue_{target}"] = blue.get(target, np.nan)
    for color, alliance in (("red", red), ("blue", blue)):
        for slot in (1, 2, 3):
            key = str(row[f"{color}_team_{slot}_key"])
            row[f"{color}_team_{slot}_missing_team_mask"] = not bool(key)
            row[f"{color}_team_{slot}_auto_status"] = alliance[f"auto_tower_robot_{slot}"]
            row[f"{color}_team_{slot}_endgame_status"] = alliance[f"endgame_tower_robot_{slot}"]
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


def team_number_from_key(team_key: str) -> int:
    text = str(team_key).strip()
    if not text.startswith("frc") or not text[3:].isdigit():
        raise ValueError(f"Invalid FRC team key: {team_key!r}")
    return int(text[3:])


def _team_key_to_v56_base_idx(team_key: str, *, max_team_number: int = V56_MAX_TEAM_NUMBER) -> int:
    number = team_number_from_key(team_key)
    if number > max_team_number:
        raise ValueError(
            f"Team {team_key} has number {number}, which exceeds V5.6 max_team_number="
            f"{max_team_number}. Rebuild the V5.6 prior with a larger max_team_number."
        )
    return number


def make_v5_team_index_maps(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    key_columns = _team_key_columns(table)
    missing = [column for column in key_columns if column not in table.columns]
    if missing:
        raise KeyError(f"Data table is missing team key columns: {', '.join(missing)}")
    if "event_key" not in table.columns:
        raise KeyError("Data table is missing event_key.")
    keys = sorted(
        {
            str(value)
            for column in key_columns
            for value in table[column].tolist()
            if pd.notna(value) and str(value)
        }
    )
    index_map = {key: _team_key_to_v56_base_idx(key) for key in keys}
    event_pairs = sorted(
        {
            (str(row["event_key"]), str(row[column]))
            for _, row in table[["event_key", *key_columns]].iterrows()
            for column in key_columns
            if pd.notna(row[column]) and str(row[column])
        }
    )
    event_index_map = {
        f"{event_key}::{team_key}": idx + 1
        for idx, (event_key, team_key) in enumerate(event_pairs)
    }
    out = table.copy()
    for column in key_columns:
        base_column = column.replace("_key", "_base_idx")
        event_column = column.replace("_key", "_event_idx")
        legacy_column = column.replace("_key", "_idx")
        missing_column = column.replace("_key", "_missing_team_mask")
        base_values = []
        event_values = []
        missing_values = []
        for event_key, value in zip(out["event_key"], out[column], strict=True):
            present = pd.notna(value) and bool(str(value))
            team_key = str(value) if present else ""
            base_values.append(index_map.get(team_key, 0))
            event_values.append(event_index_map.get(f"{event_key}::{team_key}", 0))
            missing_values.append(not present)
        out[base_column] = pd.Series(base_values, dtype="Int64")
        out[event_column] = pd.Series(event_values, dtype="Int64")
        out[legacy_column] = pd.Series(base_values, dtype="Int64")
        out[missing_column] = pd.Series(missing_values, dtype=bool)
    return out, index_map, event_index_map


def make_team_index_map(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out, index_map, _ = make_v5_team_index_maps(table)
    return out, index_map


def bounded_holdout_count(num_items: int, fraction: float) -> int:
    count = max(1, int(round(num_items * fraction)))
    return min(count, num_items - 1)


def _stratified_event_comp_validation_mask(
    table: pd.DataFrame,
    *,
    validation_fraction: float,
    random_seed: int,
) -> np.ndarray:
    missing = [column for column in ("event_key", "comp_level") if column not in table.columns]
    if missing:
        raise KeyError(
            "stratified-event-comp split requires columns: " + ", ".join(missing)
        )
    strata = pd.DataFrame(
        {
            "row_index": np.arange(len(table)),
            "event_key": table["event_key"].astype(str).to_numpy(),
            "comp_bucket": np.where(
                table["comp_level"].astype(str).to_numpy() == "qm", "qm", "elim"
            ),
        }
    )
    rng = np.random.default_rng(random_seed)
    validation_mask = np.zeros(len(table), dtype=bool)
    for _, rows in strata.groupby(["event_key", "comp_bucket"], sort=True):
        indices = rows["row_index"].to_numpy(dtype=int)
        if len(indices) < 2:
            continue
        holdout = bounded_holdout_count(len(indices), validation_fraction)
        validation_mask[rng.permutation(indices)[:holdout]] = True
    return validation_mask


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
    validation_fraction = (
        opts.validation_fraction if validation_fraction is None else validation_fraction
    )
    random_seed = opts.random_seed if random_seed is None else random_seed
    if validation_fraction < 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be >= 0 and < 1")
    n_rows = len(table)
    if n_rows < 2:
        raise ValueError("At least two rows are required to make a split.")
    validation_mask = np.zeros(n_rows, dtype=bool)
    fallback_reason = None
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
    elif policy == "stratified-event-comp":
        validation_mask = _stratified_event_comp_validation_mask(
            table,
            validation_fraction=validation_fraction,
            random_seed=random_seed,
        )
        if not np.any(validation_mask):
            fallback_reason = "no_stratifiable_event_comp_strata"
            holdout = bounded_holdout_count(n_rows, validation_fraction)
            order = np.argsort(table["sort_ordinal"].to_numpy())
            validation_mask[order[-holdout:]] = True
    else:
        raise ValueError(f"Unknown split policy {policy!r}.")
    train_mask = ~validation_mask
    return Split(
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=np.zeros(n_rows, dtype=bool),
        policy=policy,
        seed=random_seed if policy in ("event-held-out", "stratified-event-comp") else None,
        fallback_reason=fallback_reason,
    )


def target_matrix(table: pd.DataFrame, target_names: list[str] | tuple[str, ...]) -> np.ndarray:
    values = np.zeros((len(table), len(target_names)), dtype=float)
    for idx, name in enumerate(target_names):
        if name and name in table.columns:
            values[:, idx] = table[name].astype(float).to_numpy()
    return values


def optional_target_matrix(
    table: pd.DataFrame, target_names: list[str] | tuple[str, ...]
) -> np.ndarray:
    values = np.full((len(table), len(target_names)), np.nan, dtype=float)
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
        if target in out.columns:
            source = out[target].astype(float)
        else:
            source = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
        out[f"{target}_z"] = (source - stats.mu[idx]) / stats.sigma[idx]
    return out


def fit_v57_target_stats(
    table: pd.DataFrame, train_mask: np.ndarray, opts: LatentStratOptions | None = None
) -> TargetStats:
    opts = opts or default_options()
    target_names = v57_continuous_target_names(opts)
    if len(train_mask) != len(table):
        raise ValueError("train_mask must have one element per table row.")
    if not np.any(train_mask):
        raise ValueError("At least one training row is required.")
    values = optional_target_matrix(table, target_names)
    train = values[train_mask]
    mu = np.zeros(len(target_names), dtype=float)
    sigma = np.ones(len(target_names), dtype=float)
    for idx in range(len(target_names)):
        finite = train[:, idx][np.isfinite(train[:, idx])]
        if finite.size:
            mu[idx] = float(np.mean(finite))
        if finite.size > 1:
            sigma[idx] = float(np.std(finite, ddof=1))
    is_constant = (sigma == 0) | np.isnan(sigma) | np.isnan(mu)
    mu[np.isnan(mu)] = 0
    sigma[is_constant] = 1
    return TargetStats(target_names=target_names, mu=mu, sigma=sigma, is_constant=is_constant)


def apply_v57_target_stats(table: pd.DataFrame, stats: TargetStats) -> pd.DataFrame:
    out = table.copy()
    for idx, target in enumerate(stats.target_names):
        if target in out.columns:
            source = out[target].astype(float)
            out[f"{target}_z"] = (source - stats.mu[idx]) / stats.sigma[idx]
        else:
            out[f"{target}_z"] = np.nan
    return out
