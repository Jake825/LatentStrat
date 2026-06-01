"""Parse raw TBA match payloads into alliance score-breakdown rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AllianceBreakdownRow:
    row_id: str
    season: int
    event_key: str
    match_key: str
    comp_level: str
    set_number: int
    match_number: int
    alliance: Literal["red", "blue"]
    alliance_score: int | float
    opponent_score: int | float
    winning_alliance: str
    flat_breakdown: dict[str, Any]


def flatten_breakdown(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a TBA score breakdown into stable path/value pairs."""

    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key in sorted(obj):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_breakdown(obj[key], path))
        return out
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            path = f"{prefix}[{index}]"
            out.update(flatten_breakdown(value, path))
        if not obj:
            out[prefix] = []
        return out
    out[prefix] = obj
    return out


def _posted_score(match: dict[str, Any], color: str) -> int | float | None:
    alliances = match.get("alliances")
    if not isinstance(alliances, dict):
        return None
    alliance = alliances.get(color)
    if not isinstance(alliance, dict):
        return None
    score = alliance.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or score == -1:
        return None
    return score


def extract_alliance_rows(match: dict[str, Any]) -> list[AllianceBreakdownRow]:
    """Return two rows only when a raw match has complete posted alliance breakdowns."""

    breakdown = match.get("score_breakdown")
    if not isinstance(breakdown, dict):
        return []
    if not all(isinstance(breakdown.get(color), dict) for color in ("red", "blue")):
        return []
    red_score = _posted_score(match, "red")
    blue_score = _posted_score(match, "blue")
    if red_score is None or blue_score is None:
        return []
    match_key = str(match.get("key", ""))
    event_key = str(match.get("event_key", ""))
    if not match_key or not event_key:
        return []
    season = int(event_key[:4])
    rows = []
    for color, score, opponent_score in (
        ("red", red_score, blue_score),
        ("blue", blue_score, red_score),
    ):
        rows.append(
            AllianceBreakdownRow(
                row_id=f"{match_key}:{color}",
                season=season,
                event_key=event_key,
                match_key=match_key,
                comp_level=str(match.get("comp_level", "")),
                set_number=int(match.get("set_number", 1) or 1),
                match_number=int(match.get("match_number", 1) or 1),
                alliance=color,
                alliance_score=score,
                opponent_score=opponent_score,
                winning_alliance=str(match.get("winning_alliance", "")),
                flat_breakdown=flatten_breakdown(breakdown[color]),
            )
        )
    return rows
