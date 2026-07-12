"""V5.6 transductive prior feature construction."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from frc.providers.tba_provider import TbaProvider
from latentstrat.config import PriorOpts

PARQUET_ENGINE = "pyarrow"
GHOST_TEAM_NARRATIVE = (
    "This is a null robot. It does not exist on the field. It scores zero points. "
    "It has no autonomous routine. It does not play defense."
)
FRC_FIRST_SEASON = 1992
NORM_EPA_TARGET_COLUMNS = (
    "norm_epa_t_minus_4",
    "norm_epa_t_minus_3",
    "norm_epa_t_minus_2",
    "norm_epa_t_minus_1",
)
NORM_EPA_OBSERVED_COLUMNS = (
    "norm_epa_observed_t_minus_4",
    "norm_epa_observed_t_minus_3",
    "norm_epa_observed_t_minus_2",
    "norm_epa_observed_t_minus_1",
)
CULTURE_TARGET_COLUMNS = (
    "raw_rookie_year_delta",
    "raw_seasons_played",
    "raw_total_award_count",
    "raw_blue_banner_count",
    "raw_championship_appearance_count",
    "raw_championship_win_count",
    "raw_technical_award_count",
)
BLUE_BANNER_AWARD_TYPES = {0, 1, 9, 10}
TECHNICAL_AWARD_TYPES = {17, 18, 21, 29, 74}
BLUE_BANNER_NAME_PATTERNS = (
    "winner",
    "chairman",
    "impact",
    "engineering inspiration",
    "rookie all star",
    "rookie all-star",
)
TECHNICAL_AWARD_NAME_PATTERNS = (
    "autonomous",
    "excellence in engineering",
    "engineering excellence",
    "quality",
    "industrial design",
    "innovation in control",
)


@dataclass(frozen=True)
class KnownTeamRecord:
    team_number: int
    team_key: str
    profile: Any
    years: list[int]
    events: list[Any]
    awards: list[Any]


@dataclass(frozen=True)
class PriorFeatureMetadata:
    target_season: int
    team_count: int
    max_team_number: int
    embedding_model: str
    llm_dim: int
    norm_epa_source_years: list[int]
    norm_epa_target_names: list[str]
    culture_target_names: list[str]


@dataclass
class EmbeddingStats:
    cache_hits: int = 0
    cache_misses: int = 0
    api_batches: int = 0
    retry_count: int = 0


def _plain(record: Any) -> Any:
    if isinstance(record, BaseModel):
        return record.model_dump()
    if isinstance(record, Mapping):
        return dict(record)
    return record


def _field(record: Any, name: str, default: Any = None) -> Any:
    record = _plain(record)
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _year_from_key(value: Any) -> int | None:
    text = str(value or "")
    match = re.match(r"^(\d{4})", text)
    return int(match.group(1)) if match else None


def event_year(event: Any) -> int | None:
    year = _field(event, "year")
    if year is not None and str(year).strip():
        return int(year)
    return _year_from_key(_field(event, "key"))


def award_year(award: Any) -> int | None:
    year = _field(award, "year")
    if year is not None and str(year).strip():
        return int(year)
    return _year_from_key(_field(award, "event_key"))


def team_number_from_key(team_key: str) -> int:
    match = re.fullmatch(r"frc(\d+)", str(team_key).strip())
    if not match:
        raise ValueError(f"Invalid FRC team key: {team_key!r}")
    return int(match.group(1))


def clean_narrative_text(text: str) -> str:
    """Normalize generated narrative text away from JSON-looking artifacts."""

    cleaned = re.sub(r"[{}\"']", " ", text)
    cleaned = re.sub(r"\b(?:None|null)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def compress_years(years: Sequence[int]) -> str:
    """Convert [2001, 2002, 2003, 2005] into '2001-2003, 2005'."""

    unique_years = sorted({int(year) for year in years if year is not None})
    if not unique_years:
        return "No active years."
    ranges = []
    for _, group in groupby(enumerate(unique_years), lambda item: item[0] - item[1]):
        values = list(map(itemgetter(1), group))
        ranges.append(str(values[0]) if len(values) == 1 else f"{values[0]}-{values[-1]}")
    return ", ".join(ranges)


def _location(team_info: Any) -> str:
    parts = [
        _field(team_info, "city"),
        _field(team_info, "state_prov"),
        _field(team_info, "country"),
    ]
    return ", ".join(str(part) for part in parts if part) or "an unknown location"


def _historical_years(years: Sequence[int], target_season: int) -> list[int]:
    return sorted({int(year) for year in years if year is not None and int(year) < target_season})


def build_rich_team_narrative(
    team_info: Any,
    years_participated: Sequence[int],
    awards: Sequence[Any],
    events: Sequence[Any],
    target_season: int,
    *,
    active_in_target: bool,
) -> str:
    """Compile maximum TBA signal into a dense semantic narrative."""

    number = int(_field(team_info, "team_number", 0) or 0)
    nickname = _field(team_info, "nickname") or f"Team {number}"
    rookie_year = _field(team_info, "rookie_year") or "Unknown"
    sponsors = _field(team_info, "name") or "Unknown sponsors"
    location = _location(team_info)
    known_years = _historical_years(years_participated, target_season)
    archetype = "active Anchor team" if active_in_target else "historical Ghost team"
    narrative = [
        (
            f"Team {number}, {nickname}, is a FIRST Robotics Competition {archetype} "
            f"based in {location}."
        ),
        f"They were established in the {rookie_year} rookie class.",
        f"The team is officially supported by and affiliated with: {sponsors}.",
        (
            f"Before the {target_season} season, they were active during: "
            f"{compress_years(known_years)}."
        ),
    ]
    if active_in_target:
        narrative.append(f"They are registered as active for the {target_season} season.")
    else:
        narrative.append(
            "They are represented by their historical legacy and retired program identity."
        )

    event_wins = [
        award_year(award)
        for award in awards
        if _field(award, "award_type") == 1
        and award_year(award) is not None
        and award_year(award) < target_season
    ]
    if event_wins:
        narrative.append(
            f"They have won official events in the following years: {compress_years(event_wins)}."
        )

    championship_years = [
        event_year(event)
        for event in events
        if _field(event, "event_type") in {3, 4}
        and event_year(event) is not None
        and event_year(event) < target_season
    ]
    if championship_years:
        narrative.append(
            "They advanced to the FIRST World Championship in these years: "
            f"{compress_years(championship_years)}."
        )

    award_dict: dict[str, list[int]] = defaultdict(list)
    for award in awards:
        year = award_year(award)
        award_type = _field(award, "award_type")
        if award_type in {1, 2} or year is None or year >= target_season:
            continue
        name = str(_field(award, "name", "") or "").split("(", 1)[0].strip()
        if name:
            award_dict[name].append(int(year))

    if award_dict:
        narrative.append(
            "Throughout their history, the team has been recognized with specific awards:"
        )
        for award_name, years_won in sorted(award_dict.items()):
            narrative.append(f"{award_name}: {compress_years(years_won)}.")
    return clean_narrative_text(" ".join(narrative))


def build_gap_team_narrative(
    number: int,
    rookie_year: int,
    nearby_teams: Sequence[Mapping[str, Any]],
) -> str:
    narrative = [
        (
            "Team "
            f"{number} is currently an unassigned FIRST Robotics Competition number "
            f"from the {rookie_year} rookie class."
        ),
        (
            "This number slot is reserved for a future sibling team, junior varsity "
            "squad, or returning veteran program reboot from this specific era."
        ),
    ]
    if nearby_teams:
        neighbors = ", ".join(
            f"Team {team['number']} ({team['name']} from {team['location']})"
            for team in nearby_teams
        )
        narrative.append(
            f"Surrounding historical and active teams from this generation include: {neighbors}."
        )
        narrative.append(
            "A team taking this number would likely inherit the mentorship, "
            "historical build-culture, and community backing of the surrounding "
            "programs from this generation."
        )
    return clean_narrative_text(" ".join(narrative))


def projected_future_rookie_year(number: int, opts: PriorOpts | None = None) -> int:
    opts = opts or PriorOpts()
    return round(2026 + ((number - opts.future_baseline_team) / opts.future_growth_per_year))


def build_future_rookie_narrative(number: int, opts: PriorOpts | None = None) -> str:
    opts = opts or PriorOpts()
    projected_year = projected_future_rookie_year(number, opts)
    narrative = [
        f"Team {number} is a projected future FIRST Robotics Competition rookie team.",
        (
            "Based on current registration growth trends, this team number is "
            f"projected to debut in the {projected_year} season."
        ),
        (
            "As a modern-era rookie, this team will enter the competition landscape "
            "utilizing advanced commercial-off-the-shelf components."
        ),
        (
            "Their baseline technical capability is expected to include turnkey "
            "swerve drive systems, brushless motors, and community-driven software "
            "libraries such as WPILib and AdvantageKit."
        ),
    ]
    return clean_narrative_text(" ".join(narrative))


def _nearby_teams(number: int, known: Mapping[int, KnownTeamRecord], count_each_side: int = 2):
    lower = sorted((team for team in known if team < number), reverse=True)[:count_each_side]
    upper = sorted(team for team in known if team > number)[:count_each_side]
    rows = []
    for team_number in sorted([*lower, *upper]):
        record = known[team_number]
        rows.append(
            {
                "number": record.team_number,
                "name": _field(record.profile, "nickname") or f"Team {record.team_number}",
                "location": _location(record.profile),
                "rookie_year": _field(record.profile, "rookie_year"),
            }
        )
    return rows


def _gap_rookie_year(number: int, known: Mapping[int, KnownTeamRecord], target_season: int) -> int:
    nearby_years = [
        int(team["rookie_year"])
        for team in _nearby_teams(number, known)
        if team.get("rookie_year") is not None
    ]
    return round(median(nearby_years)) if nearby_years else int(target_season)


def build_known_team_universe(
    provider: TbaProvider,
    opts: PriorOpts | None = None,
) -> dict[int, KnownTeamRecord]:
    opts = opts or PriorOpts()
    known: dict[int, KnownTeamRecord] = {}
    page = 0
    while True:
        teams = provider.get_teams_page(page, simple=True)
        if not teams:
            break
        for listed in teams:
            number = int(_field(listed, "team_number", 0) or 0)
            if number <= 0 or number > opts.max_team_number:
                continue
            team_key = str(_field(listed, "key") or f"frc{number}")
            try:
                profile = provider.get_team(team_key)
            except Exception:
                profile = listed
            try:
                years = provider.get_team_years(team_key)
            except Exception:
                years = []
            try:
                events = provider.get_team_events(team_key)
            except Exception:
                events = []
            try:
                awards = provider.get_team_awards(team_key)
            except Exception:
                awards = []
            known[number] = KnownTeamRecord(number, team_key, profile, years, events, awards)
        page += 1
    return known


def build_prior_narratives(
    known: Mapping[int, KnownTeamRecord],
    target_season: int,
    opts: PriorOpts | None = None,
) -> pd.DataFrame:
    opts = opts or PriorOpts()
    rows = [
        {
            "team_number": 0,
            "team_key": "frc0",
            "target_season": target_season,
            "archetype": "ghost_token",
            "narrative": GHOST_TEAM_NARRATIVE,
            "narrative_hash": hashlib.sha256(GHOST_TEAM_NARRATIVE.encode()).hexdigest(),
        }
    ]
    for number in range(1, opts.max_team_number + 1):
        record = known.get(number)
        if record is not None:
            active = int(target_season) in set(record.years)
            archetype = "anchor" if active else "ghost"
            narrative = build_rich_team_narrative(
                record.profile,
                record.years,
                record.awards,
                record.events,
                target_season,
                active_in_target=active,
            )
            team_key = record.team_key
        elif number < opts.future_start_number:
            archetype = "sibling"
            team_key = f"frc{number}"
            narrative = build_gap_team_narrative(
                number,
                _gap_rookie_year(number, known, target_season),
                _nearby_teams(number, known),
            )
        else:
            archetype = "future"
            team_key = f"frc{number}"
            narrative = build_future_rookie_narrative(number, opts)
        rows.append(
            {
                "team_number": number,
                "team_key": team_key,
                "target_season": target_season,
                "archetype": archetype,
                "narrative": narrative,
                "narrative_hash": hashlib.sha256(narrative.encode()).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def _nested_field(record: Any, path: Sequence[str], default: Any = None) -> Any:
    value = record
    for part in path:
        value = _plain(value)
        if isinstance(value, Mapping):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
        if value is default:
            return default
    return value


def _finite_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def extract_statbotics_norm_epa(row: Any) -> float | None:
    """Extract Statbotics' own normalized EPA from a team-year row."""

    candidates = [
        _nested_field(row, ("epa", "norm")),
        _nested_field(row, ("norm_epa", "current")),
        _field(row, "norm_epa_current"),
        _field(row, "norm_epa"),
    ]
    for value in candidates:
        result = _finite_float(value)
        if result is not None:
            return result
    return None


def _statbotics_team_number(row: Any) -> int | None:
    value = _field(row, "team")
    if value is None:
        value = _field(row, "team_number")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_statbotics_team_year_rows(
    statbotics_provider: Any,
    source_year: int,
    opts: PriorOpts | None = None,
) -> list[dict[str, Any]]:
    """Fetch all available Statbotics team-year rows for one source year."""

    opts = opts or PriorOpts()
    limit = 1_000
    offset = 0
    output: list[dict[str, Any]] = []
    while True:
        try:
            page = statbotics_provider.get_team_years(
                year=int(source_year),
                limit=limit,
                offset=offset,
            )
        except TypeError:
            page = statbotics_provider.get_team_years(year=int(source_year))
            offset = limit
        if not page:
            break
        output.extend(list(page))
        if len(page) < limit:
            break
        offset += limit
    return output


def norm_epa_source_years(target_season: int, *, year_count: int = 4) -> list[int]:
    return [int(target_season) - offset for offset in range(year_count, 0, -1)]


def collect_norm_epa_trajectory(
    statbotics_provider: Any,
    target_season: int,
    opts: PriorOpts | None = None,
    *,
    source_years: Sequence[int] | None = None,
) -> dict[int, dict[int, float]]:
    """Fetch Statbotics normalized EPA trajectories keyed by team and source year."""

    opts = opts or PriorOpts()
    years = list(source_years or norm_epa_source_years(target_season))
    output: dict[int, dict[int, float]] = {}
    for source_year in years:
        for row in _collect_statbotics_team_year_rows(statbotics_provider, source_year, opts):
            team_number = _statbotics_team_number(row)
            if team_number is None or team_number <= 0 or team_number > opts.max_team_number:
                continue
            norm_epa = extract_statbotics_norm_epa(row)
            if norm_epa is not None:
                output.setdefault(int(team_number), {})[int(source_year)] = float(norm_epa)
    return output


def _is_team_recipient_award(award: Any) -> bool:
    recipients = _field(award, "recipient_list", []) or []
    if not recipients:
        return True
    return any(_field(recipient, "team_key") for recipient in recipients)


def _award_name(award: Any) -> str:
    return str(_field(award, "name", "") or "").lower()


def _matches_any_name_pattern(award: Any, patterns: Sequence[str]) -> bool:
    name = _award_name(award)
    return any(pattern in name for pattern in patterns)


def _is_blue_banner_award(award: Any) -> bool:
    award_type = _field(award, "award_type")
    return award_type in BLUE_BANNER_AWARD_TYPES or _matches_any_name_pattern(
        award, BLUE_BANNER_NAME_PATTERNS
    )


def _is_technical_award(award: Any) -> bool:
    award_type = _field(award, "award_type")
    return award_type in TECHNICAL_AWARD_TYPES or _matches_any_name_pattern(
        award, TECHNICAL_AWARD_NAME_PATTERNS
    )


def _event_by_key(events: Sequence[Any]) -> dict[str, Any]:
    return {str(_field(event, "key", "")): event for event in events if _field(event, "key")}


def _historical_awards(awards: Sequence[Any], target_season: int) -> list[Any]:
    return [
        award
        for award in awards
        if award_year(award) is not None
        and int(award_year(award) or 0) < int(target_season)
        and _is_team_recipient_award(award)
    ]


def _culture_values(
    record: KnownTeamRecord | None,
    target_season: int,
) -> dict[str, float]:
    if record is None:
        return {column: 0.0 for column in CULTURE_TARGET_COLUMNS}
    rookie_year = _finite_float(_field(record.profile, "rookie_year"))
    rookie_delta = max(float(rookie_year) - FRC_FIRST_SEASON, 0.0) if rookie_year else 0.0
    historical_years = _historical_years(record.years, target_season)
    historical_awards = _historical_awards(record.awards, target_season)
    events_by_key = _event_by_key(record.events)
    championship_events = [
        event
        for event in record.events
        if event_year(event) is not None
        and int(event_year(event) or 0) < int(target_season)
        and _field(event, "event_type") in {3, 4}
    ]
    championship_event_keys = {
        str(_field(event, "key", "")) for event in championship_events if _field(event, "key")
    }
    championship_wins = 0
    for award in historical_awards:
        if _field(award, "award_type") != 1:
            continue
        event_key = str(_field(award, "event_key", "") or "")
        event = events_by_key.get(event_key)
        if event_key in championship_event_keys or (
            event is not None and _field(event, "event_type") in {3, 4}
        ):
            championship_wins += 1
    return {
        "raw_rookie_year_delta": rookie_delta,
        "raw_seasons_played": float(len(historical_years)),
        "raw_total_award_count": float(len(historical_awards)),
        "raw_blue_banner_count": float(
            sum(1 for award in historical_awards if _is_blue_banner_award(award))
        ),
        "raw_championship_appearance_count": float(
            len({str(_field(event, "key", "")) for event in championship_events})
        ),
        "raw_championship_win_count": float(championship_wins),
        "raw_technical_award_count": float(
            sum(1 for award in historical_awards if _is_technical_award(award))
        ),
    }


def add_culture_targets(
    table: pd.DataFrame,
    known: Mapping[int, KnownTeamRecord],
    target_season: int,
) -> pd.DataFrame:
    output = table.copy()
    values_by_column = {column: [] for column in CULTURE_TARGET_COLUMNS}
    for value in output["team_number"]:
        values = _culture_values(known.get(int(value)), target_season)
        for column in CULTURE_TARGET_COLUMNS:
            values_by_column[column].append(float(values[column]))
    for column, values in values_by_column.items():
        output[column] = values
    return output


def add_norm_epa_trajectory_targets(
    table: pd.DataFrame,
    statbotics_provider: Any,
    target_season: int,
    opts: PriorOpts | None = None,
    *,
    source_years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Add Statbotics normalized EPA trajectory targets to the prior feature table."""

    opts = opts or PriorOpts()
    years = list(source_years or norm_epa_source_years(target_season))
    if len(years) != len(NORM_EPA_TARGET_COLUMNS):
        raise ValueError(f"Expected {len(NORM_EPA_TARGET_COLUMNS)} normalized EPA source years.")
    norm_epa_by_team = collect_norm_epa_trajectory(
        statbotics_provider, target_season, opts, source_years=years
    )
    output = table.copy()
    for column, observed_column, source_year in zip(
        NORM_EPA_TARGET_COLUMNS,
        NORM_EPA_OBSERVED_COLUMNS,
        years,
        strict=True,
    ):
        values = []
        observed = []
        for value in output["team_number"]:
            team_number = int(value)
            norm_epa = norm_epa_by_team.get(team_number, {}).get(int(source_year))
            values.append(float(norm_epa) if norm_epa is not None else np.nan)
            observed.append(bool(norm_epa is not None))
        output[column] = values
        output[observed_column] = observed
    output["norm_epa_source_years"] = ",".join(str(year) for year in years)
    observed_count = int(output[list(NORM_EPA_OBSERVED_COLUMNS)].to_numpy(dtype=bool).sum())
    if observed_count == 0:
        year_text = ", ".join(str(year) for year in years)
        raise ValueError(
            "No Statbotics normalized EPA observations were found for prior source years "
            f"{year_text}. Expected team-year rows to expose normalized EPA at `epa.norm` "
            "with fallbacks `norm_epa.current`, `norm_epa_current`, or numeric `norm_epa`."
        )
    return output


def embedding_cache_key(model: str, dimensions: int, narrative_text: str) -> str:
    payload = f"{model}\0{dimensions}\0{narrative_text}".encode()
    return hashlib.sha256(payload).hexdigest()


class OpenAIEmbeddingCache:
    """SQLite cache keyed by the exact embedding request payload."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._memory_connection = sqlite3.connect(":memory:") if str(path) == ":memory:" else None
        if self._memory_connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        return sqlite3.connect(self.path)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists embeddings (
                    cache_key text primary key,
                    model text not null,
                    dimensions integer not null,
                    narrative_hash text not null,
                    embedding_json text not null,
                    created_at text not null default current_timestamp
                )
                """
            )

    def get(self, model: str, dimensions: int, narrative_text: str) -> list[float] | None:
        key = embedding_cache_key(model, dimensions, narrative_text)
        with self._connect() as connection:
            row = connection.execute(
                "select embedding_json from embeddings where cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        embedding = json.loads(row[0])
        if len(embedding) != dimensions:
            return None
        return [float(value) for value in embedding]

    def put(
        self,
        model: str,
        dimensions: int,
        narrative_text: str,
        embedding: Sequence[float],
    ) -> None:
        if len(embedding) != dimensions:
            raise ValueError(f"Embedding width {len(embedding)} does not match {dimensions}.")
        key = embedding_cache_key(model, dimensions, narrative_text)
        narrative_hash = hashlib.sha256(narrative_text.encode()).hexdigest()
        with self._connect() as connection:
            connection.execute(
                """
                insert or replace into embeddings
                    (cache_key, model, dimensions, narrative_hash, embedding_json)
                values (?, ?, ?, ?, ?)
                """,
                (
                    key,
                    model,
                    dimensions,
                    narrative_hash,
                    json.dumps([float(value) for value in embedding]),
                ),
            )


def _openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def _response_embeddings(response: Any) -> list[list[float]]:
    data = _field(response, "data", []) or []
    return [[float(value) for value in _field(item, "embedding", [])] for item in data]


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("embedding_batch_size must be positive.")
    return [list(values[idx : idx + size]) for idx in range(0, len(values), size)]


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return int(status)
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return int(response_status) if response_status is not None else None


def _is_retryable_embedding_error(exc: Exception) -> bool:
    status = _status_code(exc)
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def _embedding_create_with_backoff(
    client: Any,
    narratives: Sequence[str],
    opts: PriorOpts,
    *,
    stats: EmbeddingStats,
) -> Any:
    delay = float(opts.embedding_retry_initial_delay)
    rng = random.Random(opts.random_seed)
    for attempt in range(opts.max_embedding_retries + 1):
        try:
            return client.embeddings.create(
                model=opts.embedding_model,
                input=list(narratives),
                dimensions=opts.llm_dim,
            )
        except Exception as exc:
            if attempt >= opts.max_embedding_retries or not _is_retryable_embedding_error(exc):
                raise
            stats.retry_count += 1
            jitter = rng.uniform(0.0, max(float(opts.embedding_retry_jitter), 0.0))
            time.sleep(min(delay + jitter, float(opts.embedding_retry_max_delay)))
            delay = min(
                delay * float(opts.embedding_retry_multiplier),
                float(opts.embedding_retry_max_delay),
            )
    raise RuntimeError("Embedding retry loop exited unexpectedly.")


def embed_narratives(
    narratives: Mapping[int | str, str],
    opts: PriorOpts | None = None,
    *,
    client: Any | None = None,
    cache: OpenAIEmbeddingCache | None = None,
    stats: EmbeddingStats | None = None,
) -> dict[int | str, list[float]]:
    opts = opts or PriorOpts()
    cache = cache or OpenAIEmbeddingCache(opts.cache_path)
    stats = stats or EmbeddingStats()
    outputs: dict[int | str, list[float]] = {}
    pending: dict[str, tuple[str, list[int | str]]] = {}
    for key, narrative in narratives.items():
        cached = cache.get(opts.embedding_model, opts.llm_dim, narrative)
        if cached is not None:
            stats.cache_hits += 1
            outputs[key] = cached
            continue
        stats.cache_misses += 1
        cache_key = embedding_cache_key(opts.embedding_model, opts.llm_dim, narrative)
        pending.setdefault(cache_key, (narrative, []))[1].append(key)

    if pending:
        client = client or _openai_client()
        pending_items = list(pending.values())
        for item_batch in _chunks(pending_items, opts.embedding_batch_size):
            narrative_batch = [item[0] for item in item_batch]
            response = _embedding_create_with_backoff(client, narrative_batch, opts, stats=stats)
            stats.api_batches += 1
            embeddings = _response_embeddings(response)
            if len(embeddings) != len(narrative_batch):
                raise ValueError("OpenAI embedding response length did not match request length.")
            for (narrative, keys), embedding in zip(item_batch, embeddings, strict=True):
                cache.put(opts.embedding_model, opts.llm_dim, narrative, embedding)
                for key in keys:
                    outputs[key] = embedding
    return outputs


def build_prior_feature_table(
    provider: TbaProvider,
    target_season: int,
    opts: PriorOpts | None = None,
    *,
    client: Any | None = None,
    known: Mapping[int, KnownTeamRecord] | None = None,
    statbotics_provider: Any | None = None,
    epa_source_year: int | None = None,
) -> pd.DataFrame:
    opts = opts or PriorOpts()
    if statbotics_provider is None:
        from frc.providers.statbotics_provider import StatboticsProvider

        statbotics_provider = StatboticsProvider()
    known = known if known is not None else build_known_team_universe(provider, opts)
    source_years = (
        [
            int(epa_source_year) - 3,
            int(epa_source_year) - 2,
            int(epa_source_year) - 1,
            int(epa_source_year),
        ]
        if epa_source_year is not None
        else norm_epa_source_years(target_season)
    )
    table = build_prior_narratives(known, target_season, opts)
    table = add_culture_targets(table, known, target_season)
    table = add_norm_epa_trajectory_targets(
        table,
        statbotics_provider,
        target_season,
        opts,
        source_years=source_years,
    )
    stats = EmbeddingStats()
    embeddings = embed_narratives(
        dict(zip(table["team_number"], table["narrative"], strict=True)),
        opts,
        client=client,
        stats=stats,
    )
    table = table.copy()
    table["embedding_model"] = opts.embedding_model
    table["llm_dim"] = opts.llm_dim
    table["openai_narrative_vector"] = [
        embeddings[int(team_number)] for team_number in table["team_number"]
    ]
    table.attrs["embedding_stats"] = stats.__dict__.copy()
    table.attrs["metadata"] = PriorFeatureMetadata(
        target_season=target_season,
        team_count=len(table),
        max_team_number=opts.max_team_number,
        embedding_model=opts.embedding_model,
        llm_dim=opts.llm_dim,
        norm_epa_source_years=source_years,
        norm_epa_target_names=list(NORM_EPA_TARGET_COLUMNS),
        culture_target_names=list(CULTURE_TARGET_COLUMNS),
    ).__dict__
    return table


def write_prior_feature_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, engine=PARQUET_ENGINE, index=False)
    return path


def read_prior_feature_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Prior feature file does not exist: {path}")
    return pd.read_parquet(path, engine=PARQUET_ENGINE)
