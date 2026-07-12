"""Durable raw TBA match-breakdown corpus and historical synchronization."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from frc.providers.tba_provider import TbaJsonResponse, TbaProvider

CORPUS_SCHEMA_VERSION = 1
DEFAULT_EVENT_TYPES = frozenset(range(6))
FOC_EVENT_TYPE = 6
REMOTE_EVENT_TYPE = 7
TBA_OPENAPI_URL = "https://www.thebluealliance.com/swagger/api_v3.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OpenApiMetadata:
    version: str | None
    swagger_sha256: str | None


@dataclass(frozen=True)
class SyncResult:
    run_id: int
    status: str
    start_season: int
    end_season: int
    event_types: tuple[int, ...]
    counts: dict[str, int]
    errors: tuple[str, ...]


class MatchBreakdownCorpus:
    """SQLite corpus storing raw TBA responses separately from HTTP request caching."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                create table if not exists ingestion_runs (
                    run_id integer primary key autoincrement,
                    started_at text not null,
                    completed_at text,
                    start_season integer not null,
                    end_season integer not null,
                    include_foc integer not null,
                    include_remote integer not null,
                    refresh integer not null,
                    status text not null,
                    counts_json text not null,
                    errors_json text not null,
                    corpus_schema_version integer not null,
                    tba_api_version text,
                    swagger_sha256 text
                );
                create table if not exists source_responses (
                    source_url text primary key,
                    response_etag text,
                    fetched_at text not null,
                    payload_hash text not null,
                    raw_json text not null
                );
                create table if not exists events (
                    event_key text primary key,
                    season integer not null,
                    event_type integer,
                    source_url text not null,
                    response_etag text,
                    fetched_at text not null,
                    payload_hash text not null,
                    raw_json text not null
                );
                create table if not exists matches (
                    match_key text primary key,
                    event_key text not null,
                    season integer not null,
                    source_url text not null,
                    response_etag text,
                    fetched_at text not null,
                    payload_hash text not null,
                    played integer not null,
                    has_breakdown integer not null,
                    raw_json text not null
                );
                create index if not exists idx_matches_season on matches(season);
                create index if not exists idx_matches_event on matches(event_key);
                """
            )

    def source_response(self, source_url: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "select * from source_responses where source_url = ?", (source_url,)
            ).fetchone()

    def upsert_source_response(
        self, source_url: str, payload: Any, *, response_etag: str | None, fetched_at: str
    ) -> str:
        raw_json = _json_text(payload)
        payload_hash = _sha256_text(raw_json)
        with self.connect() as connection:
            connection.execute(
                """
                insert into source_responses (
                    source_url, response_etag, fetched_at, payload_hash, raw_json
                ) values (?, ?, ?, ?, ?)
                on conflict(source_url) do update set
                    response_etag=excluded.response_etag,
                    fetched_at=excluded.fetched_at,
                    payload_hash=excluded.payload_hash,
                    raw_json=excluded.raw_json
                """,
                (source_url, response_etag, fetched_at, payload_hash, raw_json),
            )
        return payload_hash

    def upsert_event(
        self,
        event: dict[str, Any],
        *,
        source_url: str,
        response_etag: str | None,
        fetched_at: str,
    ) -> None:
        raw_json = _json_text(event)
        with self.connect() as connection:
            connection.execute(
                """
                insert into events (
                    event_key, season, event_type, source_url, response_etag,
                    fetched_at, payload_hash, raw_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(event_key) do update set
                    season=excluded.season,
                    event_type=excluded.event_type,
                    source_url=excluded.source_url,
                    response_etag=excluded.response_etag,
                    fetched_at=excluded.fetched_at,
                    payload_hash=excluded.payload_hash,
                    raw_json=excluded.raw_json
                """,
                (
                    str(event["key"]),
                    int(event["year"]),
                    event.get("event_type"),
                    source_url,
                    response_etag,
                    fetched_at,
                    _sha256_text(raw_json),
                    raw_json,
                ),
            )

    def upsert_match(
        self,
        match: dict[str, Any],
        *,
        source_url: str,
        response_etag: str | None,
        fetched_at: str,
    ) -> None:
        raw_json = _json_text(match)
        alliances = match.get("alliances")
        red_score = alliances.get("red", {}).get("score") if isinstance(alliances, dict) else None
        blue_score = alliances.get("blue", {}).get("score") if isinstance(alliances, dict) else None
        played = red_score not in (None, -1) and blue_score not in (None, -1)
        breakdown = match.get("score_breakdown")
        has_breakdown = isinstance(breakdown, dict) and all(
            isinstance(breakdown.get(color), dict) for color in ("red", "blue")
        )
        with self.connect() as connection:
            connection.execute(
                """
                insert into matches (
                    match_key, event_key, season, source_url, response_etag,
                    fetched_at, payload_hash, played, has_breakdown, raw_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(match_key) do update set
                    event_key=excluded.event_key,
                    season=excluded.season,
                    source_url=excluded.source_url,
                    response_etag=excluded.response_etag,
                    fetched_at=excluded.fetched_at,
                    payload_hash=excluded.payload_hash,
                    played=excluded.played,
                    has_breakdown=excluded.has_breakdown,
                    raw_json=excluded.raw_json
                """,
                (
                    str(match["key"]),
                    str(match["event_key"]),
                    int(str(match["event_key"])[:4]),
                    source_url,
                    response_etag,
                    fetched_at,
                    _sha256_text(raw_json),
                    int(played),
                    int(has_breakdown),
                    raw_json,
                ),
            )

    def start_run(
        self,
        *,
        start_season: int,
        end_season: int,
        include_foc: bool,
        include_remote: bool,
        refresh: bool,
        openapi: OpenApiMetadata,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                insert into ingestion_runs (
                    started_at, start_season, end_season, include_foc, include_remote,
                    refresh, status, counts_json, errors_json, corpus_schema_version,
                    tba_api_version, swagger_sha256
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    start_season,
                    end_season,
                    int(include_foc),
                    int(include_remote),
                    int(refresh),
                    "running",
                    "{}",
                    "[]",
                    CORPUS_SCHEMA_VERSION,
                    openapi.version,
                    openapi.swagger_sha256,
                ),
            )
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, *, status: str, counts: dict[str, int], errors: list[str]):
        with self.connect() as connection:
            connection.execute(
                """
                update ingestion_runs
                set completed_at = ?, status = ?, counts_json = ?, errors_json = ?
                where run_id = ?
                """,
                (_utc_now(), status, _json_text(counts), _json_text(errors), run_id),
            )

    def raw_matches(
        self,
        *,
        start_season: int,
        end_season: int,
        event_types: set[int] | frozenset[int] | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            select matches.raw_json
            from matches
            join events on events.event_key = matches.event_key
            where matches.season between ? and ?
        """
        parameters: list[Any] = [start_season, end_season]
        if event_types is not None:
            ordered_types = sorted(event_types)
            placeholders = ", ".join("?" for _ in ordered_types)
            query += f" and events.event_type in ({placeholders})"
            parameters.extend(ordered_types)
        query += " order by matches.season, matches.event_key, matches.match_key"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [json.loads(row["raw_json"]) for row in rows]


def fetch_openapi_metadata() -> OpenApiMetadata:
    """Read the public spec version and content hash without requiring a TBA API key."""

    request = urllib.request.Request(
        TBA_OPENAPI_URL,
        headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0 LatentStrat/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
        payload = json.loads(content)
        return OpenApiMetadata(
            version=str(payload.get("info", {}).get("version") or "") or None,
            swagger_sha256=hashlib.sha256(content).hexdigest(),
        )
    except Exception:
        return OpenApiMetadata(version=None, swagger_sha256=None)


def _load_response_payload(
    corpus: MatchBreakdownCorpus,
    provider: TbaProvider,
    path: str,
    *,
    refresh: bool,
) -> tuple[Any, TbaJsonResponse, str]:
    source_url = f"https://www.thebluealliance.com/api/v3/{path.lstrip('/')}"
    previous = corpus.source_response(source_url)
    etag = previous["response_etag"] if refresh and previous is not None else None
    response: TbaJsonResponse | None = None
    for attempt in range(3):
        try:
            response = provider.get_json_response(path, etag=etag, refresh=refresh)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    assert response is not None
    fetched_at = _utc_now()
    if response.status_code == 304:
        if previous is None:
            raise ValueError(f"TBA returned 304 without a stored payload for {source_url}.")
        response = TbaJsonResponse(
            source_url=response.source_url,
            status_code=response.status_code,
            payload=response.payload,
            response_etag=response.response_etag or previous["response_etag"],
            from_cache=response.from_cache,
        )
        return json.loads(previous["raw_json"]), response, fetched_at
    corpus.upsert_source_response(
        response.source_url,
        response.payload,
        response_etag=response.response_etag,
        fetched_at=fetched_at,
    )
    return response.payload, response, fetched_at


def sync_match_breakdowns(
    provider: TbaProvider,
    corpus_path: str | Path,
    *,
    start_season: int = 2015,
    end_season: int = 2026,
    include_foc: bool = False,
    include_remote: bool = False,
    refresh: bool = False,
    openapi_metadata: OpenApiMetadata | None = None,
) -> SyncResult:
    """Synchronize official historical event matches into a durable raw SQLite corpus."""

    if start_season > end_season:
        raise ValueError("start_season must be less than or equal to end_season.")
    corpus = MatchBreakdownCorpus(corpus_path)
    openapi = openapi_metadata or fetch_openapi_metadata()
    event_types = set(DEFAULT_EVENT_TYPES)
    if include_foc:
        event_types.add(FOC_EVENT_TYPE)
    if include_remote:
        event_types.add(REMOTE_EVENT_TYPE)
    run_id = corpus.start_run(
        start_season=start_season,
        end_season=end_season,
        include_foc=include_foc,
        include_remote=include_remote,
        refresh=refresh,
        openapi=openapi,
    )
    counts = {
        "seasons_requested": end_season - start_season + 1,
        "events_seen": 0,
        "events_eligible": 0,
        "events_excluded_type": 0,
        "matches_upserted": 0,
        "matches_played": 0,
        "matches_with_breakdown": 0,
    }
    errors = []
    for season in range(start_season, end_season + 1):
        try:
            events, response, fetched_at = _load_response_payload(
                corpus, provider, f"events/{season}/simple", refresh=refresh
            )
        except Exception as exc:
            errors.append(f"events/{season}/simple: {exc}")
            continue
        if not isinstance(events, list):
            errors.append(f"events/{season}/simple: expected list payload")
            continue
        counts["events_seen"] += len(events)
        for event in events:
            if not isinstance(event, dict):
                errors.append(f"events/{season}/simple: invalid event payload")
                continue
            if event.get("event_type") not in event_types:
                counts["events_excluded_type"] += 1
                continue
            counts["events_eligible"] += 1
            corpus.upsert_event(
                event,
                source_url=response.source_url,
                response_etag=response.response_etag,
                fetched_at=fetched_at,
            )
            event_key = str(event["key"])
            try:
                matches, match_response, matches_fetched_at = _load_response_payload(
                    corpus, provider, f"event/{event_key}/matches", refresh=refresh
                )
            except Exception as exc:
                errors.append(f"event/{event_key}/matches: {exc}")
                continue
            if not isinstance(matches, list):
                errors.append(f"event/{event_key}/matches: expected list payload")
                continue
            for match in matches:
                if not isinstance(match, dict):
                    errors.append(f"event/{event_key}/matches: invalid match payload")
                    continue
                corpus.upsert_match(
                    match,
                    source_url=match_response.source_url,
                    response_etag=match_response.response_etag,
                    fetched_at=matches_fetched_at,
                )
                counts["matches_upserted"] += 1
                alliances = match.get("alliances")
                if isinstance(alliances, dict) and all(
                    alliances.get(color, {}).get("score") not in (None, -1)
                    for color in ("red", "blue")
                ):
                    counts["matches_played"] += 1
                breakdown = match.get("score_breakdown")
                if isinstance(breakdown, dict) and all(
                    isinstance(breakdown.get(color), dict) for color in ("red", "blue")
                ):
                    counts["matches_with_breakdown"] += 1
    status = "failed" if errors else "complete"
    corpus.finish_run(run_id, status=status, counts=counts, errors=errors)
    return SyncResult(
        run_id=run_id,
        status=status,
        start_season=start_season,
        end_season=end_season,
        event_types=tuple(sorted(event_types)),
        counts=counts,
        errors=tuple(errors),
    )
