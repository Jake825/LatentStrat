"""V5.5 text-only prior feature construction."""

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
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from frc.providers.tba_provider import TbaProvider
from latentstrat.config import PriorOpts

PARQUET_ENGINE = "pyarrow"
TEAM_SLOT_COLUMNS = (
    "red_team_1_key",
    "red_team_2_key",
    "red_team_3_key",
    "blue_team_1_key",
    "blue_team_2_key",
    "blue_team_3_key",
)


@dataclass(frozen=True)
class PriorFeatureMetadata:
    target_season: int
    team_count: int
    source_feature_path: str
    embedding_model: str
    llm_dim: int


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


def clean_narrative_text(text: str) -> str:
    """Normalize generated narrative text away from JSON-looking artifacts."""

    cleaned = re.sub(r"[{}\"']", " ", text)
    cleaned = re.sub(r"\b(?:None|null)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def extract_team_keys_from_feature_table(table: pd.DataFrame) -> list[str]:
    columns = [column for column in TEAM_SLOT_COLUMNS if column in table.columns]
    if not columns:
        columns = [column for column in table.columns if column.endswith("_team_key")]
    if not columns:
        raise KeyError("Feature table does not contain team key columns.")
    keys = {
        str(value)
        for column in columns
        for value in table[column].tolist()
        if pd.notna(value) and str(value)
    }
    return sorted(keys)


def read_team_keys_from_feature_file(path: str | Path) -> list[str]:
    table = pd.read_parquet(path, engine=PARQUET_ENGINE)
    return extract_team_keys_from_feature_table(table)


def _event_sort_key(event: Any) -> tuple[int, str, str]:
    year = event_year(event) or 0
    start_date = str(_field(event, "start_date", "") or "")
    key = str(_field(event, "key", "") or "")
    return year, start_date, key


def _award_sort_key(award: Any) -> tuple[int, str, str]:
    year = award_year(award) or 0
    event_key = str(_field(award, "event_key", "") or "")
    name = str(_field(award, "name", "") or "")
    return year, event_key, name


def _recipient_team_keys(award: Any) -> set[str]:
    recipients = _field(award, "recipient_list", []) or []
    return {
        str(_field(recipient, "team_key", "") or "")
        for recipient in recipients
        if _field(recipient, "team_key", None)
    }


def build_team_narrative(
    team_key: str,
    team_profile: Any,
    team_events: Sequence[Any],
    team_awards: Sequence[Any],
    target_season: int,
) -> str:
    """Build a target-season-quarantined natural-language team history."""

    profile_parts = [f"Team {team_key}."]
    nickname = _field(team_profile, "nickname")
    if nickname:
        profile_parts.append(f"Nickname {nickname}.")
    home_parts = [
        _field(team_profile, "city"),
        _field(team_profile, "state_prov"),
        _field(team_profile, "country"),
    ]
    home = ", ".join(str(part) for part in home_parts if part)
    if home:
        profile_parts.append(f"Home {home}.")
    rookie_year = _field(team_profile, "rookie_year")
    if rookie_year:
        profile_parts.append(f"Rookie year {rookie_year}.")

    awards_by_event: dict[str, list[str]] = defaultdict(list)
    for award in sorted(team_awards, key=_award_sort_key):
        year = award_year(award)
        if year is None or year >= target_season:
            continue
        if team_key not in _recipient_team_keys(award):
            continue
        event_key = str(_field(award, "event_key", "") or "")
        award_name = str(_field(award, "name", "") or "").strip()
        if award_name:
            awards_by_event[event_key].append(award_name)

    historical_events = [
        event
        for event in sorted(team_events, key=_event_sort_key)
        if (event_year(event) is not None and event_year(event) < target_season)
    ]

    event_parts: list[str] = []
    for event in historical_events:
        year = event_year(event)
        event_key = str(_field(event, "key", "") or "")
        name = str(_field(event, "name", "") or event_key).strip()
        phrase = f"Played {year} {name}."
        awards = awards_by_event.get(event_key, [])
        if awards:
            phrase += f" Won {', '.join(awards)}."
        event_parts.append(phrase)

    known_event_keys = {str(_field(event, "key", "") or "") for event in historical_events}
    for event_key, awards in awards_by_event.items():
        if event_key and event_key not in known_event_keys:
            year = _year_from_key(event_key)
            event_parts.append(f"Won {', '.join(awards)} at {year or 'historical'} {event_key}.")

    if not event_parts:
        event_parts.append("No historical target-season-safe TBA events found.")
    return clean_narrative_text(" ".join([*profile_parts, *event_parts]))


def embedding_cache_key(model: str, dimensions: int, narrative_text: str) -> str:
    payload = f"{model}\0{dimensions}\0{narrative_text}".encode()
    return hashlib.sha256(payload).hexdigest()


class OpenAIEmbeddingCache:
    """SQLite cache keyed by the exact embedding request payload."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._memory_connection = (
            sqlite3.connect(":memory:") if str(path) == ":memory:" else None
        )
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
        narrative_hash = hashlib.sha256(narrative_text.encode("utf-8")).hexdigest()
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
    retryable_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }
    return type(exc).__name__ in retryable_names


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
    narratives: Mapping[str, str],
    opts: PriorOpts | None = None,
    *,
    client: Any | None = None,
    cache: OpenAIEmbeddingCache | None = None,
    stats: EmbeddingStats | None = None,
) -> dict[str, list[float]]:
    opts = opts or PriorOpts()
    cache = cache or OpenAIEmbeddingCache(opts.cache_path)
    stats = stats or EmbeddingStats()
    outputs: dict[str, list[float]] = {}
    pending: dict[str, tuple[str, list[str]]] = {}
    for team_key, narrative in narratives.items():
        cached = cache.get(opts.embedding_model, opts.llm_dim, narrative)
        if cached is not None:
            stats.cache_hits += 1
            outputs[team_key] = cached
            continue
        stats.cache_misses += 1
        key = embedding_cache_key(opts.embedding_model, opts.llm_dim, narrative)
        if key not in pending:
            pending[key] = (narrative, [])
        pending[key][1].append(team_key)

    if pending:
        client = client or _openai_client()
        pending_items = list(pending.values())
        for item_batch in _chunks(pending_items, opts.embedding_batch_size):
            narrative_batch = [item[0] for item in item_batch]
            response = _embedding_create_with_backoff(
                client, narrative_batch, opts, stats=stats
            )
            stats.api_batches += 1
            embeddings = _response_embeddings(response)
            if len(embeddings) != len(narrative_batch):
                raise ValueError("OpenAI embedding response length did not match request length.")
            for (narrative, team_keys), embedding in zip(item_batch, embeddings, strict=True):
                cache.put(opts.embedding_model, opts.llm_dim, narrative, embedding)
                for team_key in team_keys:
                    outputs[team_key] = embedding
    return outputs


def build_prior_feature_table(
    team_keys: Sequence[str],
    provider: TbaProvider,
    target_season: int,
    opts: PriorOpts | None = None,
    *,
    client: Any | None = None,
) -> pd.DataFrame:
    opts = opts or PriorOpts()
    narratives: dict[str, str] = {}
    for team_key in sorted({str(key) for key in team_keys if str(key)}):
        profile = provider.get_team(team_key)
        events = provider.get_team_events(team_key)
        awards = provider.get_team_awards(team_key)
        narratives[team_key] = build_team_narrative(
            team_key,
            profile,
            events,
            awards,
            target_season,
        )

    stats = EmbeddingStats()
    embeddings = embed_narratives(narratives, opts, client=client, stats=stats)
    rows = []
    for team_key, narrative in narratives.items():
        rows.append(
            {
                "team_key": team_key,
                "target_season": target_season,
                "narrative": narrative,
                "narrative_hash": hashlib.sha256(narrative.encode("utf-8")).hexdigest(),
                "embedding_model": opts.embedding_model,
                "llm_dim": opts.llm_dim,
                "openai_narrative_vector": embeddings[team_key],
            }
        )
    out = pd.DataFrame(rows).sort_values("team_key").reset_index(drop=True)
    out.attrs["embedding_stats"] = stats.__dict__.copy()
    return out


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


def build_prior_feature_file(
    target_season: int,
    teams_from: str | Path,
    output: str | Path,
    provider: TbaProvider,
    opts: PriorOpts | None = None,
    *,
    client: Any | None = None,
) -> Path:
    source = Path(teams_from)
    team_keys = read_team_keys_from_feature_file(source)
    table = build_prior_feature_table(
        team_keys,
        provider,
        target_season,
        opts,
        client=client,
    )
    metadata = PriorFeatureMetadata(
        target_season=target_season,
        team_count=len(team_keys),
        source_feature_path=str(source),
        embedding_model=(opts or PriorOpts()).embedding_model,
        llm_dim=(opts or PriorOpts()).llm_dim,
    )
    table.attrs["metadata"] = metadata.__dict__
    return write_prior_feature_table(table, output)
