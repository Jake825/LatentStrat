"""Thin adapter over the statbotics Python package."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class StatboticsProvider:
    """Stable LatentStrat-facing wrapper around `statbotics.Statbotics`."""

    def __init__(
        self,
        *,
        cache_path: str | Path = "data/cache/statbotics.sqlite",
        cache_dir: str | Path | None = None,
    ) -> None:
        import statbotics

        self.client = statbotics.Statbotics()
        if cache_dir is not None:
            cache_path = Path(cache_dir) / "statbotics.sqlite"
        self.cache_path = Path(cache_path)
        if self.cache_path.exists() and self.cache_path.is_dir():
            self.cache_path = self.cache_path / "statbotics.sqlite"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path)

    def _init_cache(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists statbotics_cache (
                    cache_key text primary key,
                    payload_json text not null
                )
                """
            )

    def _cache_key(self, key: tuple[Any, ...]) -> str:
        return json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)

    def _memoized(self, key: tuple[Any, ...], fetcher: Any) -> Any:
        cache_key = self._cache_key(key)
        with self._connect() as connection:
            row = connection.execute(
                "select payload_json from statbotics_cache where cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                return json.loads(row[0])
        value = fetcher()
        try:
            payload_json = json.dumps(value, sort_keys=True)
        except TypeError as exc:
            raise TypeError(
                f"Statbotics response for cache key {cache_key} is not JSON-serializable."
            ) from exc
        with self._connect() as connection:
            connection.execute(
                """
                insert or replace into statbotics_cache (cache_key, payload_json)
                values (?, ?)
                """,
                (cache_key, payload_json),
            )
        return value

    def get_team(self, team: int) -> dict[str, Any]:
        return self._memoized(("team", int(team)), lambda: self.client.get_team(int(team)))

    def get_teams(self, **filters: Any) -> list[dict[str, Any]]:
        key = ("teams", tuple(sorted(filters.items())))
        return self._memoized(key, lambda: self.client.get_teams(**filters))

    def get_year(self, year: int) -> dict[str, Any]:
        return self._memoized(("year", int(year)), lambda: self.client.get_year(int(year)))

    def get_years(self, **filters: Any) -> list[dict[str, Any]]:
        key = ("years", tuple(sorted(filters.items())))
        return self._memoized(key, lambda: self.client.get_years(**filters))

    def get_event(self, event_key: str) -> dict[str, Any]:
        return self._memoized(("event", event_key), lambda: self.client.get_event(event_key))

    def get_events(self, **filters: Any) -> list[dict[str, Any]]:
        key = ("events", tuple(sorted(filters.items())))
        return self._memoized(key, lambda: self.client.get_events(**filters))

    def get_match(self, match_key: str) -> dict[str, Any]:
        return self._memoized(("match", match_key), lambda: self.client.get_match(match_key))

    def get_matches(self, **filters: Any) -> list[dict[str, Any]]:
        key = ("matches", tuple(sorted(filters.items())))
        return self._memoized(key, lambda: self.client.get_matches(**filters))

    def get_team_year(self, team: int, year: int) -> dict[str, Any]:
        return self._memoized(
            ("team_year", int(team), int(year)),
            lambda: self.client.get_team_year(int(team), int(year)),
        )

    def get_team_years(self, **filters: Any) -> list[dict[str, Any]]:
        key = ("team_years", tuple(sorted(filters.items())))
        return self._memoized(key, lambda: self.client.get_team_years(**filters))

    def get_team_event(self, team: int, event_key: str) -> dict[str, Any]:
        return self._memoized(
            ("team_event", int(team), event_key),
            lambda: self.client.get_team_event(int(team), event_key),
        )

    def get_team_events(self, **filters: Any) -> list[dict[str, Any]]:
        key = ("team_events", tuple(sorted(filters.items())))
        return self._memoized(key, lambda: self.client.get_team_events(**filters))

    def get_team_epa(self, team: int) -> float | None:
        data = self.get_team(team)
        epa = data.get("epa_end", None)
        if epa is not None:
            return float(epa)
        norm_epa = data.get("norm_epa") or {}
        current = norm_epa.get("current")
        return None if current is None else float(current)
