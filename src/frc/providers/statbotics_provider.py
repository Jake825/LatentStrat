"""Thin adapter over the statbotics Python package."""

from __future__ import annotations

from typing import Any


class StatboticsProvider:
    """Stable LatentStrat-facing wrapper around `statbotics.Statbotics`."""

    def __init__(self, *, cache_dir: str = "statbotics_offline_cache") -> None:
        import statbotics
        from diskcache import Cache

        self.client = statbotics.Statbotics()
        self.cache = Cache(cache_dir)

    def _memoized(self, key: tuple[Any, ...], fetcher: Any) -> Any:
        if key in self.cache:
            return self.cache[key]
        value = fetcher()
        self.cache[key] = value
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
