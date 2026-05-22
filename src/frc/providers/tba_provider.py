"""Thin adapter over tbapy for The Blue Alliance API v3."""

from __future__ import annotations

import os
from typing import Any

from frc.models import Match, TbaAward, TbaEvent, TbaMatch, TbaTeam, _as_plain_data


class TbaProvider:
    """Stable LatentStrat-facing wrapper around `tbapy.TBA`."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_name: str | None = "tba_cache",
        backend: str = "sqlite",
    ) -> None:
        self.api_key = (api_key or os.getenv("TBA_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("TBA API key is required via argument or TBA_API_KEY.")

        self.cache_name = cache_name
        if cache_name:
            import requests_cache

            requests_cache.install_cache(cache_name, backend=backend)

        import tbapy

        self.client = tbapy.TBA(self.api_key)
        self._patch_tbapy_session(cache_name, backend)

    def _patch_tbapy_session(self, cache_name: str | None, backend: str) -> None:
        """Best-effort cache shim for tbapy's class-level requests session."""
        if not cache_name:
            return
        try:
            import requests_cache

            cached = requests_cache.CachedSession(cache_name, backend=backend)
            old_session = getattr(self.client, "session", None)
            if old_session is not None and hasattr(old_session, "headers"):
                cached.headers.update(old_session.headers)
            cached.headers.update({"X-TBA-Auth-Key": self.api_key})
            self.client.session = cached
        except Exception:
            # Global requests-cache is already installed. Provider methods remain usable.
            return

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.client, name)
        return _as_plain_data(method(*args, **kwargs))

    def get_status(self) -> dict[str, Any]:
        return _as_plain_data(self.client.status())

    def get_event(self, event_key: str) -> TbaEvent:
        return TbaEvent.model_validate(self._call("event", event_key))

    def get_events_by_year(self, year: int, simple: bool = True) -> list[TbaEvent]:
        return [TbaEvent.model_validate(item) for item in self._call("events", year, simple=simple)]

    def get_event_matches(self, event_key: str) -> list[TbaMatch]:
        return [TbaMatch.model_validate(item) for item in self._call("event_matches", event_key)]

    def get_event_teams(self, event_key: str) -> list[TbaTeam]:
        return [TbaTeam.model_validate(item) for item in self._call("event_teams", event_key)]

    def get_event_rankings(self, event_key: str) -> dict[str, Any]:
        return _as_plain_data(self.client.event_rankings(event_key))

    def get_event_awards(self, event_key: str) -> list[TbaAward]:
        return [TbaAward.model_validate(item) for item in self._call("event_awards", event_key)]

    def get_event_alliances(self, event_key: str) -> list[dict[str, Any]]:
        return list(self._call("event_alliances", event_key))

    def get_event_statuses(self, event_key: str) -> list[dict[str, Any]]:
        return list(self._call("team_status", event=event_key))

    def get_event_team_keys(self, event_key: str) -> list[str]:
        return list(self._call("event_teams", event_key, keys=True))

    def get_team(self, team_key: str) -> TbaTeam:
        return TbaTeam.model_validate(self._call("team", team_key))

    def get_team_events(self, team_key: str) -> list[TbaEvent]:
        try:
            payload = self._call("team_events", team_key, simple=True)
        except TypeError:
            payload = self._call("team_events", team_key)
        return [TbaEvent.model_validate(item) for item in payload]

    def get_team_awards(self, team_key: str) -> list[TbaAward]:
        return [TbaAward.model_validate(item) for item in self._call("team_awards", team_key)]

    def get_match(self, match_key: str) -> TbaMatch:
        return TbaMatch.model_validate(self._call("match", key=match_key))

    def get_match_domain(self, match_key: str) -> Match:
        return Match.from_tba(self.get_match(match_key))
