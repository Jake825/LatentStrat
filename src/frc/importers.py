"""Import helpers that populate FRCDataStore from provider adapters."""

from __future__ import annotations

import pandas as pd

from frc.datastore import FRCDataStore
from frc.models import Match, TbaEvent, Team, TeamMetadata
from frc.providers.tba_provider import TbaProvider


class TBAImporter:
    @staticmethod
    def import_event(provider: TbaProvider, store: FRCDataStore, event_key: str) -> TbaEvent:
        event = provider.get_event(event_key)
        store.add_event(event)
        for team_payload in provider.get_event_teams(event_key):
            store.add_team(
                Team(
                    metadata=TeamMetadata(
                        nickname=team_payload.nickname,
                        team_number=team_payload.team_number,
                        rookie_year=team_payload.rookie_year,
                    )
                )
            )
        matches = [Match.from_tba(match) for match in provider.get_event_matches(event_key)]
        store.add_match(matches)
        return event

    @staticmethod
    def import_season(
        provider: TbaProvider,
        store: FRCDataStore,
        season: int,
        *,
        event_limit: int | None = None,
    ) -> pd.DataFrame:
        events = [event for event in provider.get_events_by_year(season, simple=True) if event.event_type != 99]
        if event_limit is not None:
            events = events[:event_limit]
        for event in events:
            TBAImporter.import_event(provider, store, event.key)
        return pd.DataFrame([event.model_dump() for event in events])


class StatboticsImporter:
    @staticmethod
    def import_season_stats(provider, store: FRCDataStore, season: int) -> None:
        _ = store
        provider.get_team_years(year=season)
