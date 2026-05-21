"""In-memory FRC datastore used by importers and table builders."""

from __future__ import annotations

from dataclasses import dataclass, field

from frc.models import Match, ScoutingData, TbaEvent, Team


@dataclass
class FRCDataStore:
    teams: dict[str, Team] = field(default_factory=dict)
    matches: dict[str, Match] = field(default_factory=dict)
    events: dict[str, TbaEvent] = field(default_factory=dict)
    scouting_data: dict[tuple[str, str], ScoutingData] = field(default_factory=dict)

    def add_team(self, team: Team) -> None:
        if team.team_key:
            self.teams[team.team_key] = team

    def add_match(self, match: Match | list[Match]) -> None:
        matches = match if isinstance(match, list) else [match]
        for item in matches:
            self.matches[item.match_key] = item

    def add_event(self, event: TbaEvent) -> None:
        self.events[event.key] = event

    def add_scouting_data(self, scouting: ScoutingData) -> None:
        self.scouting_data[(scouting.team_key, scouting.match_key)] = scouting

    def get_scouting_data(self, team_key: str, match_key: str) -> ScoutingData | None:
        return self.scouting_data.get((team_key, match_key))
