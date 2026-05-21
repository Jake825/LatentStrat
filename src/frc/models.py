"""Typed FRC domain objects used by provider and LatentStrat ingestion code."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _as_plain_data(value: Any) -> Any:
    """Convert tbapy/statbotics objects into nested Python containers."""
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=False)
    if isinstance(value, dict):
        return {str(k): _as_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_as_plain_data(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _as_plain_data(v) for k, v in vars(value).items() if not str(k).startswith("_")
        }
    return value


class PayloadModel(BaseModel):
    """Base model for external FRC payloads.

    TBA can add fields mid-season. Ignoring extras keeps ingestion stable while
    required LatentStrat target fields are still checked at table-build time.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True, arbitrary_types_allowed=True)


class TeamMetadata(PayloadModel):
    nickname: str = ""
    team_number: int = 0
    rookie_year: int | None = None


class Team(PayloadModel):
    metadata: TeamMetadata
    latent_vector: list[float] = Field(default_factory=list)
    colors: tuple[str | None, str | None, bool] = (None, None, False)

    @property
    def team_key(self) -> str:
        return f"frc{self.metadata.team_number}" if self.metadata.team_number else ""


class MatchAlliance(PayloadModel):
    team_keys: list[str] = Field(default_factory=list)
    score: int = -1
    surrogate_team_keys: list[str] = Field(default_factory=list)
    dq_team_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_missing_lists(cls, data: Any) -> Any:
        data = _as_plain_data(data)
        if not isinstance(data, dict):
            return data
        for name in ("team_keys", "surrogate_team_keys", "dq_team_keys"):
            if data.get(name) is None:
                data[name] = []
        return data


class MatchScoreBreakdown(PayloadModel):
    """Known score-breakdown fields with raw payload retained by callers."""

    total_points: float | None = Field(default=None, alias="totalPoints")
    total_auto_points: float | None = Field(default=None, alias="totalAutoPoints")
    total_teleop_points: float | None = Field(default=None, alias="totalTeleopPoints")
    total_tower_points: float | None = Field(default=None, alias="totalTowerPoints")
    auto_tower_points: float | None = Field(default=None, alias="autoTowerPoints")
    end_game_tower_points: float | None = Field(default=None, alias="endGameTowerPoints")
    foul_points: float | None = Field(default=None, alias="foulPoints")
    major_foul_count: float | None = Field(default=None, alias="majorFoulCount")
    minor_foul_count: float | None = Field(default=None, alias="minorFoulCount")
    energized_achieved: bool | None = Field(default=None, alias="energizedAchieved")
    supercharged_achieved: bool | None = Field(default=None, alias="superchargedAchieved")
    traversal_achieved: bool | None = Field(default=None, alias="traversalAchieved")


class TbaTeam(PayloadModel):
    key: str = ""
    team_number: int = 0
    nickname: str = ""
    rookie_year: int | None = None
    name: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_raw(cls, data: Any) -> Any:
        return _as_plain_data(data)


class TbaEvent(PayloadModel):
    key: str = ""
    name: str = ""
    year: int | None = None
    event_type: int | None = None
    week: int | None = None
    start_date: str | None = None
    end_date: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_raw(cls, data: Any) -> Any:
        return _as_plain_data(data)


class TbaMatch(PayloadModel):
    key: str
    event_key: str = ""
    comp_level: str = ""
    set_number: int | float | None = None
    match_number: int | float | None = None
    time: int | float | None = None
    actual_time: int | float | None = None
    post_result_time: int | float | None = None
    alliances: dict[str, MatchAlliance] = Field(default_factory=dict)
    score_breakdown: dict[str, dict[str, Any]] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_raw(cls, data: Any) -> Any:
        raw = _as_plain_data(data)
        if not isinstance(raw, dict):
            return raw
        normalized = dict(raw)
        normalized["raw"] = raw
        alliances = normalized.get("alliances") or {}
        if isinstance(alliances, dict):
            normalized["alliances"] = {
                color: MatchAlliance.model_validate(payload)
                for color, payload in alliances.items()
                if color in {"red", "blue"}
            }
        return normalized

    def alliance(self, color: str) -> MatchAlliance:
        return self.alliances.get(color, MatchAlliance())

    def breakdown(self, color: str) -> dict[str, Any]:
        if not self.score_breakdown:
            return {}
        value = self.score_breakdown.get(color, {})
        return _as_plain_data(value) if isinstance(value, dict) else {}


class Match(PayloadModel):
    match_key: str
    red_alliance: MatchAlliance
    blue_alliance: MatchAlliance
    tba_data: dict[str, Any] = Field(default_factory=dict)
    event_key: str = ""

    @classmethod
    def from_tba(cls, match: TbaMatch) -> Match:
        return cls(
            match_key=match.key,
            red_alliance=match.alliance("red"),
            blue_alliance=match.alliance("blue"),
            tba_data=match.raw,
            event_key=match.event_key or match.key.split("_", 1)[0],
        )


class ScoutingData(PayloadModel):
    team_key: str
    match_key: str
    notes: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
