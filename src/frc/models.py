"""Typed FRC domain objects used by provider and LatentStrat ingestion code."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        if not isinstance(data, dict):
            return data
        for name in ("team_keys", "surrogate_team_keys", "dq_team_keys"):
            if data.get(name) is None:
                data[name] = []
        return data


class TbaTeam(PayloadModel):
    key: str = ""
    team_number: int = 0
    nickname: str = ""
    rookie_year: int | None = None
    name: str = ""
    city: str | None = None
    state_prov: str | None = None
    country: str | None = None


class TbaEvent(PayloadModel):
    key: str = ""
    name: str = ""
    year: int | None = None
    event_type: int | None = None
    week: int | None = None
    start_date: str | None = None
    end_date: str | None = None


class TbaAwardRecipient(PayloadModel):
    team_key: str | None = None
    awardee: str | None = None


class TbaAward(PayloadModel):
    name: str = ""
    award_type: int | None = None
    event_key: str = ""
    recipient_list: list[TbaAwardRecipient] = Field(default_factory=list)
    year: int | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_raw(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        recipients = normalized.get("recipient_list") or []
        normalized["recipient_list"] = [
            TbaAwardRecipient.model_validate(recipient) for recipient in recipients
        ]
        return normalized


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
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["raw"] = data
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
        return value if isinstance(value, dict) else {}


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
