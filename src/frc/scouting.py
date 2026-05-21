"""Transactional scouting database schema."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel, create_engine


class TeamScouting(SQLModel, table=True):
    __tablename__ = "team_scouting"

    team_key: str = Field(primary_key=True, description="TBA team key, e.g. frc254.")
    drive_base: str | None = Field(default=None, description="Swerve, Tank, Mecanum, etc.")
    robot_weight_lbs: float | None = None
    frame_dimensions_inches: str | None = None
    programming_language: str | None = None


class EventScouting(SQLModel, table=True):
    __tablename__ = "event_scouting"

    event_key: str = Field(primary_key=True, description="TBA event key, e.g. 2026ilch.")
    carpet_condition: str | None = Field(default=None, description="New, Worn, Taped, etc.")
    venue_notes: str | None = None


class MatchScouting(SQLModel, table=True):
    __tablename__ = "match_scouting"

    match_key: str = Field(primary_key=True, description="TBA match key, e.g. 2026ilch_qm1.")
    event_key: str = Field(foreign_key="event_scouting.event_key", index=True)
    field_fault_occurred: bool = False
    audience_delay_minutes: int = 0
    referee_strictness_rating: int | None = Field(default=None, ge=1, le=5)


class TeamEventScouting(SQLModel, table=True):
    __tablename__ = "team_event_scouting"

    team_key: str = Field(primary_key=True, foreign_key="team_scouting.team_key")
    event_key: str = Field(primary_key=True, foreign_key="event_scouting.event_key")
    passed_inspection: bool = False
    is_functional: bool = True
    major_breakdown_notes: str | None = None


class MatchAllianceScouting(SQLModel, table=True):
    __tablename__ = "match_alliance_scouting"

    match_key: str = Field(primary_key=True, foreign_key="match_scouting.match_key")
    alliance_color: str = Field(primary_key=True, description="red or blue")
    coopertition_agreed: bool = False
    coordinated_auto_run: bool = False
    strategic_meltdown: bool = False


class TeamMatchScouting(SQLModel, table=True):
    __tablename__ = "team_match_scouting"

    team_key: str = Field(primary_key=True, foreign_key="team_scouting.team_key")
    match_key: str = Field(primary_key=True, foreign_key="match_scouting.match_key")
    alliance_color: str = Field(index=True, description="red or blue")
    auto_pieces_scored: int = 0
    teleop_pieces_scored: int = 0
    endgame_status: str | None = Field(default=None, description="Deep, Shallow, Parked, None")
    driver_ability_rating: int = Field(default=3, ge=1, le=5)
    played_defense: bool = False
    defended_by_team: str | None = Field(default=None, description="TBA key of opponent defender")
    scouter_name: str | None = None


SCOUTING_MODELS = (
    TeamScouting,
    EventScouting,
    MatchScouting,
    TeamEventScouting,
    MatchAllianceScouting,
    TeamMatchScouting,
)


def create_scouting_engine(path: str | Path = "data/scouting.db") -> Engine:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def create_db_and_tables(path: str | Path = "data/scouting.db") -> Path:
    db_path = Path(path)
    engine = create_scouting_engine(db_path)
    SQLModel.metadata.create_all(engine)
    return db_path
