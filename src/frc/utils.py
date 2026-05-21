"""Small FRC utility functions."""

from __future__ import annotations

from pathlib import Path


def normalize_team_key(team_identifier: int | str) -> str:
    text = str(team_identifier).strip()
    return text if text.startswith("frc") else f"frc{text}"


def generate_match_key(event_key: str, match_type: str, match_number: int) -> str:
    return f"{event_key}_{match_type}{match_number}"


def get_artifact_dir() -> Path:
    path = Path("artifacts")
    path.mkdir(exist_ok=True)
    return path
