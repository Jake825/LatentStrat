"""Provider-boundary conversion helpers for third-party API objects."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def as_plain_payload(value: Any) -> Any:
    """Convert tbapy-style objects into JSON-compatible Python containers."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=False)
    if isinstance(value, dict):
        return {str(key): as_plain_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_plain_payload(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): as_plain_payload(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value
