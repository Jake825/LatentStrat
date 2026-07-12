"""Season-specific typed schemas and raw match-breakdown vectorization."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from latentstrat.pretraining.match_breakdown.parse import AllianceBreakdownRow

FIELD_KINDS = frozenset({"numeric", "boolean", "categorical"})
GROUPS = ("total_score", "fouls", "auto", "teleop", "endgame", "bonus", "misc")


@dataclass(frozen=True)
class EncodedField:
    name: str
    group: str
    kind: str
    source_path: str
    category: str | None = None


@dataclass(frozen=True)
class SeasonSchema:
    season: int
    encoded_fields: tuple[EncodedField, ...]
    source_kinds: dict[str, str]
    categories: dict[str, tuple[str, ...]]
    unsupported_paths: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.encoded_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "width": self.width,
            "encoded_fields": [asdict(field) for field in self.encoded_fields],
            "source_kinds": dict(sorted(self.source_kinds.items())),
            "categories": {path: list(values) for path, values in sorted(self.categories.items())},
            "unsupported_paths": list(self.unsupported_paths),
        }


def classify_group(path: str) -> str:
    """Assign a stable reconstruction group from a season-specific TBA leaf path."""

    normalized = path.lower().replace("_", "")
    if normalized in {"score", "total", "totalpoints"} or normalized.endswith(".totalpoints"):
        return "total_score"
    if any(token in normalized for token in ("foul", "penalt", "adjustpoint")):
        return "fouls"
    if "auto" in normalized:
        return "auto"
    if any(token in normalized for token in ("teleop", "shift", "transition")):
        return "teleop"
    if any(
        token in normalized
        for token in ("endgame", "stage", "climb", "hang", "tower", "park", "trap", "barge")
    ):
        return "endgame"
    if any(
        token in normalized
        for token in ("achieved", "rankingpoint", "bonus", "coop", "melody", "ensemble", "rp")
    ):
        return "bonus"
    if "totalpoint" in normalized:
        return "total_score"
    return "misc"


def _value_kind(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return "numeric"
    if isinstance(value, str):
        return "categorical"
    return None


def discover_season_schema(rows: Iterable[AllianceBreakdownRow], season: int) -> SeasonSchema:
    """Build one deterministic typed vector schema for a single FRC season."""

    source_kinds: dict[str, str] = {}
    categories: dict[str, set[str]] = {}
    unsupported: set[str] = set()
    for row in rows:
        if row.season != season:
            continue
        for path, value in sorted(row.flat_breakdown.items()):
            kind = _value_kind(value)
            if kind is None:
                if value is not None:
                    unsupported.add(path)
                continue
            previous = source_kinds.setdefault(path, kind)
            if previous != kind:
                raise ValueError(
                    f"Season {season} breakdown field {path!r} has conflicting "
                    f"types: {previous!r} and {kind!r}."
                )
            if kind == "categorical":
                categories.setdefault(path, set()).add(str(value))
    fields = []
    for path, kind in sorted(source_kinds.items()):
        group = classify_group(path)
        if kind == "categorical":
            for category in sorted(categories[path]):
                fields.append(
                    EncodedField(
                        name=f"{path}={category}",
                        group=group,
                        kind=kind,
                        source_path=path,
                        category=category,
                    )
                )
        else:
            fields.append(EncodedField(name=path, group=group, kind=kind, source_path=path))
    if not fields:
        raise ValueError(f"Season {season} has no supported match-breakdown fields.")
    return SeasonSchema(
        season=season,
        encoded_fields=tuple(fields),
        source_kinds=source_kinds,
        categories={path: tuple(sorted(values)) for path, values in categories.items()},
        unsupported_paths=tuple(sorted(unsupported)),
    )


def vectorize_flat_row(
    flat: dict[str, Any], encoded_fields: tuple[EncodedField, ...] | list[EncodedField]
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw values and observed masks without applying normalization."""

    values = np.zeros(len(encoded_fields), dtype=np.float32)
    mask = np.zeros(len(encoded_fields), dtype=np.float32)
    for index, field in enumerate(encoded_fields):
        raw = flat.get(field.source_path)
        if raw is None:
            continue
        if field.kind == "numeric":
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"Expected numeric value for {field.source_path!r}.")
            if not math.isfinite(float(raw)):
                continue
            values[index] = float(raw)
            mask[index] = 1.0
        elif field.kind == "boolean":
            if not isinstance(raw, bool):
                raise ValueError(f"Expected boolean value for {field.source_path!r}.")
            values[index] = float(raw)
            mask[index] = 1.0
        elif field.kind == "categorical":
            if not isinstance(raw, str):
                raise ValueError(f"Expected categorical value for {field.source_path!r}.")
            values[index] = float(raw == field.category)
            mask[index] = 1.0
        else:
            raise ValueError(f"Unsupported encoded field kind: {field.kind!r}.")
    return values, mask


def union_schema_audit(schemas: dict[int, SeasonSchema]) -> dict[str, Any]:
    """Report cross-season field presence without constructing a union training tensor."""

    all_paths = sorted({path for schema in schemas.values() for path in schema.source_kinds})
    return {
        "training_uses_union_padding": False,
        "seasons": sorted(schemas),
        "season_widths": {str(season): schema.width for season, schema in sorted(schemas.items())},
        "source_path_presence": {
            path: [
                season for season, schema in sorted(schemas.items()) if path in schema.source_kinds
            ]
            for path in all_paths
        },
        "misc_fields": {
            str(season): [
                path for path in sorted(schema.source_kinds) if classify_group(path) == "misc"
            ]
            for season, schema in sorted(schemas.items())
        },
    }
