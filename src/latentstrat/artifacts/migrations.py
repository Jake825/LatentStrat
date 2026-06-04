"""One-release readers for generated files relocated by the V6.1 migration."""

from __future__ import annotations

import json
from pathlib import Path

LEGACY_PATH_RELOCATIONS = {
    "data/world_model/match_breakdowns.sqlite": "data/pretraining/match-breakdown/corpus.sqlite",
    (
        "data/features/world_model/match_breakdown_alliances_2015_2026.parquet"
    ): "data/features/pretraining/match-breakdown/match_breakdown_alliances_2015_2026.parquet",
}


def resolve_migrated_path(
    path: str | Path,
    *,
    archive_index: str | Path = "artifacts/archive/index.json",
) -> Path:
    """Resolve a generated legacy path without mutating historical artifact metadata."""

    candidate = Path(path)
    if candidate.exists():
        return candidate
    normalized = candidate.as_posix()
    mapped = LEGACY_PATH_RELOCATIONS.get(normalized)
    if mapped is not None and Path(mapped).exists():
        return Path(mapped)
    index_path = Path(archive_index)
    if index_path.exists():
        records = json.loads(index_path.read_text(encoding="utf-8-sig"))
        for record in records:
            if Path(record["source"]).as_posix() != normalized:
                continue
            destination = Path(record["destination"])
            if destination.exists():
                return destination
    return candidate
