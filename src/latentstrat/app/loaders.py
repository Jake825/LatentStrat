"""Stable, read-only loaders for workbench data and checkpoints."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from latentstrat.app.catalog import ensure_workspace_path
from latentstrat.app.contracts import ArtifactRef, FileSignature

T = TypeVar("T")

SCOUTING_TABLES: dict[str, dict[str, str]] = {
    "team_scouting": {"grain": "team", "timing": "pre-match candidate", "prefix": "pit_"},
    "event_scouting": {
        "grain": "event",
        "timing": "event context",
        "prefix": "event_scout_",
    },
    "match_scouting": {
        "grain": "match",
        "timing": "during/post-match unless proven otherwise",
        "prefix": "match_scout_",
    },
    "team_event_scouting": {
        "grain": "team-event",
        "timing": "event state",
        "prefix": "event_scout_",
    },
    "match_alliance_scouting": {
        "grain": "alliance-match",
        "timing": "during/post-match",
        "prefix": "alliance_scout_",
    },
    "team_match_scouting": {
        "grain": "team-match",
        "timing": "during/post-match",
        "prefix": "match_scout_",
    },
}


@dataclass(frozen=True)
class ParquetProfile:
    path: Path
    rows: int
    columns: int
    row_groups: int
    size_bytes: int
    schema: pd.DataFrame


@dataclass(frozen=True)
class CheckpointSummary:
    path: Path
    kind: str
    schema_version: int | None
    season: int | None
    latent_dim: int | None
    parameter_count: int
    options: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ScoutingTableSummary:
    table: str
    rows: int
    columns: int
    grain: str
    timing: str
    merge_prefix: str


def stable_read(path: str | Path, reader: Callable[[Path], T]) -> T:
    resolved = Path(path)
    before = FileSignature.capture(resolved)
    value = reader(resolved)
    after = FileSignature.capture(resolved)
    if before != after:
        raise RuntimeError(f"Artifact changed while it was being read: {resolved}")
    return value


def profile_parquet(path: str | Path) -> ParquetProfile:
    def _read(resolved: Path) -> ParquetProfile:
        parquet = pq.ParquetFile(resolved)
        arrow_schema = parquet.schema_arrow
        schema = pd.DataFrame(
            {
                "column": arrow_schema.names,
                "dtype": [str(field.type) for field in arrow_schema],
                "nullable": [bool(field.nullable) for field in arrow_schema],
            }
        )
        return ParquetProfile(
            path=resolved,
            rows=int(parquet.metadata.num_rows),
            columns=int(parquet.metadata.num_columns),
            row_groups=int(parquet.metadata.num_row_groups),
            size_bytes=int(resolved.stat().st_size),
            schema=schema,
        )

    return stable_read(path, _read)


def load_parquet(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    selected = list(columns) if columns is not None else None
    return stable_read(path, lambda resolved: pd.read_parquet(resolved, columns=selected))


def profile_dataframe(table: pd.DataFrame) -> pd.DataFrame:
    rows = max(len(table), 1)
    return pd.DataFrame(
        {
            "column": table.columns,
            "dtype": [str(dtype) for dtype in table.dtypes],
            "missing": [int(table[column].isna().sum()) for column in table.columns],
            "missing_percent": [
                100.0 * float(table[column].isna().sum()) / rows for column in table
            ],
            "unique": [int(table[column].nunique(dropna=True)) for column in table.columns],
        }
    )


def _sqlite_readonly_uri(path: Path) -> str:
    return f"file:{quote(path.resolve().as_posix(), safe='/:')}?mode=ro"


def _sqlite_connection(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(_sqlite_readonly_uri(Path(path)), uri=True)


def scouting_overview(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)

    def _read(_: Path) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        with _sqlite_connection(resolved) as connection:
            existing = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            for table, semantics in SCOUTING_TABLES.items():
                if table not in existing:
                    continue
                count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                column_count = len(connection.execute(f'PRAGMA table_info("{table}")').fetchall())
                rows.append(
                    {
                        "table": table,
                        "rows": count,
                        "columns": column_count,
                        "grain": semantics["grain"],
                        "timing": semantics["timing"],
                        "merge_prefix": semantics["prefix"],
                    }
                )
        return pd.DataFrame(rows)

    return stable_read(resolved, _read)


def load_scouting_table(path: str | Path, table: str, limit: int = 500) -> pd.DataFrame:
    if table not in SCOUTING_TABLES:
        raise ValueError(f"Unsupported scouting table: {table}")
    resolved = Path(path)

    def _read(_: Path) -> pd.DataFrame:
        with _sqlite_connection(resolved) as connection:
            return pd.read_sql_query(
                f'SELECT * FROM "{table}" LIMIT ?', connection, params=(limit,)
            )

    return stable_read(resolved, _read)


def scouting_feature_trace(columns: Sequence[str]) -> pd.DataFrame:
    patterns = (
        ("event_scout_", "event", "event_scouting", "event context"),
        ("match_scout_", "match", "match_scouting", "during/post-match"),
        ("red_alliance_scout_", "alliance-match", "match_alliance_scouting", "during/post-match"),
        ("blue_alliance_scout_", "alliance-match", "match_alliance_scouting", "during/post-match"),
    )
    rows: list[dict[str, str]] = []
    for column in columns:
        matched = False
        for prefix, grain, source, timing in patterns:
            if column.startswith(prefix):
                rows.append(
                    {
                        "feature_column": column,
                        "source_table": source,
                        "grain": grain,
                        "timing": timing,
                    }
                )
                matched = True
                break
        if matched:
            continue
        match = re.match(
            r"^(red|blue)_team_[123]_(pit|event_scout|match_scout)_", column
        )
        if match:
            category = match.group(2)
            source, grain, timing = {
                "pit": ("team_scouting", "team", "pre-match candidate"),
                "event_scout": ("team_event_scouting", "team-event", "event state"),
                "match_scout": ("team_match_scouting", "team-match", "during/post-match"),
            }[category]
            rows.append(
                {"feature_column": column, "source_table": source, "grain": grain, "timing": timing}
            )
    return pd.DataFrame(rows)


def load_json(path: str | Path) -> dict[str, Any]:
    def _read(resolved: Path) -> dict[str, Any]:
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {resolved}")
        return value

    return stable_read(path, _read)


def load_csv(path: str | Path) -> pd.DataFrame:
    return stable_read(path, pd.read_csv)


def load_checkpoint_payload(
    artifact: ArtifactRef | str | Path, workspace: str | Path
) -> dict[str, Any]:
    path = artifact.path if isinstance(artifact, ArtifactRef) else Path(artifact)
    resolved = ensure_workspace_path(path, workspace)

    def _read(value: Path) -> dict[str, Any]:
        payload = torch.load(value, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint payload is not a mapping: {value}")
        return payload

    return stable_read(resolved, _read)


def _state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for name in ("model_state_dict", "state_dict"):
        state = payload.get(name)
        if isinstance(state, dict):
            return {str(key): value for key, value in state.items() if torch.is_tensor(value)}
    embedding = payload.get("embedding_table")
    return {"embedding_table": embedding} if torch.is_tensor(embedding) else {}


def parameter_table(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, tensor in _state_dict(payload).items():
        value = tensor.detach().cpu()
        numeric = value.float()
        finite = numeric[torch.isfinite(numeric)]
        rows.append(
            {
                "parameter": name,
                "shape": " × ".join(str(item) for item in value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
                "count": int(value.numel()),
                "mean": float(finite.mean()) if finite.numel() else math.nan,
                "std": float(finite.std(unbiased=False)) if finite.numel() else math.nan,
                "min": float(finite.min()) if finite.numel() else math.nan,
                "max": float(finite.max()) if finite.numel() else math.nan,
                "norm": float(torch.linalg.vector_norm(finite)) if finite.numel() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def module_hierarchy(payload: dict[str, Any]) -> pd.DataFrame:
    aggregates: dict[str, dict[str, int]] = {}
    for name, tensor in _state_dict(payload).items():
        parts = name.split(".")[:-1]
        modules = ["<root>"] + [".".join(parts[:depth]) for depth in range(1, len(parts) + 1)]
        for module in modules:
            row = aggregates.setdefault(module, {"tensors": 0, "parameters": 0})
            row["tensors"] += 1
            row["parameters"] += int(tensor.numel())
    frame = pd.DataFrame(
        [
            {
                "module": module,
                "depth": 0 if module == "<root>" else module.count(".") + 1,
                **values,
            }
            for module, values in aggregates.items()
        ],
        columns=["module", "depth", "tensors", "parameters"],
    )
    return frame.sort_values(["depth", "module"], kind="mergesort")


def checkpoint_summary(path: Path, payload: dict[str, Any]) -> CheckpointSummary:
    state = _state_dict(payload)
    options = payload.get("options") or payload.get("prior_opts") or {}
    options = options if isinstance(options, dict) else {}
    if "model_state_dict" in payload:
        kind = "season"
        schema = payload.get("checkpoint_schema_version")
        season = options.get("season")
        base = state.get("Z_base.weight")
        latent_dim = int(base.shape[1]) if base is not None and base.ndim == 2 else None
    elif payload.get("kind") == "v6-lite-match-breakdown-shared-bottleneck":
        kind = "match-breakdown"
        schema = payload.get("schema_version")
        season = None
        latent_dim = int(payload["latent_dim"])
    elif "embedding_table" in payload:
        kind = "prior"
        schema = None
        season = payload.get("target_season")
        embedding = payload.get("embedding_table")
        latent_dim = int(embedding.shape[1]) if torch.is_tensor(embedding) else None
    else:
        kind = "unknown"
        schema = None
        season = None
        latent_dim = None
    metadata = {
        "team_count": len(payload.get("team_base_index_map") or {}),
        "team_event_count": len(payload.get("team_event_index_map") or {}),
        "split_policy": payload.get("split_policy"),
        "prior_checkpoint": payload.get("prior_checkpoint"),
        "frozen_embedding_tables": payload.get("frozen_embedding_tables") or [],
        "sidecar_tables": payload.get("sidecar_tables") or [],
        "provenance": payload.get("provenance"),
    }
    return CheckpointSummary(
        path=path,
        kind=kind,
        schema_version=int(schema) if schema is not None else None,
        season=int(season) if season is not None else None,
        latent_dim=latent_dim,
        parameter_count=sum(int(tensor.numel()) for tensor in state.values()),
        options=options,
        metadata=metadata,
    )


def tensor_values(payload: dict[str, Any], parameter: str, max_values: int = 50_000) -> np.ndarray:
    state = _state_dict(payload)
    if parameter not in state:
        raise KeyError(parameter)
    values = state[parameter].detach().cpu().float().reshape(-1).numpy()
    values = values[np.isfinite(values)]
    if len(values) > max_values:
        indices = np.linspace(0, len(values) - 1, max_values, dtype=int)
        values = values[indices]
    return values


__all__ = [
    "SCOUTING_TABLES",
    "CheckpointSummary",
    "ParquetProfile",
    "checkpoint_summary",
    "load_checkpoint_payload",
    "load_csv",
    "load_json",
    "load_parquet",
    "load_scouting_table",
    "module_hierarchy",
    "parameter_table",
    "profile_dataframe",
    "profile_parquet",
    "scouting_feature_trace",
    "scouting_overview",
    "stable_read",
    "tensor_values",
]
