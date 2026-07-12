"""Shared manifest envelope for LatentStrat-generated artifacts."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from latentstrat.artifacts.hashes import sha256_file
from latentstrat.training_runtime import runtime_provenance

MANIFEST_SCHEMA_VERSION = 1


def _git_head_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _file_records(root: Path, manifest_path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        records[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def write_artifact_manifest(
    output_dir: str | Path,
    *,
    workflow: str,
    artifact_version: str,
    resolved_config: dict[str, Any] | None = None,
    source_files: dict[str, str | Path] | None = None,
    promotion_eligible: bool = False,
    promotion_note: str | None = None,
    migration_provenance: dict[str, Any] | None = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    sources = {}
    for name, source in sorted((source_files or {}).items()):
        path = Path(source)
        sources[name] = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workflow": workflow,
        "artifact_version": artifact_version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_head_commit(),
        "resolved_config": resolved_config or {},
        "sources": sources,
        "files": _file_records(root, manifest_path),
        "promotion": {
            "eligible": bool(promotion_eligible),
            "note": promotion_note,
        },
        "migration_provenance": migration_provenance,
        "runtime_provenance": runtime_provenance(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
