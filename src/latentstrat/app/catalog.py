"""Read-only discovery of completed LatentStrat workspace artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from latentstrat.app.contracts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStatus,
    FileSignature,
    ProvenanceStatus,
    WorkspaceCatalog,
)

_SEASON_RE = re.compile(r"(?<!\d)((?:20)\d{2})(?!\d)")
_IGNORED_PARTS = {"archive", "__pycache__", ".git"}
_IGNORED_NAMES = {"checkpoint_resume.pt", "resume_checkpoint.pt"}
_IGNORED_SUFFIXES = {".log", ".pid", ".tmp", ".partial"}


def workspace_root(value: str | Path | None = None) -> Path:
    configured = value or os.environ.get("LATENTSTRAT_APP_ROOT") or Path.cwd()
    root = Path(configured).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"LatentStrat workspace root does not exist: {root}")
    return root


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def ensure_workspace_path(path: str | Path, root: str | Path) -> Path:
    resolved = Path(path).resolve()
    root_path = Path(root).resolve()
    if not _inside_root(resolved, root_path):
        raise ValueError(f"Path is outside the configured LatentStrat workspace: {resolved}")
    return resolved


def _ignored(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part.lower() in _IGNORED_PARTS for part in relative.parts):
        return True
    if path.name.lower() in _IGNORED_NAMES:
        return True
    return path.suffix.lower() in _IGNORED_SUFFIXES or path.name.startswith("~")


def _season_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = _SEASON_RE.search(part)
        if match:
            return int(match.group(1))
    return None


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _nearest_manifest(path: Path, artifact_root: Path) -> tuple[Path | None, dict[str, Any]]:
    current = path.parent
    while _inside_root(current, artifact_root):
        candidate = current / "manifest.json"
        if candidate.exists():
            return candidate, _read_manifest(candidate)
        if current == artifact_root:
            break
        current = current.parent
    return None, {}


def _manifest_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    promotion = payload.get("promotion") if isinstance(payload.get("promotion"), dict) else {}
    return {
        "workflow": payload.get("workflow"),
        "schema_version": payload.get("schema_version"),
        "promotion_eligible": promotion.get("eligible"),
        "note": promotion.get("note"),
        "artifact_version": payload.get("artifact_version"),
        "created_at_utc": payload.get("created_at_utc"),
        "git_commit": payload.get("git_commit"),
        "sources": payload.get("sources") or {},
    }


def _kind_for(path: Path) -> ArtifactKind | None:
    lower_parts = tuple(part.lower() for part in path.parts)
    name = path.name.lower()
    if path.suffix.lower() == ".db" and "scouting" in lower_parts:
        return ArtifactKind.SCOUTING_DATABASE
    if name == "embeddings.parquet" and "match-breakdown" in lower_parts:
        return ArtifactKind.MATCH_BREAKDOWN_EMBEDDINGS
    if path.suffix.lower() == ".parquet" and "features" in lower_parts:
        return ArtifactKind.FEATURE_TABLE
    if name in {"walk_forward_predictions.parquet", "paired_predictions.parquet"}:
        return ArtifactKind.PREDICTIONS
    if name.startswith("statbotics_predictions_") and name.endswith(".parquet"):
        return ArtifactKind.PREDICTIONS
    if name in {"checkpoint.pt", "pretrained_prior_2026.pt", "prior_checkpoint.pt"}:
        if name.startswith("prior_") or "prior" in lower_parts or (
            "pretraining" in lower_parts and "prior" in path.as_posix()
        ):
            return ArtifactKind.PRIOR_CHECKPOINT
        return ArtifactKind.SEASON_CHECKPOINT
    if path.suffix.lower() == ".pt" and "checkpoints" in lower_parts:
        return ArtifactKind.SEASON_CHECKPOINT
    if name in {"v5_checkpoint.pt", "full_season_checkpoint.pt"}:
        return ArtifactKind.SEASON_CHECKPOINT
    if name in {"model.pt", "eval_model.pt"} and "match-breakdown" in lower_parts:
        return ArtifactKind.MATCH_BREAKDOWN_CHECKPOINT
    if name == "manifest.json":
        return ArtifactKind.MANIFEST
    if name in {
        "metrics.csv",
        "walk_forward_metrics.csv",
        "feature_common_metrics.csv",
        "feature_continuous_metrics.csv",
        "feature_binary_metrics.csv",
        "calibration.csv",
        "feature_calibration.csv",
        "paired_bootstrap.csv",
        "coverage.json",
        "feature_availability_slices.csv",
        "feature_award_metrics.csv",
        "feature_endgame_metrics.csv",
        "feature_history.csv",
        "feature_training_diagnostics.json",
        "prior_training_history.csv",
        "training_history.csv",
        "auxiliary_metrics.csv",
        "parameter_utilization.csv",
        "initialization_selection.csv",
        "optimization_grid.csv",
        "development_confirmation.csv",
        "development_selection.json",
        "hierarchical_bootstrap.csv",
        "interaction_decisions.csv",
        "external_comparison.csv",
        "external_verdict.json",
        "validation_report.json",
        "walk_forward_history.csv",
    }:
        return ArtifactKind.EVALUATION
    return None


def _candidate_paths(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (root / "data", root / "artifacts"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not _ignored(path, root) and _kind_for(path) is not None:
                candidates.append(path)
    return sorted(set(candidates))


def _manifest_declaration(
    path: Path, root: Path, manifests: list[tuple[Path, dict[str, Any]]]
) -> tuple[Path | None, dict[str, Any] | None]:
    resolved = path.resolve()
    for manifest_path, payload in manifests:
        files = payload.get("files")
        if isinstance(files, dict):
            for relative, value in files.items():
                if not isinstance(value, dict):
                    continue
                candidate = (manifest_path.parent / str(relative)).resolve()
                if candidate == resolved:
                    return manifest_path, value
        sources = payload.get("sources")
        if isinstance(sources, dict):
            for value in sources.values():
                if not isinstance(value, dict) or not value.get("path"):
                    continue
                candidate = (root / str(value["path"])).resolve()
                if candidate == resolved:
                    return manifest_path, value
    return None, None


def verify_artifact_provenance(artifact: ArtifactRef) -> ProvenanceStatus:
    expected = artifact.metadata.get("expected_sha256")
    if not isinstance(expected, str) or not expected:
        return ProvenanceStatus.MISSING
    digest = hashlib.sha256()
    before = FileSignature.capture(artifact.path)
    with artifact.path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = FileSignature.capture(artifact.path)
    if before != after:
        raise RuntimeError(f"Artifact changed while it was being verified: {artifact.path}")
    return (
        ProvenanceStatus.VERIFIED
        if digest.hexdigest().lower() == expected.lower()
        else ProvenanceStatus.MISMATCH
    )


def discover_workspace(root: str | Path | None = None) -> WorkspaceCatalog:
    resolved_root = workspace_root(root)
    artifact_root = resolved_root / "artifacts"
    candidates = _candidate_paths(resolved_root)
    manifests = [
        (path, _read_manifest(path))
        for path in candidates
        if path.name.lower() == "manifest.json"
    ]
    artifacts: list[ArtifactRef] = []
    for path in candidates:
        kind = _kind_for(path)
        if kind is None:
            continue
        manifest_path, manifest = _nearest_manifest(path, artifact_root)
        declaration_manifest, declaration = _manifest_declaration(
            path, resolved_root, manifests
        )
        if not manifest and declaration_manifest is not None:
            manifest_path = declaration_manifest
            manifest = _read_manifest(declaration_manifest)
        manifest_values = _manifest_metadata(manifest)
        promotion = manifest_values["promotion_eligible"]
        status = ArtifactStatus.READY if manifest else ArtifactStatus.UNVERIFIED
        if path.stat().st_size == 0:
            status = ArtifactStatus.INCOMPLETE
        relative = path.relative_to(resolved_root).as_posix()
        artifacts.append(
            ArtifactRef(
                artifact_id=f"{kind.value}:{relative}",
                kind=kind,
                path=path.resolve(),
                signature=FileSignature.capture(path),
                status=status,
                provenance_status=(
                    ProvenanceStatus.DECLARED
                    if declaration and declaration.get("sha256")
                    else ProvenanceStatus.MISSING
                ),
                season=_season_from_path(path),
                schema_version=manifest_values["schema_version"],
                promotion_eligible=promotion if isinstance(promotion, bool) else None,
                workflow=(
                    str(manifest_values["workflow"])
                    if manifest_values["workflow"] is not None
                    else None
                ),
                note=(str(manifest_values["note"]) if manifest_values["note"] else None),
                metadata={
                    **{key: value for key, value in manifest_values.items() if value is not None},
                    "manifest_path": str(manifest_path) if manifest_path else None,
                    "expected_sha256": declaration.get("sha256") if declaration else None,
                    "expected_bytes": declaration.get("bytes") if declaration else None,
                },
            )
        )
    return WorkspaceCatalog(root=resolved_root, artifacts=tuple(artifacts))


__all__ = [
    "discover_workspace",
    "ensure_workspace_path",
    "verify_artifact_provenance",
    "workspace_root",
]
