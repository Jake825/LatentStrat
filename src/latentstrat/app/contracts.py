"""Typed, UI-independent workbench contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ArtifactKind(StrEnum):
    FEATURE_TABLE = "feature-table"
    SCOUTING_DATABASE = "scouting-database"
    SEASON_CHECKPOINT = "season-checkpoint"
    PRIOR_CHECKPOINT = "prior-checkpoint"
    MATCH_BREAKDOWN_CHECKPOINT = "match-breakdown-checkpoint"
    MATCH_BREAKDOWN_EMBEDDINGS = "match-breakdown-embeddings"
    PREDICTIONS = "predictions"
    EVALUATION = "evaluation"
    MANIFEST = "manifest"


class ArtifactStatus(StrEnum):
    READY = "ready"
    INCOMPLETE = "incomplete"
    INCOMPATIBLE = "incompatible"
    UNVERIFIED = "unverified"


class ProvenanceStatus(StrEnum):
    MISSING = "missing"
    DECLARED = "declared"
    VERIFIED = "verified"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class FileSignature:
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: str | Path) -> FileSignature:
        stat = Path(path).stat()
        return cls(size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: ArtifactKind
    path: Path
    signature: FileSignature
    status: ArtifactStatus = ArtifactStatus.UNVERIFIED
    provenance_status: ProvenanceStatus = ProvenanceStatus.MISSING
    season: int | None = None
    schema_version: int | None = None
    promotion_eligible: bool | None = None
    workflow: str | None = None
    note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        season = f" {self.season}" if self.season is not None else ""
        return f"{self.path.parent.name or self.path.name}{season}"


@dataclass(frozen=True)
class EmbeddingSpaceRef:
    space_id: str
    label: str
    basis_id: str
    artifact: ArtifactRef
    vector_columns: tuple[str, ...] = ()
    description: str = ""

    def assert_same_basis(self, other: EmbeddingSpaceRef) -> None:
        if self.basis_id != other.basis_id:
            raise ValueError(
                f"Embedding spaces {self.label!r} and {other.label!r} use different bases."
            )


@dataclass(frozen=True)
class WorkspaceCatalog:
    root: Path
    artifacts: tuple[ArtifactRef, ...]

    def of_kind(self, *kinds: ArtifactKind) -> tuple[ArtifactRef, ...]:
        selected = set(kinds)
        return tuple(item for item in self.artifacts if item.kind in selected)

    def by_id(self, artifact_id: str) -> ArtifactRef:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise KeyError(artifact_id)
