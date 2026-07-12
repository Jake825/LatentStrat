"""Read-only contracts for the LatentStrat research workbench."""

from latentstrat.app.catalog import discover_workspace, workspace_root
from latentstrat.app.contracts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStatus,
    EmbeddingSpaceRef,
    FileSignature,
    ProvenanceStatus,
    WorkspaceCatalog,
)

__all__ = [
    "ArtifactKind",
    "ArtifactRef",
    "ArtifactStatus",
    "EmbeddingSpaceRef",
    "FileSignature",
    "ProvenanceStatus",
    "WorkspaceCatalog",
    "discover_workspace",
    "workspace_root",
]
