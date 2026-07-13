from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from latentstrat.app.analysis import (
    build_season_model,
    checkpoint_feature_compatibility,
    compute_pca,
    cosine_neighbors,
    diagnose_match,
    season_embedding_frame,
    season_event_trajectory_frame,
)
from latentstrat.app.catalog import discover_workspace, verify_artifact_provenance
from latentstrat.app.contracts import (
    ArtifactKind,
    ArtifactRef,
    ArtifactStatus,
    EmbeddingSpaceRef,
    FileSignature,
    ProvenanceStatus,
)
from latentstrat.app.loaders import (
    checkpoint_summary,
    load_checkpoint_payload,
    load_scouting_table,
    scouting_feature_trace,
    scouting_overview,
    stable_read,
)
from latentstrat.config import default_options
from latentstrat.experimental.frozen_targets import WorldModelOptions
from latentstrat.season.model import init_model


def _season_checkpoint(path: Path) -> dict:
    opts = default_options().model_copy(
        update={"latent_dim": 4, "attention_heads": 1, "set_ffn_dim": 8}
    )
    model = init_model(
        8,
        4,
        len(opts.target_map),
        len(opts.binary_targets),
        opts,
        num_event_teams=8,
        num_endgame_classes=len(opts.endgame_class_order),
        num_awards=len(opts.award_targets),
        world_model_opts=WorldModelOptions(enabled=False),
    )
    payload = {
        "checkpoint_schema_version": 6,
        "model_state_dict": model.state_dict(),
        "options": opts.model_dump(mode="json"),
        "world_model": WorldModelOptions(enabled=False).model_dump(mode="json"),
        "target_stats": {
            "target_names": [item.target_name for item in opts.target_map],
            "mu": [0.0, 0.0, 0.0, 0.0],
            "sigma": [1.0, 1.0, 1.0, 1.0],
            "is_constant": [False, False, False, False],
        },
        "team_base_index_map": {f"frc{number}": number for number in range(1, 7)},
        "team_event_index_map": {f"2026test::frc{number}": number for number in range(1, 7)},
    }
    torch.save(payload, path)
    return payload


def _feature_row() -> dict:
    return {
        "season": 2026,
        "event_key": "2026test",
        "match_key": "2026test_qm1",
        "comp_level": "qm",
        "red_team_1_key": "frc1",
        "red_team_2_key": "frc2",
        "red_team_3_key": "frc3",
        "blue_team_1_key": "frc4",
        "blue_team_2_key": "frc5",
        "blue_team_3_key": "frc6",
        "red_total_score": 100.0,
        "blue_total_score": 90.0,
        "red_auto_pts": 20.0,
        "blue_auto_pts": 18.0,
    }


def test_static_reference_checkpoint_suppresses_event_state_claims():
    opts = default_options().model_copy(
        update={
            "latent_dim": 4,
            "attention_heads": 1,
            "set_ffn_dim": 8,
            "state_model": "static-z-base",
            "match_architecture": "full-match",
        }
    )
    model = init_model(8, 4, len(opts.target_map), len(opts.binary_targets), opts)
    payload = {
        "checkpoint_schema_version": 7,
        "model_state_dict": model.state_dict(),
        "options": opts.model_dump(mode="json"),
        "world_model": WorldModelOptions(enabled=False).model_dump(mode="json"),
        "team_base_index_map": {f"frc{number}": number for number in range(1, 7)},
        "team_event_index_map": {},
    }

    restored = build_season_model(payload)
    frame, columns, basis = season_embedding_frame(payload)
    compatibility = checkpoint_feature_compatibility(payload, pd.DataFrame([_feature_row()]))

    assert not hasattr(restored, "Z_event")
    assert len(frame) == 6
    assert len(columns) == 4
    assert basis == "base"
    assert compatibility.compatible
    with pytest.raises(ValueError, match="no event states"):
        season_event_trajectory_frame(payload)


def _complete_workspace(root: Path) -> None:
    feature_path = root / "data" / "features" / "season" / "features_2026.parquet"
    feature_path.parent.mkdir(parents=True)
    pd.DataFrame([_feature_row()]).to_parquet(feature_path)
    checkpoint = root / "artifacts" / "season" / "run" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    _season_checkpoint(checkpoint)
    scouting = root / "data" / "scouting" / "scouting.db"
    scouting.parent.mkdir(parents=True)
    with sqlite3.connect(scouting) as connection:
        connection.execute(
            "CREATE TABLE team_scouting (team_key TEXT PRIMARY KEY, drive_base TEXT)"
        )
        connection.execute("INSERT INTO team_scouting VALUES ('frc1', 'swerve')")
    evaluation = root / "artifacts" / "evaluation" / "run"
    evaluation.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "scope": "aggregate",
                "model": model,
                "metric": "winner_brier",
                "value": value,
                "count": 1,
            }
            for model, value in (("latentstrat", 0.2), ("statbotics", 0.25))
        ]
    ).to_csv(evaluation / "metrics.csv", index=False)
    pd.DataFrame({"epoch": [1, 2], "train_loss": [2.0, 1.0], "validation_loss": [2.2, 1.2]}).to_csv(
        evaluation / "training_history.csv", index=False
    )
    (evaluation / "manifest.json").write_text(
        json.dumps(
            {
                "workflow": "artifacts.evaluate-predictions",
                "promotion": {"eligible": False, "note": "fixture evidence"},
            }
        ),
        encoding="utf-8",
    )


def test_catalog_discovers_supported_files_and_excludes_archive(tmp_path: Path) -> None:
    feature_dir = tmp_path / "data" / "features" / "season"
    feature_dir.mkdir(parents=True)
    pd.DataFrame({"season": [2026], "match_key": ["2026test_qm1"]}).to_parquet(
        feature_dir / "features_2026.parquet"
    )
    checkpoint_dir = tmp_path / "artifacts" / "season" / "current"
    checkpoint_dir.mkdir(parents=True)
    _season_checkpoint(checkpoint_dir / "checkpoint.pt")
    digest = hashlib.sha256((checkpoint_dir / "checkpoint.pt").read_bytes()).hexdigest()
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "season.train",
                "promotion": {"eligible": False, "note": "development"},
                "files": {
                    "checkpoint.pt": {
                        "bytes": (checkpoint_dir / "checkpoint.pt").stat().st_size,
                        "sha256": digest,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "artifacts" / "archive" / "old"
    archive.mkdir(parents=True)
    _season_checkpoint(archive / "checkpoint.pt")
    (checkpoint_dir / "train.log").write_text("ignored", encoding="utf-8")

    catalog = discover_workspace(tmp_path)

    assert len(catalog.of_kind(ArtifactKind.FEATURE_TABLE)) == 1
    checkpoints = catalog.of_kind(ArtifactKind.SEASON_CHECKPOINT)
    assert len(checkpoints) == 1
    assert checkpoints[0].promotion_eligible is False
    assert checkpoints[0].status == ArtifactStatus.READY
    assert checkpoints[0].provenance_status == ProvenanceStatus.DECLARED
    assert verify_artifact_provenance(checkpoints[0]) == ProvenanceStatus.VERIFIED
    assert all("archive" not in artifact.path.parts for artifact in catalog.artifacts)
    assert all(artifact.path.suffix != ".log" for artifact in catalog.artifacts)


def test_stable_read_detects_concurrent_change(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("before", encoding="utf-8")

    def changing_reader(selected: Path) -> str:
        selected.write_text("after with a different length", encoding="utf-8")
        return "value"

    with pytest.raises(RuntimeError, match="changed while"):
        stable_read(path, changing_reader)


def test_checkpoint_loader_is_workspace_bounded_and_weights_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "artifacts" / "season" / "run" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    expected = _season_checkpoint(checkpoint)
    artifact = ArtifactRef(
        artifact_id="checkpoint",
        kind=ArtifactKind.SEASON_CHECKPOINT,
        path=checkpoint,
        signature=FileSignature.capture(checkpoint),
    )

    loaded = load_checkpoint_payload(artifact, tmp_path)

    assert loaded["checkpoint_schema_version"] == 6
    assert loaded["team_base_index_map"] == expected["team_base_index_map"]
    outside = tmp_path.parent / "outside-checkpoint.pt"
    torch.save({"value": torch.ones(1)}, outside)
    try:
        with pytest.raises(ValueError, match="outside"):
            load_checkpoint_payload(outside, tmp_path)
    finally:
        outside.unlink(missing_ok=True)


def test_checkpoint_summaries_cover_v5_prior_and_match_breakdown(tmp_path: Path) -> None:
    v5_path = tmp_path / "v5_checkpoint.pt"
    v5_payload = _season_checkpoint(v5_path)
    v5_payload.pop("checkpoint_schema_version")
    torch.save(v5_payload, v5_path)
    prior_path = tmp_path / "prior_checkpoint.pt"
    prior_payload = {
        "embedding_table": torch.ones((9, 4)),
        "target_season": 2026,
    }
    torch.save(prior_payload, prior_path)
    breakdown_path = tmp_path / "model.pt"
    breakdown_payload = {
        "kind": "v6-lite-match-breakdown-shared-bottleneck",
        "schema_version": 1,
        "latent_dim": 4,
        "state_dict": {"encoder.weight": torch.ones((4, 3))},
    }
    torch.save(breakdown_payload, breakdown_path)

    assert checkpoint_summary(v5_path, v5_payload).kind == "season"
    assert checkpoint_summary(prior_path, prior_payload).kind == "prior"
    summary = checkpoint_summary(breakdown_path, breakdown_payload)
    assert summary.kind == "match-breakdown"
    assert summary.parameter_count == 12


def test_checkpoint_summary_counts_shared_parameter_alias_once(tmp_path: Path) -> None:
    path = tmp_path / "shared_checkpoint.pt"
    shared = torch.ones((9, 4))
    payload = {
        "checkpoint_schema_version": 8,
        "model_state_dict": {
            "Z_base.weight": shared,
            "team_embedding.weight": shared,
            "head.weight": torch.ones((1, 4)),
        },
        "options": {"season": 2026},
    }

    summary = checkpoint_summary(path, payload)

    assert summary.parameter_count == 40


def test_scouting_is_read_at_native_grain_and_traced_to_features(tmp_path: Path) -> None:
    path = tmp_path / "data" / "scouting" / "scouting.db"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE team_scouting (team_key TEXT PRIMARY KEY, drive_base TEXT)"
        )
        connection.execute("INSERT INTO team_scouting VALUES ('frc1', 'swerve')")
        connection.execute(
            "CREATE TABLE team_match_scouting "
            "(team_key TEXT, match_key TEXT, alliance_color TEXT, played_defense INTEGER)"
        )
        connection.execute(
            "INSERT INTO team_match_scouting VALUES ('frc1', '2026test_qm1', 'red', 1)"
        )

    overview = scouting_overview(path)
    team_match = overview[overview["table"] == "team_match_scouting"].iloc[0]

    assert team_match["grain"] == "team-match"
    assert "post-match" in team_match["timing"]
    assert len(load_scouting_table(path, "team_scouting")) == 1
    trace = scouting_feature_trace(
        [
            "red_team_1_pit_drive_base",
            "red_team_1_match_scout_played_defense",
            "red_total_score",
        ]
    )
    assert set(trace["source_table"]) == {"team_scouting", "team_match_scouting"}


def test_embedding_analysis_is_deterministic_and_basis_safe(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.parquet"
    frame = pd.DataFrame(
        {
            "team_key": ["frc1", "frc2", "frc3", "frc4"],
            "z_000": [1.0, 0.9, -1.0, 0.0],
            "z_001": [0.0, 0.1, 0.0, 0.0],
        }
    )
    frame.to_parquet(path)
    artifact = ArtifactRef(
        artifact_id="embeddings",
        kind=ArtifactKind.MATCH_BREAKDOWN_EMBEDDINGS,
        path=path,
        signature=FileSignature.capture(path),
    )
    space = EmbeddingSpaceRef(
        space_id="one",
        label="One",
        basis_id="basis-one",
        artifact=artifact,
        vector_columns=("z_000", "z_001"),
    )

    first = compute_pca(space, frame)
    second = compute_pca(space, frame)
    np.testing.assert_allclose(
        first.coordinates[["PC1", "PC2"]],
        second.coordinates[["PC1", "PC2"]],
    )
    neighbors = cosine_neighbors(space, frame, 0, limit=2)
    assert neighbors.iloc[0]["team_key"] == "frc2"
    other = EmbeddingSpaceRef(
        space_id="two",
        label="Two",
        basis_id="basis-two",
        artifact=artifact,
        vector_columns=("z_000", "z_001"),
    )
    with pytest.raises(ValueError, match="different bases"):
        space.assert_same_basis(other)


def test_match_diagnostic_routes_unknowns_and_checks_swap(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _season_checkpoint(checkpoint)
    row = pd.Series(
        {
            "event_key": "2026test",
            "match_key": "2026test_qm1",
            "red_team_1_key": "frc1",
            "red_team_2_key": "frc2",
            "red_team_3_key": "frc999",
            "blue_team_1_key": "frc4",
            "blue_team_2_key": "frc5",
            "blue_team_3_key": "frc6",
        }
    )

    diagnostic = diagnose_match(payload, row)
    compatibility = checkpoint_feature_compatibility(payload, pd.DataFrame([row]))
    trajectories, trajectory_columns = season_event_trajectory_frame(payload)

    assert diagnostic.unknown_teams == ("frc999",)
    assert 0.0 <= diagnostic.red_win_probability <= 1.0
    assert diagnostic.swap_probability_gap < 1e-5
    assert len(diagnostic.red_pma_weights) == 3
    assert len(diagnostic.zero_slot) == 6
    assert compatibility.compatible
    assert compatibility.warnings
    assert len(trajectories) == 6
    assert len(trajectory_columns) == 4


def test_streamlit_entrypoint_smoke_renders_empty_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LATENTSTRAT_APP_ROOT", str(tmp_path))
    app_path = Path(__file__).parents[1] / "app" / "app.py"
    test_app = AppTest.from_file(app_path, default_timeout=20).run()

    assert not test_app.exception
    assert streamlit.__version__.startswith("1.")
    rendered = " ".join(markdown.value for markdown in test_app.markdown)
    assert "Workspace" in rendered


@pytest.mark.parametrize(
    "page_name",
    ["Workspace", "Data", "Runs + Evaluation", "Model", "Embeddings", "Match"],
)
def test_each_streamlit_page_handles_empty_workspace(
    page_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LATENTSTRAT_APP_ROOT", str(tmp_path))
    test_app = AppTest.from_string(
        "\n".join(
            (
                "from latentstrat.app.ui import PAGES, configure_page, render_sidebar",
                "configure_page()",
                "render_sidebar()",
                f"PAGES[{page_name!r}]()",
            )
        ),
        default_timeout=20,
    ).run()

    assert not test_app.exception


@pytest.mark.parametrize(
    "page_name",
    ["Workspace", "Data", "Runs + Evaluation", "Model", "Embeddings", "Match"],
)
def test_each_streamlit_page_handles_complete_fixture_workspace(
    page_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    _complete_workspace(tmp_path)
    monkeypatch.setenv("LATENTSTRAT_APP_ROOT", str(tmp_path))
    test_app = AppTest.from_string(
        "\n".join(
            (
                "from latentstrat.app.ui import PAGES, configure_page, render_sidebar",
                "configure_page()",
                "render_sidebar()",
                f"PAGES[{page_name!r}]()",
            )
        ),
        default_timeout=20,
    ).run()

    assert not test_app.exception


def test_match_page_handles_partial_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    feature = tmp_path / "data" / "features" / "season" / "features_2026.parquet"
    feature.parent.mkdir(parents=True)
    pd.DataFrame([_feature_row()]).to_parquet(feature)
    monkeypatch.setenv("LATENTSTRAT_APP_ROOT", str(tmp_path))
    test_app = AppTest.from_string(
        "\n".join(
            (
                "from latentstrat.app.ui import configure_page, render_match",
                "configure_page()",
                "render_match()",
            )
        ),
        default_timeout=20,
    ).run()

    assert not test_app.exception


def test_app_sources_do_not_import_mutating_runtime_surfaces() -> None:
    app_root = Path(__file__).parents[1] / "src" / "latentstrat" / "app"
    forbidden = (
        "latentstrat.providers",
        "latentstrat.season.train",
        "latentstrat.pretraining.train",
        "statbotics",
        "tbapy",
        "torch.optim",
    )
    imported: list[str] = []
    for path in app_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
    assert not [name for name in imported if name.startswith(forbidden)]
