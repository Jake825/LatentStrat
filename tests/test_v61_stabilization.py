from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import torch
from pydantic import BaseModel
from typer.testing import CliRunner

from frc.providers.payloads import as_plain_payload
from latentstrat.artifacts.checkpoints import load_season_checkpoint_model
from latentstrat.artifacts.hashes import sha256_file
from latentstrat.artifacts.manifests import write_artifact_manifest
from latentstrat.artifacts.migrations import resolve_migrated_path
from latentstrat.cli import app
from latentstrat.config import PriorOpts, default_options
from latentstrat.experimental.frozen_targets import WorldModelOptions
from latentstrat.features import load_award_catalog
from latentstrat.model import ForwardOutput, init_model, optimizer_parameter_groups


class _PayloadModel(BaseModel):
    value: int


def test_provider_boundary_adapter_uses_plain_json_containers():
    payload = SimpleNamespace(
        key="2026test",
        nested={"model": _PayloadModel(value=3)},
        items=(SimpleNamespace(score=0),),
        _private="ignore",
    )

    assert as_plain_payload(payload) == {
        "key": "2026test",
        "nested": {"model": {"value": 3}},
        "items": [{"score": 0}],
    }


def test_forward_output_is_nested_and_optimizer_groups_cover_parameters_once():
    opts = default_options()
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    red = torch.tensor([[1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[4, 5, 6]], dtype=torch.long)

    output = model(red, blue)
    assert isinstance(output, ForwardOutput)
    assert output.predictions["continuous"] is output.cont_z
    assert output.representations.z_match is output.z_match
    assert output.masks.red_missing_mask is output.red_missing_mask

    groups = optimizer_parameter_groups(model, opts)
    grouped = [parameter for group in groups for parameter in group["params"]]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}


def test_schema6_checkpoint_factory_preserves_registered_state_keys(tmp_path):
    opts = default_options()
    model = init_model(8, opts.latent_dim, 4, 1, opts)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "checkpoint_schema_version": 6,
            "model_state_dict": model.state_dict(),
            "options": opts.model_dump(mode="json"),
            "world_model": WorldModelOptions(enabled=False).model_dump(mode="json"),
        },
        path,
    )

    loaded = load_season_checkpoint_model(path)
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_manifest_contract_hashes_sources_and_excludes_itself(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "artifact"
    source.write_text("source", encoding="utf-8")
    output.mkdir()
    (output / "result.csv").write_text("value\n1\n", encoding="utf-8")

    manifest_path = write_artifact_manifest(
        output,
        workflow="test",
        artifact_version="v1",
        resolved_config={"seed": 2026},
        source_files={"source": source},
        promotion_note="fixture",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["workflow"] == "test"
    assert payload["resolved_config"] == {"seed": 2026}
    assert payload["sources"]["source"]["sha256"] == sha256_file(source)
    assert payload["files"]["result.csv"]["sha256"] == sha256_file(output / "result.csv")
    assert "manifest.json" not in payload["files"]


def test_migrated_path_reader_uses_archive_index_without_mutating_metadata(tmp_path):
    source = tmp_path / "legacy" / "feature.parquet"
    destination = tmp_path / "canonical" / "feature.parquet"
    archive_index = tmp_path / "archive" / "index.json"
    destination.parent.mkdir(parents=True)
    archive_index.parent.mkdir(parents=True)
    destination.write_text("feature", encoding="utf-8")
    archive_index.write_text(
        json.dumps([{"source": str(source), "destination": str(destination)}]),
        encoding="utf-8",
    )

    assert resolve_migrated_path(source, archive_index=archive_index) == destination


def test_tracked_catalog_and_season_defaults_are_active():
    opts = default_options(2026)
    catalog = load_award_catalog()

    assert opts.endgame_class_order == ("None", "Level1", "Level2", "Level3")
    assert any(award["id"] == "impact" and award["tba_types"] == [0] for award in catalog)
    assert "epa_source_year" in PriorOpts.model_fields
    assert "epa_rookie_baseline_z" not in PriorOpts.model_fields


def test_grouped_cli_and_flat_tombstone_are_exposed():
    runner = CliRunner()

    grouped = runner.invoke(app, ["pretrain", "match-breakdown", "train", "--help"])
    tombstone = runner.invoke(app, ["full-season-offline"])

    assert grouped.exit_code == 0, grouped.output
    assert "--config" in grouped.output
    assert tombstone.exit_code == 2
    assert "season build-features" in tombstone.output
    assert "season train" in tombstone.output


def test_source_tree_has_no_removed_world_model_namespace_imports():
    removed_namespace = "latentstrat." + "world_model"
    for path in Path("src").rglob("*.py"):
        assert removed_namespace not in path.read_text(encoding="utf-8")


def test_migration_refuses_destination_collisions(tmp_path):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return
    source = tmp_path / "data" / "features_2026.parquet"
    destination = tmp_path / "data" / "features" / "season" / "features_2026.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source", encoding="utf-8")
    destination.write_text("different", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "organize_local_outputs.ps1"

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Root",
            str(tmp_path),
            "-Apply",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing collision" in result.stderr
    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "different"


def test_active_markdown_links_resolve():
    markdown_link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    roots = [Path("README.md"), *(path for path in Path("docs").glob("*.md"))]

    for path in roots:
        for target in markdown_link.findall(path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            clean_target = unquote(target.split("#", 1)[0])
            if not clean_target:
                continue
            assert (path.parent / clean_target).exists(), f"{path}: missing {target}"
