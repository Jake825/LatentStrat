from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from frc.providers.statbotics_provider import StatboticsProvider
from latentstrat import cli
from latentstrat.cli import app
from latentstrat.paths import (
    EMBEDDING_DB_PATH,
    MATCH_BREAKDOWN_CORPUS_PATH,
    OPENAI_EMBEDDING_CACHE_PATH,
    SCOUTING_DB_PATH,
    STATBOTICS_CACHE_PATH,
    TBA_CACHE_BASE,
    event_features_path,
    prior_features_path,
    season_features_path,
)


def test_canonical_storage_paths_are_stable():
    assert TBA_CACHE_BASE == Path("data/cache/tba")
    assert STATBOTICS_CACHE_PATH == Path("data/cache/statbotics.sqlite")
    assert OPENAI_EMBEDDING_CACHE_PATH == Path("data/cache/openai_embeddings.sqlite")
    assert SCOUTING_DB_PATH == Path("data/scouting/scouting.db")
    assert EMBEDDING_DB_PATH == Path("data/embeddings/latentstrat_embeddings.sqlite")
    assert MATCH_BREAKDOWN_CORPUS_PATH == Path("data/world_model/match_breakdowns.sqlite")
    assert prior_features_path(2026) == Path("data/features/prior/prior_features_2026.parquet")
    assert season_features_path(2026) == Path("data/features/season/features_2026.parquet")
    assert event_features_path("2026ilch") == Path("data/features/event/features_2026ilch.parquet")


class _FakeStatboticsClient:
    calls = 0

    def get_team_years(self, **filters):
        self.__class__.calls += 1
        return [{"team": filters.get("team", 1), "year": filters.get("year", 2025)}]


def test_statbotics_sqlite_cache_persists_across_provider_instances(tmp_path, monkeypatch):
    _FakeStatboticsClient.calls = 0
    monkeypatch.setitem(
        sys.modules,
        "statbotics",
        SimpleNamespace(Statbotics=lambda: _FakeStatboticsClient()),
    )
    cache_path = tmp_path / "cache" / "statbotics.sqlite"

    first = StatboticsProvider(cache_path=cache_path)
    assert first.get_team_years(year=2025) == [{"team": 1, "year": 2025}]
    assert first.get_team_years(year=2025) == [{"team": 1, "year": 2025}]

    second = StatboticsProvider(cache_path=cache_path)
    assert second.get_team_years(year=2025) == [{"team": 1, "year": 2025}]
    assert second.get_team_years(year=2024) == [{"team": 1, "year": 2024}]

    assert _FakeStatboticsClient.calls == 2
    with sqlite3.connect(cache_path) as connection:
        row_count = connection.execute("select count(*) from statbotics_cache").fetchone()[0]
    assert row_count == 2


def test_statbotics_legacy_cache_dir_argument_writes_sqlite_inside_directory(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "statbotics",
        SimpleNamespace(Statbotics=lambda: _FakeStatboticsClient()),
    )
    legacy_dir = tmp_path / "statbotics_offline_cache"

    provider = StatboticsProvider(cache_dir=legacy_dir)
    provider.get_team_years(year=2025)

    assert provider.cache_path == legacy_dir / "statbotics.sqlite"
    assert provider.cache_path.exists()


def test_build_features_default_outputs_use_canonical_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    written_paths = []
    table = pd.DataFrame(
        {
            "event_key": ["2026ilch"],
            "event_week": [1],
            "raw_event_week": [0],
        }
    )
    monkeypatch.setattr(cli, "_provider", lambda: object())
    monkeypatch.setattr(cli, "build_event_feature_table", lambda *args, **kwargs: table)
    monkeypatch.setattr(cli, "build_season_feature_table", lambda *args, **kwargs: table)
    monkeypatch.setattr(
        cli,
        "write_feature_table",
        lambda frame, path: written_paths.append(path) or path,
    )
    runner = CliRunner()

    event_result = runner.invoke(
        app,
        ["build-features", "--event-key", "2026ilch", "--no-scouting"],
    )
    season_result = runner.invoke(app, ["build-features", "--season", "2026", "--no-scouting"])

    assert event_result.exit_code == 0, event_result.output
    assert season_result.exit_code == 0, season_result.output
    assert written_paths == [event_features_path("2026ilch"), season_features_path(2026)]


def test_build_prior_features_default_output_and_cache_use_canonical_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_build(provider, target_season, opts, **kwargs):
        captured["cache_path"] = opts.cache_path
        table = pd.DataFrame({"norm_epa_source_years": ["2022,2023,2024,2025"]})
        table.attrs["embedding_stats"] = {}
        return table

    monkeypatch.setattr(cli, "_provider", lambda: object())
    monkeypatch.setattr(cli, "_statbotics_provider", lambda: object())
    monkeypatch.setattr(cli, "build_prior_feature_table", fake_build)
    monkeypatch.setattr(cli, "write_prior_feature_table", lambda frame, path: path)

    result = CliRunner().invoke(app, ["build-prior-features", "--target-season", "2026"])

    assert result.exit_code == 0, result.output
    assert str(prior_features_path(2026)) in result.output
    assert captured["cache_path"] == str(OPENAI_EMBEDDING_CACHE_PATH)


def test_clear_cache_removes_new_and_legacy_caches_but_keeps_features(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in (
        Path("data/cache/tba.sqlite"),
        Path("data/cache/statbotics.sqlite"),
        Path("data/cache/openai_embeddings.sqlite"),
        Path("data/prior_cache/openai_embeddings.sqlite"),
        Path("tba_cache.sqlite"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cache", encoding="utf-8")
    Path("statbotics_offline_cache").mkdir()
    feature = Path("data/features/season/features_2026.parquet")
    feature.parent.mkdir(parents=True, exist_ok=True)
    feature.write_text("feature", encoding="utf-8")
    corpus = Path("data/world_model/match_breakdowns.sqlite")
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text("durable corpus", encoding="utf-8")

    result = CliRunner().invoke(app, ["clear-cache"])

    assert result.exit_code == 0, result.output
    assert not Path("data/cache/tba.sqlite").exists()
    assert not Path("data/cache/statbotics.sqlite").exists()
    assert not Path("data/cache/openai_embeddings.sqlite").exists()
    assert not Path("data/prior_cache/openai_embeddings.sqlite").exists()
    assert not Path("tba_cache.sqlite").exists()
    assert not Path("statbotics_offline_cache").exists()
    assert feature.exists()
    assert corpus.exists()


def test_organize_local_outputs_dry_run_and_apply(tmp_path):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return
    root = tmp_path
    source = root / "data" / "prior_cache" / "openai_embeddings.sqlite"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("cache", encoding="utf-8")
    season = root / "data" / "features_2026.parquet"
    season.write_text("feature", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "organize_local_outputs.ps1"

    dry_run = subprocess.run(
        [powershell, "-ExecutionPolicy", "Bypass", "-File", str(script), "-Root", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "[dry-run]" in dry_run.stdout
    assert source.exists()

    applied = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Root",
            str(root),
            "-Apply",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert applied.returncode == 0, applied.stderr
    assert not source.exists()
    assert (root / "data" / "cache" / "openai_embeddings.sqlite").exists()
    assert not season.exists()
    assert (root / "data" / "features" / "season" / "features_2026.parquet").exists()
