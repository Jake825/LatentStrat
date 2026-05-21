# ruff: noqa: E402, I001
"""Command-line entry points for LatentStrat."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from latentstrat.secrets import load_environment

load_environment()

import pandas as pd
import typer

from frc.datastore import FRCDataStore
from frc.importers import TBAImporter
from frc.providers.tba_provider import TbaProvider
from frc.scouting import create_db_and_tables
from latentstrat.baselines import fit_baselines
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.data import (
    apply_target_stats,
    build_season_match_table,
    fit_target_stats,
    make_split,
    make_team_index_map,
)
from latentstrat.evaluation import evaluate_model
from latentstrat.experiments import build_evidence_packet
from latentstrat.features import (
    build_event_feature_table,
    build_season_feature_table,
    train_feature_file,
    write_feature_table,
)
from latentstrat.inspection import inspect_embeddings
from latentstrat.training import train_model

app = typer.Typer(help="LatentStrat Python CLI")


def _provider(cache_name: str = "tba_cache") -> TbaProvider:
    return TbaProvider(cache_name=cache_name)


def _load_event_table(
    event_key: str, opts: LatentStratOptions
) -> tuple[pd.DataFrame, dict[str, int]]:
    provider = _provider()
    store = FRCDataStore()
    TBAImporter.import_event(provider, store, event_key)
    metadata = pd.DataFrame([{"key": event_key, "week": 0}])
    table = build_season_match_table(store, opts, event_metadata=metadata)
    return make_team_index_map(table)


@app.command("api-smoke")
def api_smoke() -> None:
    """Check TBA API connectivity."""
    provider = _provider()
    typer.echo(provider.get_status())


@app.command("smoke-test")
def smoke_test(event_key: str | None = None) -> None:
    """Run the single-event smoke training path."""
    opts = default_options()
    if event_key is None:
        event_key = opts.smoke_event_key
    opts = opts.model_copy(update={"mini_batch_size": opts.smoke_mini_batch_size})
    table, team_map = _load_event_table(event_key, opts)
    split = make_split(
        table, opts, policy=opts.split_policy, validation_fraction=opts.validation_fraction
    )
    stats = fit_target_stats(table, split.train_mask, opts)
    prepared = apply_target_stats(table, stats)
    baselines = fit_baselines(prepared, split, opts)
    model, history, diagnostics = train_model(prepared, split, opts)
    report = evaluate_model(model, prepared, split, stats, opts, baselines)
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    prepared.to_csv(out / "smoke_match_table.csv", index=False)
    history.to_csv(out / "smoke_history.csv", index=False)
    report.continuous_metrics.to_csv(out / "smoke_continuous_metrics.csv", index=False)
    typer.echo(
        f"Smoke test complete: rows={len(prepared)}, teams={len(team_map)}, "
        f"initial_loss={diagnostics.initial_loss:.4f}, final_loss={diagnostics.final_loss:.4f}"
    )


@app.command("build-features")
def build_features(
    event_key: Annotated[
        str | None, typer.Option("--event-key", help="Build features for a single event key.")
    ] = None,
    season: Annotated[int, typer.Option("--season", help="Build features for this season.")] = 2026,
    output: Annotated[
        Path | None, typer.Option("--output", help="Output Parquet feature file.")
    ] = None,
    event_limit: Annotated[
        int | None,
        typer.Option("--event-limit", help="Limit season imports for quick validation runs."),
    ] = None,
    scouting_db: Annotated[
        Path, typer.Option("--scouting-db", help="Optional SQLite scouting database.")
    ] = Path("data/scouting.db"),
    include_scouting: Annotated[
        bool,
        typer.Option(
            "--include-scouting/--no-scouting",
            help="Merge scouting rows into the Parquet feature table when available.",
        ),
    ] = True,
) -> None:
    """Build a reusable Parquet feature file from TBA data."""
    opts = default_options().model_copy(update={"season": season})
    provider = _provider()
    scouting_path = scouting_db if include_scouting and scouting_db.exists() else None
    if include_scouting and scouting_path is None:
        typer.echo(f"Scouting DB not found at {scouting_db}; building TBA-only features.")
    if event_key is not None:
        table = build_event_feature_table(
            event_key, opts, provider=provider, scouting_db_path=scouting_path
        )
        output_path = output or Path(f"data/features_{event_key}.parquet")
    else:
        table = build_season_feature_table(
            season,
            opts,
            provider=provider,
            event_limit=event_limit,
            scouting_db_path=scouting_path,
        )
        output_path = output or Path(f"data/features_{season}.parquet")
    path = write_feature_table(table, output_path)
    typer.echo(f"Feature table written: {path} rows={len(table)} columns={len(table.columns)}")


@app.command("init-scouting-db")
def init_scouting_db(
    path: Annotated[
        Path, typer.Option("--path", help="SQLite scouting database path.")
    ] = Path("data/scouting.db"),
) -> None:
    """Create the local SQLite scouting database schema."""
    db_path = create_db_and_tables(path)
    typer.echo(f"Scouting database initialized: {db_path}")


@app.command("train-features")
def train_features(
    input_path: Annotated[Path, typer.Argument(help="Input Parquet feature file.")],
    output: Annotated[
        Path, typer.Option("--output", help="Directory for training artifacts.")
    ] = Path("artifacts/features_run"),
    epochs: Annotated[
        int | None, typer.Option("--epochs", help="Override training epochs.")
    ] = None,
    mini_batch_size: Annotated[
        int | None, typer.Option("--mini-batch-size", help="Override training mini-batch size.")
    ] = None,
) -> None:
    """Train LatentStrat from a local Parquet feature file."""
    updates = {}
    if epochs is not None:
        updates["epochs"] = epochs
    if mini_batch_size is not None:
        updates["mini_batch_size"] = mini_batch_size
    opts = default_options().model_copy(update=updates)
    result = train_feature_file(input_path, opts, output_dir=output)
    typer.echo(
        f"Feature training complete: rows={len(result.prepared)}, "
        f"teams={len(result.team_index_map)}, final_loss={result.diagnostics.final_loss:.4f}"
    )


@app.command("full-season-offline")
def full_season_offline(season: int = 2026, event_limit: int | None = None) -> None:
    """Import a season, train the model, and write core artifacts."""
    opts = default_options().model_copy(update={"season": season})
    provider = _provider()
    store = FRCDataStore()
    events = TBAImporter.import_season(provider, store, season, event_limit=event_limit)
    table = build_season_match_table(store, opts, event_metadata=events)
    table, team_map = make_team_index_map(table)
    split = make_split(
        table,
        opts,
        policy=opts.full_season_split_policy,
        validation_fraction=opts.validation_fraction,
    )
    stats = fit_target_stats(table, split.train_mask, opts)
    prepared = apply_target_stats(table, stats)
    baselines = fit_baselines(prepared, split, opts)
    model, history, diagnostics = train_model(prepared, split, opts)
    report = evaluate_model(model, prepared, split, stats, opts, baselines)
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    prepared.to_csv(out / "full_season_match_table.csv", index=False)
    history.to_csv(out / "full_season_history.csv", index=False)
    report.continuous_metrics.to_csv(out / "full_season_continuous_metrics.csv", index=False)
    typer.echo(
        f"Full-season run complete: rows={len(prepared)}, teams={len(team_map)}, "
        f"final_loss={diagnostics.final_loss:.4f}"
    )


@app.command("inspect-embeddings")
def inspect_embeddings_command(event_key: str | None = None) -> None:
    """Train a small run and write embedding inspection tables."""
    opts = default_options()
    if event_key is None:
        event_key = opts.smoke_event_key
    table, team_map = _load_event_table(event_key, opts)
    split = make_split(table, opts)
    stats = fit_target_stats(table, split.train_mask, opts)
    prepared = apply_target_stats(table, stats)
    baselines = fit_baselines(prepared, split, opts)
    model, _, _ = train_model(prepared, split, opts, verbose=False)
    inspection = inspect_embeddings(model, prepared, team_map, opts, baselines)
    out = Path("artifacts/inspection")
    out.mkdir(parents=True, exist_ok=True)
    inspection.team_embedding_table.to_csv(out / "team_embeddings.csv", index=False)
    inspection.nearest_neighbors.to_csv(out / "nearest_neighbors.csv", index=False)
    inspection.archetype_similarity.to_csv(out / "archetype_similarity.csv", index=False)
    typer.echo(f"Inspection complete: {out}")


@app.command("build-evidence-packet")
def build_evidence_packet_command(event_key: str | None = None) -> None:
    """Train evidence runs and write scoreboard/diagnostic artifacts."""
    opts = default_options()
    if event_key is None:
        event_key = opts.smoke_event_key
    table, team_map = _load_event_table(event_key, opts)
    build_evidence_packet(table, team_map, opts)
    typer.echo("Evidence packet written to artifacts/evidence_packet")


@app.command("clear-cache")
def clear_cache() -> None:
    """Remove Python provider caches."""
    import shutil

    for path in ("tba_cache.sqlite", "statbotics_offline_cache"):
        target = Path(path)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    typer.echo("Caches cleared.")


if __name__ == "__main__":
    app()
