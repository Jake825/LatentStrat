"""Command-line entry points for LatentStrat."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from frc.datastore import FRCDataStore
from frc.importers import TBAImporter
from frc.providers.tba_provider import TbaProvider
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
