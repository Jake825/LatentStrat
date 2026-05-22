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
from latentstrat.config import LatentStratOptions, PriorOpts, default_options
from latentstrat.data import (
    apply_target_stats,
    build_season_match_table,
    fit_target_stats,
    make_split,
    make_team_index_map,
)
from latentstrat.evaluation import evaluate_model
from latentstrat.experiments import build_evidence_packet
from latentstrat.embedding_store import consolidate_event_checkpoint
from latentstrat.features import (
    build_event_feature_table,
    build_season_feature_table,
    load_v5_checkpoint_model,
    train_feature_file,
    write_feature_table,
)
from latentstrat.inspection import inspect_embeddings
from latentstrat.pretrain_features import (
    build_prior_feature_table,
    read_team_keys_from_feature_file,
    write_prior_feature_table,
)
from latentstrat.prior_inspection import inspect_prior_checkpoint, write_prior_inspection_artifacts
from latentstrat.pretrain_loop import train_prior_file
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
    venue_mode: Annotated[
        bool,
        typer.Option(
            "--venue-mode/--standard-mode",
            help="Freeze the V5 trunk and train only event-delta embeddings.",
        ),
    ] = False,
    event_key: Annotated[
        str | None, typer.Option("--event-key", help="Event key to isolate for venue mode.")
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Optional V5 checkpoint to resume/fine-tune."),
    ] = None,
    prior_checkpoint: Annotated[
        Path | None,
        typer.Option("--prior-checkpoint", help="Optional V5.5 team-key prior checkpoint."),
    ] = None,
) -> None:
    """Train LatentStrat from a local Parquet feature file."""
    updates = {}
    if epochs is not None:
        updates["epochs"] = epochs
    if mini_batch_size is not None:
        updates["mini_batch_size"] = mini_batch_size
    opts = default_options().model_copy(update=updates)
    initial_model = load_v5_checkpoint_model(checkpoint) if checkpoint is not None else None
    result = train_feature_file(
        input_path,
        opts,
        output_dir=output,
        venue_mode=venue_mode,
        venue_event_key=event_key,
        initial_model=initial_model,
        prior_checkpoint=prior_checkpoint,
    )
    typer.echo(
        f"Feature training complete: rows={len(result.prepared)}, "
        f"teams={len(result.team_index_map)}, final_loss={result.diagnostics.final_loss:.4f}"
    )


@app.command("build-prior-features")
def build_prior_features(
    target_season: Annotated[
        int, typer.Option("--target-season", help="Target season to quarantine from priors.")
    ] = 2026,
    teams_from: Annotated[
        Path | None,
        typer.Option(
            "--teams-from",
            help="Target-season feature Parquet used only for team identity.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output V5.5 prior feature Parquet path."),
    ] = None,
    cache_path: Annotated[
        Path,
        typer.Option("--cache-path", help="SQLite cache for OpenAI text embeddings."),
    ] = Path("data/prior_cache/openai_embeddings.sqlite"),
) -> None:
    """Build quarantined text-only prior features for V5.5."""
    source = teams_from or Path(f"data/features_{target_season}.parquet")
    output_path = output or Path(f"data/prior_features_{target_season}.parquet")
    opts = PriorOpts(cache_path=str(cache_path))
    provider = _provider()
    team_keys = read_team_keys_from_feature_file(source)
    table = build_prior_feature_table(team_keys, provider, target_season, opts)
    path = write_prior_feature_table(table, output_path)
    stats = table.attrs.get("embedding_stats", {})
    typer.echo(
        f"Prior feature table written: {path} teams={len(table)} "
        f"cache_hits={stats.get('cache_hits', 0)} "
        f"cache_misses={stats.get('cache_misses', 0)} "
        f"api_batches={stats.get('api_batches', 0)} "
        f"retries={stats.get('retry_count', 0)}"
    )


@app.command("train-prior")
def train_prior(
    features: Annotated[Path, typer.Option("--features", help="V5.5 prior feature Parquet.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output V5.5 prior checkpoint."),
    ] = None,
    epochs: Annotated[
        int | None, typer.Option("--epochs", help="Override prior pretraining epochs.")
    ] = None,
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", help="Override prior pretraining batch size.")
    ] = None,
) -> None:
    """Train the V5.5 text-prior denoising autoencoder."""
    updates = {}
    if epochs is not None:
        updates["epochs"] = epochs
    if batch_size is not None:
        updates["batch_size"] = batch_size
    opts = PriorOpts(**updates)
    output_path = output or Path("data") / f"pretrained_prior_{pd.Timestamp.utcnow().year}.pt"
    result = train_prior_file(features, output_path, opts)
    typer.echo(
        f"Prior checkpoint written: {output_path} teams={len(result.team_vectors)} "
        f"final_loss={result.history['train_loss'].iloc[-1]:.6f}"
    )


@app.command("inspect-prior")
def inspect_prior(
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", help="V5.5 prior checkpoint from train-prior.")
    ],
    features: Annotated[
        Path, typer.Option("--features", help="V5.5 prior feature Parquet.")
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for prior latent-space diagnostics."),
    ] = Path("artifacts/prior_2026"),
    top_k: Annotated[
        int, typer.Option("--top-k", help="Nearest neighbors per team.")
    ] = 10,
) -> None:
    """Write V5.5 prior latent-space tables and static PNG visuals."""
    inspection = inspect_prior_checkpoint(checkpoint, features, top_k=top_k)
    out = write_prior_inspection_artifacts(inspection, output)
    typer.echo(
        f"Prior inspection complete: {out} teams={len(inspection.latent_table)} "
        f"neighbors={len(inspection.nearest_neighbors)}"
    )


@app.command("consolidate-event")
def consolidate_event(
    checkpoint: Annotated[Path, typer.Argument(help="V5 checkpoint produced by train-features.")],
    event_key: Annotated[str, typer.Option("--event-key", help="Event key to consolidate.")],
    embedding_db: Annotated[
        Path,
        typer.Option("--embedding-db", help="SQLite store for durable base embeddings."),
    ] = Path("data/latentstrat_embeddings.sqlite"),
    delta_weeks: Annotated[
        float,
        typer.Option("--delta-weeks", help="Calendar weeks since this team's previous event."),
    ] = 1.0,
) -> None:
    """Fold event delta embeddings into durable base embeddings."""
    output = consolidate_event_checkpoint(
        checkpoint, event_key=event_key, embedding_db=embedding_db, delta_weeks=delta_weeks
    )
    typer.echo(f"Consolidated embeddings written to {embedding_db}; checkpoint written: {output}")


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
