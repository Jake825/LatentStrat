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
from frc.providers.statbotics_provider import StatboticsProvider
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
    build_feature_sidecars,
    build_event_feature_table,
    build_season_feature_table,
    load_v5_checkpoint_model,
    train_feature_file,
    write_feature_table,
)
from latentstrat.inspection import inspect_embeddings
from latentstrat.pretrain_features import (
    build_prior_feature_table,
    write_prior_feature_table,
)
from latentstrat.prior_inspection import inspect_prior_checkpoint, write_prior_inspection_artifacts
from latentstrat.pretrain_loop import train_prior_file
from latentstrat.prior_grid import run_prior_grid
from latentstrat.training import train_model

app = typer.Typer(help="LatentStrat Python CLI")


def _provider(cache_name: str = "tba_cache") -> TbaProvider:
    return TbaProvider(cache_name=cache_name)


def _statbotics_provider(cache_dir: str = "statbotics_offline_cache") -> StatboticsProvider:
    return StatboticsProvider(cache_dir=cache_dir)


def _parse_int_csv(value: str, *, option_name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be a comma-separated integer list.") from exc
    if not parsed:
        raise typer.BadParameter(f"{option_name} must contain at least one integer.")
    if any(item <= 0 for item in parsed):
        raise typer.BadParameter(f"{option_name} values must be positive.")
    return parsed


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
    sidecar_output_dir: Annotated[
        Path | None,
        typer.Option(
            "--sidecar-output-dir",
            help="Optional directory for V5.7 rankings/selections/playoffs sidecar Parquet.",
        ),
    ] = None,
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
        sidecar_event_keys = [event_key]
    else:
        table = build_season_feature_table(
            season,
            opts,
            provider=provider,
            event_limit=event_limit,
            scouting_db_path=scouting_path,
        )
        output_path = output or Path(f"data/features_{season}.parquet")
        sidecar_event_keys = sorted(table["event_key"].astype(str).unique())
    path = write_feature_table(table, output_path)
    if sidecar_output_dir is not None:
        sidecars = build_feature_sidecars(
            sidecar_event_keys, provider, season, output_dir=sidecar_output_dir
        )
        typer.echo(
            "Sidecars written: "
            + ", ".join(f"{name}={len(frame)}" for name, frame in sidecars.items())
        )
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
        typer.Option("--prior-checkpoint", help="Optional V5.6 embedding-table prior checkpoint."),
    ] = None,
    rankings_sidecar: Annotated[
        Path | None,
        typer.Option("--rankings-sidecar", help="Optional V5.7 rankings sidecar Parquet."),
    ] = None,
    selections_sidecar: Annotated[
        Path | None,
        typer.Option("--selections-sidecar", help="Optional V5.7 selections sidecar Parquet."),
    ] = None,
    playoffs_sidecar: Annotated[
        Path | None,
        typer.Option("--playoffs-sidecar", help="Optional V5.7 playoff sidecar Parquet."),
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
    sidecar_tables = {}
    for name, path in (
        ("rankings", rankings_sidecar),
        ("selections", selections_sidecar),
        ("playoffs", playoffs_sidecar),
    ):
        if path is not None:
            sidecar_tables[name] = pd.read_parquet(path, engine="pyarrow")
    result = train_feature_file(
        input_path,
        opts,
        output_dir=output,
        venue_mode=venue_mode,
        venue_event_key=event_key,
        initial_model=initial_model,
        prior_checkpoint=prior_checkpoint,
        sidecar_tables=sidecar_tables or None,
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
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output V5.6 prior feature Parquet path."),
    ] = None,
    cache_path: Annotated[
        Path,
        typer.Option("--cache-path", help="SQLite cache for OpenAI text embeddings."),
    ] = Path("data/prior_cache/openai_embeddings.sqlite"),
    max_team_number: Annotated[
        int,
        typer.Option("--max-team-number", help="Largest team number to include."),
    ] = 12_500,
    epa_source_year: Annotated[
        int | None,
        typer.Option(
            "--epa-source-year",
            help="Completed Statbotics season to use for EPA targets.",
        ),
    ] = None,
) -> None:
    """Build the full V5.6 transductive prior feature table."""
    output_path = output or Path(f"data/prior_features_{target_season}.parquet")
    opts = PriorOpts(
        cache_path=str(cache_path),
        max_team_number=max_team_number,
        epa_source_year=epa_source_year,
    )
    provider = _provider()
    table = build_prior_feature_table(
        provider,
        target_season,
        opts,
        statbotics_provider=_statbotics_provider(),
        epa_source_year=epa_source_year,
    )
    path = write_prior_feature_table(table, output_path)
    stats = table.attrs.get("embedding_stats", {})
    typer.echo(
        f"Prior feature table written: {path} rows={len(table)} "
        f"cache_hits={stats.get('cache_hits', 0)} "
        f"cache_misses={stats.get('cache_misses', 0)} "
        f"api_batches={stats.get('api_batches', 0)} "
        f"retries={stats.get('retry_count', 0)} "
        f"epa_source_year={int(table['epa_source_year'].iloc[0])}"
    )


@app.command("train-prior")
def train_prior(
    features: Annotated[Path, typer.Option("--features", help="V5.6 prior feature Parquet.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output V5.6 checkpoint path or directory."),
    ] = None,
    epochs: Annotated[
        int | None, typer.Option("--epochs", help="Override prior pretraining epochs.")
    ] = None,
    batch_size: Annotated[
        int | None, typer.Option("--batch-size", help="Override prior pretraining batch size.")
    ] = None,
    latent_dim: Annotated[
        int | None, typer.Option("--latent-dim", help="Override prior latent dimension.")
    ] = None,
    max_team_number: Annotated[
        int | None,
        typer.Option("--max-team-number", help="Largest team number in the distiller table."),
    ] = None,
    tensorboard: Annotated[
        bool,
        typer.Option(
            "--tensorboard/--no-tensorboard",
            help="Write local TensorBoard event files for prior training.",
        ),
    ] = True,
    tensorboard_logdir: Annotated[
        Path,
        typer.Option("--tensorboard-logdir", help="TensorBoard root log directory."),
    ] = Path("runs"),
    tensorboard_run_name: Annotated[
        str | None,
        typer.Option("--tensorboard-run-name", help="Optional TensorBoard run name."),
    ] = None,
) -> None:
    """Train the V5.6 team-number prior distiller."""
    updates = {}
    if epochs is not None:
        updates["epochs"] = epochs
    if batch_size is not None:
        updates["batch_size"] = batch_size
    if latent_dim is not None:
        updates["latent_dim"] = latent_dim
    if max_team_number is not None:
        updates["max_team_number"] = max_team_number
    opts = PriorOpts(**updates)
    output_path = output or Path("data") / f"pretrained_prior_{pd.Timestamp.utcnow().year}.pt"
    result = train_prior_file(
        features,
        output_path,
        opts,
        tensorboard_logdir=tensorboard_logdir if tensorboard else None,
        tensorboard_run_name=tensorboard_run_name,
    )
    if result.tensorboard_logdir is not None:
        typer.echo(f"TensorBoard active: tensorboard --logdir={tensorboard_logdir}")
    team_count = result.embedding_table.shape[0] - 1
    typer.echo(
        f"Prior checkpoint written: {result.output_path} teams={team_count} "
        f"final_loss={result.history['train_loss'].iloc[-1]:.6f}"
    )


@app.command("inspect-prior")
def inspect_prior(
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", help="V5.6 prior checkpoint from train-prior.")
    ],
    features: Annotated[
        Path | None, typer.Option("--features", help="Optional V5.6 prior feature Parquet.")
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", help="Directory for prior latent-space diagnostics."),
    ] = Path("artifacts/prior_2026"),
    top_k: Annotated[
        int, typer.Option("--top-k", help="Nearest neighbors per team.")
    ] = 10,
) -> None:
    """Write V5.6 prior latent-space tables and static PNG visuals."""
    inspection = inspect_prior_checkpoint(checkpoint, features, top_k=top_k)
    out = write_prior_inspection_artifacts(inspection, output)
    typer.echo(
        f"Prior inspection complete: {out} teams={len(inspection.latent_table)} "
        f"neighbors={len(inspection.nearest_neighbors)}"
    )


@app.command("run-prior-grid")
def run_prior_grid_command(
    features: Annotated[Path, typer.Option("--features", help="V5.6 prior feature Parquet.")],
    output: Annotated[
        Path, typer.Option("--output", help="Directory for prior-grid artifacts.")
    ] = Path("artifacts/prior_grid_2026"),
    epochs: Annotated[int, typer.Option("--epochs", help="Epochs per grid run.")] = 500,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Training batch size for each grid run.")
    ] = 256,
    learning_rate: Annotated[
        float, typer.Option("--learning-rate", help="AdamW learning rate.")
    ] = 1e-3,
    latent_dims: Annotated[
        str,
        typer.Option("--latent-dims", help="Comma-separated latent dimensions."),
    ] = "2,4,8,16,32,64,128,256",
    validation_fraction: Annotated[
        float, typer.Option("--validation-fraction", help="Validation holdout fraction.")
    ] = 0.2,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 2026,
    device: Annotated[str, typer.Option("--device", help='Torch device, or "auto".')] = "auto",
) -> None:
    """Run the V5.6 prior bottleneck grid experiment."""
    result = run_prior_grid(
        features,
        output,
        latent_dims=_parse_int_csv(latent_dims, option_name="latent-dims"),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        seed=seed,
        device=device,
    )
    ok = int((result.results["status"] == "ok").sum()) if not result.results.empty else 0
    typer.echo(
        f"Prior grid complete: {result.output_dir} runs={len(result.results)} "
        f"ok={ok} failures={len(result.failures)}"
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
