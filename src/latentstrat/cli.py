# ruff: noqa: E402, I001
"""Command-line entry points for LatentStrat."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import subprocess
import time
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
from latentstrat.baseline_manifest import write_baseline_manifest
from latentstrat.baselines import fit_baselines
from latentstrat.config import LatentStratOptions, PriorOpts, default_options
from latentstrat.season.data import (
    apply_target_stats,
    build_season_match_table,
    fit_target_stats,
    make_split,
    make_team_index_map,
)
from latentstrat.season.evaluate import evaluate_model
from latentstrat.dev.diagnostics import build_evidence_packet, inspect_embeddings
from latentstrat.experimental.venue import consolidate_event_checkpoint
from latentstrat.season.features import (
    build_feature_sidecars,
    build_event_feature_table,
    build_season_feature_table,
    event_week_table_from_feature_table,
    load_season_checkpoint_model,
    train_feature_file,
    write_feature_table,
)
from latentstrat.paths import (
    CACHE_ROOT,
    EMBEDDING_DB_PATH,
    EVIDENCE_ARTIFACT_ROOT,
    EXPERIMENTAL_FROZEN_TARGET_ARTIFACT_ROOT,
    INSPECTION_ARTIFACT_ROOT,
    MATCH_BREAKDOWN_CORPUS_PATH,
    OPENAI_EMBEDDING_CACHE_PATH,
    PRIOR_ARTIFACT_ROOT,
    PRIOR_GRID_ARTIFACT_ROOT,
    SCOUTING_DB_PATH,
    SEASON_ARTIFACT_ROOT,
    SMOKE_ARTIFACT_ROOT,
    STATBOTICS_CACHE_PATH,
    TBA_CACHE_BASE,
    WALK_FORWARD_ARTIFACT_ROOT,
    event_features_path,
    match_breakdown_artifact_dir,
    match_breakdown_features_path,
    prior_features_path,
    season_features_path,
)
from latentstrat.pretraining.prior import (
    build_prior_feature_table,
    inspect_prior_checkpoint,
    run_prior_grid,
    train_prior_file,
    write_prior_feature_table,
    write_prior_inspection_artifacts,
)
from latentstrat.season.train import create_tensorboard_writer, train_model
from latentstrat.season.walk_forward import run_walk_forward_validation
from latentstrat.experimental.frozen_targets import (
    build_world_model_bundle,
    load_world_model_options,
    paired_bootstrap_noninferiority,
)
from latentstrat.pretraining.match_breakdown import (
    MatchBreakdownInspectionOptions,
    load_match_breakdown_training_options,
    sync_match_breakdowns,
    train_match_breakdown_encoder,
    write_match_breakdown_inspection,
)

app = typer.Typer(help="LatentStrat Python CLI")
pretrain_app = typer.Typer(help="Offline representation pretraining workflows.")
prior_app = typer.Typer(help="Day Zero prior pretraining.")
match_breakdown_app = typer.Typer(help="Historical match-breakdown pretraining.")
season_app = typer.Typer(help="Supported supervised season workflow.")
artifact_app = typer.Typer(help="Artifact manifests and comparisons.")
scouting_app = typer.Typer(help="Scouting database commands.")
dev_app = typer.Typer(help="Developer utilities.")
diagnostics_app = typer.Typer(help="Exploratory developer diagnostics.")
experimental_app = typer.Typer(help="Explicitly experimental workflows.")
frozen_targets_app = typer.Typer(help="Experimental frozen-target workflows.")
venue_app = typer.Typer(help="Experimental venue-mode workflow.")

app.add_typer(pretrain_app, name="pretrain")
pretrain_app.add_typer(prior_app, name="prior")
pretrain_app.add_typer(match_breakdown_app, name="match-breakdown")
app.add_typer(season_app, name="season")
app.add_typer(artifact_app, name="artifacts")
app.add_typer(scouting_app, name="scouting")
app.add_typer(dev_app, name="dev")
dev_app.add_typer(diagnostics_app, name="diagnostics")
app.add_typer(experimental_app, name="experimental")
experimental_app.add_typer(frozen_targets_app, name="frozen-targets")
experimental_app.add_typer(venue_app, name="venue")

FLAT_ALIAS_REPLACEMENTS = {
    "api-smoke": "dev api-smoke",
    "smoke-test": "dev smoke-test",
    "build-features": "season build-features",
    "init-scouting-db": "scouting init",
    "train-features": "season train",
    "validate-walk-forward": "season validate",
    "build-world-model": "experimental frozen-targets build",
    "sync-match-breakdowns": "pretrain match-breakdown sync",
    "train-match-breakdown-encoder": "pretrain match-breakdown train",
    "inspect-match-breakdown-encoder": "pretrain match-breakdown inspect",
    "compare-world-model": "artifacts compare-predictions",
    "build-prior-features": "pretrain prior build",
    "train-prior": "pretrain prior train",
    "inspect-prior": "pretrain prior inspect",
    "run-prior-grid": "pretrain prior grid",
    "consolidate-event": "experimental venue consolidate",
    "write-baseline-manifest": "artifacts baseline-manifest",
    "full-season-offline": "season build-features followed by season train",
    "inspect-embeddings": "dev diagnostics embeddings",
    "build-evidence-packet": "dev diagnostics evidence",
    "clear-cache": "dev clear-cache",
}


@app.callback()
def root_callback(ctx: typer.Context) -> None:
    """Route top-level commands while retaining V6.1 flat aliases."""

    replacement = FLAT_ALIAS_REPLACEMENTS.get(ctx.invoked_subcommand or "")
    if replacement is not None:
        typer.echo(
            f"Deprecated V6.1 alias: use `latentstrat {replacement}`. "
            "Flat aliases will be removed in V6.2.",
            err=True,
        )


DEFAULT_MATCH_BREAKDOWN_ARTIFACT_DIR = match_breakdown_artifact_dir(2015, 2026)


def _provider(cache_name: str | Path = TBA_CACHE_BASE) -> TbaProvider:
    return TbaProvider(cache_name=str(cache_name))


def _statbotics_provider(cache_path: str | Path = STATBOTICS_CACHE_PATH) -> StatboticsProvider:
    return StatboticsProvider(cache_path=cache_path)


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
    out = SMOKE_ARTIFACT_ROOT
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
    ] = SCOUTING_DB_PATH,
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
    event_metadata_output: Annotated[
        Path | None,
        typer.Option("--event-metadata-output", help="Optional event metadata Parquet output."),
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
        output_path = output or event_features_path(event_key)
        sidecar_event_keys = [event_key]
    else:
        table = build_season_feature_table(
            season,
            opts,
            provider=provider,
            event_limit=event_limit,
            scouting_db_path=scouting_path,
        )
        output_path = output or season_features_path(season)
        sidecar_event_keys = sorted(table["event_key"].astype(str).unique())
    path = write_feature_table(table, output_path)
    if event_metadata_output is not None:
        if event_key is not None:
            events = [provider.get_event(event_key)]
        else:
            events = [
                event
                for event in provider.get_events_by_year(season, simple=True)
                if event.event_type != 99
            ]
        event_metadata_output.parent.mkdir(parents=True, exist_ok=True)
        event_rows = []
        for event in events:
            row = event.model_dump()
            end_date = pd.to_datetime(row.get("end_date"), errors="coerce")
            row["event_completed"] = bool(pd.notna(end_date) and end_date.date() < date.today())
            event_rows.append(row)
        pd.DataFrame(event_rows).to_parquet(
            event_metadata_output, engine="pyarrow", index=False
        )
        typer.echo(f"Event metadata written: {event_metadata_output} rows={len(events)}")
    if sidecar_output_dir is not None:
        sidecars = build_feature_sidecars(
            sidecar_event_keys,
            provider,
            season,
            output_dir=sidecar_output_dir,
            event_weeks=event_week_table_from_feature_table(table),
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
    ] = SCOUTING_DB_PATH,
) -> None:
    """Create the local SQLite scouting database schema."""
    db_path = create_db_and_tables(path)
    typer.echo(f"Scouting database initialized: {db_path}")


@app.command("train-features")
def train_features(
    input_path: Annotated[Path, typer.Argument(help="Input Parquet feature file.")],
    output: Annotated[
        Path, typer.Option("--output", help="Directory for training artifacts.")
    ] = SEASON_ARTIFACT_ROOT / "features_run",
    epochs: Annotated[
        int | None, typer.Option("--epochs", help="Override training epochs.")
    ] = None,
    mini_batch_size: Annotated[
        int | None, typer.Option("--mini-batch-size", help="Override training mini-batch size.")
    ] = None,
    learning_rate: Annotated[
        float | None, typer.Option("--learning-rate", help="Override AdamW learning rate.")
    ] = None,
    split_policy: Annotated[
        str | None,
        typer.Option(
            "--split-policy",
            help=(
                "Override validation split policy: chronological-holdout, week-held-out, "
                "event-held-out, or stratified-event-comp."
            ),
        ),
    ] = None,
    early_stopping: Annotated[
        bool,
        typer.Option(
            "--early-stopping/--no-early-stopping",
            help="Stop training after validation patience is exhausted.",
        ),
    ] = True,
    restore_best: Annotated[
        bool,
        typer.Option(
            "--restore-best/--save-final",
            help="Restore the best validation weights before writing artifacts.",
        ),
    ] = True,
    lr_eta_min: Annotated[
        float | None,
        typer.Option("--lr-eta-min", help="Minimum learning rate for cosine annealing."),
    ] = None,
    loss_log_var_min: Annotated[
        float | None,
        typer.Option("--loss-log-var-min", help="Minimum homoscedastic log variance."),
    ] = None,
    loss_log_var_max: Annotated[
        float | None,
        typer.Option("--loss-log-var-max", help="Maximum homoscedastic log variance."),
    ] = None,
    venue_mode: Annotated[
        bool,
        typer.Option(
            "--venue-mode/--standard-mode",
            help="Freeze the V5 trunk and train only event-delta embeddings.",
        ),
    ] = False,
    freeze_team_embeddings: Annotated[
        bool,
        typer.Option(
            "--freeze-team-embeddings/--train-team-embeddings",
            help="Freeze Z_base and Z_event while training the Set Transformer and heads.",
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
    world_model_bundle: Annotated[
        Path | None,
        typer.Option(
            "--frozen-target-bundle",
            "--world-model-bundle",
            help="Optional experimental frozen-target bundle.",
        ),
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
    tensorboard: Annotated[
        bool,
        typer.Option(
            "--tensorboard/--no-tensorboard",
            help="Write local TensorBoard event files for feature training.",
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
    """Train LatentStrat from a local Parquet feature file."""
    updates = {}
    if epochs is not None:
        updates["epochs"] = epochs
    if mini_batch_size is not None:
        updates["mini_batch_size"] = mini_batch_size
    if learning_rate is not None:
        updates["learning_rate"] = learning_rate
    if split_policy is not None:
        updates["split_policy"] = split_policy
    updates["use_early_stopping"] = early_stopping
    updates["restore_best_validation_model"] = restore_best
    if lr_eta_min is not None:
        updates["lr_eta_min"] = lr_eta_min
    if loss_log_var_min is not None:
        updates["loss_log_var_min"] = loss_log_var_min
    if loss_log_var_max is not None:
        updates["loss_log_var_max"] = loss_log_var_max
    if freeze_team_embeddings and venue_mode:
        typer.echo("Error: --freeze-team-embeddings cannot be used with --venue-mode.", err=True)
        raise typer.Exit(code=2)
    if freeze_team_embeddings and checkpoint is None and prior_checkpoint is None:
        typer.echo(
            "Error: --freeze-team-embeddings requires --prior-checkpoint or --checkpoint.",
            err=True,
        )
        raise typer.Exit(code=2)
    opts = default_options().model_copy(update=updates)
    initial_model = load_season_checkpoint_model(checkpoint) if checkpoint is not None else None
    sidecar_tables = {}
    for name, path in (
        ("rankings", rankings_sidecar),
        ("selections", selections_sidecar),
        ("playoffs", playoffs_sidecar),
    ):
        if path is not None:
            sidecar_tables[name] = pd.read_parquet(path, engine="pyarrow")
    writer = None
    run_logdir = None
    if tensorboard:
        run_name = tensorboard_run_name or f"v57_features_{opts.season}_{int(time.time())}"
        run_logdir = tensorboard_logdir / run_name
        writer = create_tensorboard_writer(str(run_logdir))
        writer.add_text(
            "Run/Context",
            "\n".join(
                (
                    f"- input_path: {input_path}",
                    f"- output: {output}",
                    f"- epochs: {opts.epochs}",
                    f"- mini_batch_size: {opts.mini_batch_size}",
                    f"- learning_rate: {opts.learning_rate}",
                    f"- split_policy: {opts.split_policy}",
                    f"- lr_eta_min: {opts.lr_eta_min}",
                    f"- use_early_stopping: {opts.use_early_stopping}",
                    f"- restore_best_validation_model: {opts.restore_best_validation_model}",
                    f"- loss_log_var_bounds: [{opts.loss_log_var_min}, {opts.loss_log_var_max}]",
                    f"- freeze_team_embeddings: {freeze_team_embeddings}",
                    f"- prior_checkpoint: {prior_checkpoint}",
                    f"- world_model_bundle: {world_model_bundle}",
                    f"- rankings_sidecar: {rankings_sidecar}",
                    f"- selections_sidecar: {selections_sidecar}",
                    f"- playoffs_sidecar: {playoffs_sidecar}",
                )
            ),
            0,
        )
        typer.echo(f"TensorBoard active: tensorboard --logdir={tensorboard_logdir}")
    try:
        result = train_feature_file(
            input_path,
            opts,
            output_dir=output,
            venue_mode=venue_mode,
            venue_event_key=event_key,
            initial_model=initial_model,
            prior_checkpoint=prior_checkpoint,
            freeze_team_embeddings=freeze_team_embeddings,
            sidecar_tables=sidecar_tables or None,
            tensorboard_writer=writer,
            tensorboard_logdir=run_logdir,
            world_model_bundle=world_model_bundle,
        )
    finally:
        if writer is not None:
            writer.close()
    typer.echo(
        f"Feature training complete: rows={len(result.prepared)}, "
        f"teams={len(result.team_index_map)}, final_loss={result.diagnostics.final_loss:.4f}"
    )


@app.command("validate-walk-forward")
def validate_walk_forward(
    features: Annotated[Path, typer.Option("--features", help="V5.8 feature Parquet.")],
    prior_checkpoint: Annotated[
        Path, typer.Option("--prior-checkpoint", help="V5.6.4 prior checkpoint.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Directory for walk-forward artifacts.")
    ] = WALK_FORWARD_ARTIFACT_ROOT / "v58_walk_forward",
    epochs: Annotated[int, typer.Option("--epochs", help="Training epochs per fold.")] = 5,
    mini_batch_size: Annotated[
        int | None, typer.Option("--mini-batch-size", help="Override mini-batch size.")
    ] = None,
    latent_dim: Annotated[
        int | None,
        typer.Option("--latent-dim", help="Override model latent dimension."),
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
    min_train_week: Annotated[
        int, typer.Option("--min-train-week", help="First canonical week allowed to train.")
    ] = 1,
    max_validation_week: Annotated[
        int | None,
        typer.Option("--max-validation-week", help="Last canonical validation week."),
    ] = None,
    save_fold_checkpoints: Annotated[
        bool,
        typer.Option(
            "--save-fold-checkpoints/--no-save-fold-checkpoints",
            help="Write one model checkpoint per walk-forward fold.",
        ),
    ] = False,
    tensorboard: Annotated[
        bool,
        typer.Option(
            "--tensorboard/--no-tensorboard",
            help="Write local TensorBoard event files for walk-forward validation.",
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
    world_model_config: Annotated[
        Path | None,
        typer.Option(
            "--frozen-target-config",
            "--world-model-config",
            help="Optional experimental fold-local frozen-target config.",
        ),
    ] = None,
    event_metadata: Annotated[
        Path | None,
        typer.Option(
            "--event-metadata", help="Event metadata Parquet for official-event filtering."
        ),
    ] = None,
    award_catalog: Annotated[
        Path | None,
        typer.Option("--award-catalog", help="Award prototype YAML for enabled award targets."),
    ] = None,
    openai_cache: Annotated[
        Path,
        typer.Option("--openai-cache", help="OpenAI embedding cache for award prototypes."),
    ] = OPENAI_EMBEDDING_CACHE_PATH,
) -> None:
    """Run temporal walk-forward validation."""
    updates = {"epochs": epochs, "use_early_stopping": False}
    if mini_batch_size is not None:
        updates["mini_batch_size"] = mini_batch_size
    if latent_dim is not None:
        updates["latent_dim"] = latent_dim
    opts = default_options().model_copy(update=updates)
    sidecar_tables = {}
    source_paths = {}
    for name, path in (
        ("rankings", rankings_sidecar),
        ("selections", selections_sidecar),
        ("playoffs", playoffs_sidecar),
    ):
        if path is not None:
            sidecar_tables[name] = pd.read_parquet(path, engine="pyarrow")
            source_paths[f"{name}_sidecar"] = path
    result = run_walk_forward_validation(
        features,
        output,
        prior_checkpoint=prior_checkpoint,
        opts=opts,
        sidecar_tables=sidecar_tables or None,
        min_train_week=min_train_week,
        max_validation_week=max_validation_week,
        save_fold_checkpoints=save_fold_checkpoints,
        tensorboard_logdir=tensorboard_logdir if tensorboard else None,
        tensorboard_run_name=tensorboard_run_name,
        source_paths=source_paths,
        world_model_options=(
            load_world_model_options(world_model_config)
            if world_model_config is not None
            else None
        ),
        world_model_events_path=event_metadata,
        award_catalog_path=award_catalog,
        openai_cache_path=openai_cache,
    )
    if result.tensorboard_logdir is not None:
        typer.echo(f"TensorBoard active: tensorboard --logdir={tensorboard_logdir}")
    fold_count = int((result.metrics["fold_number"] != "AVERAGE").sum())
    typer.echo(f"Walk-forward validation complete: {result.output_dir} folds={fold_count}")


@app.command("build-world-model")
def build_world_model(
    config: Annotated[Path, typer.Option("--config", help="Experimental frozen-target YAML.")],
    features: Annotated[Path, typer.Option("--features", help="Season feature Parquet.")],
    events: Annotated[
        Path, typer.Option("--events", help="Event metadata Parquet with event_type.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Frozen-target bundle output directory.")
    ] = EXPERIMENTAL_FROZEN_TARGET_ARTIFACT_ROOT / "v6_lite",
    rankings: Annotated[
        Path | None, typer.Option("--rankings", help="Optional rankings sidecar Parquet.")
    ] = None,
    selections: Annotated[
        Path | None, typer.Option("--selections", help="Optional selections sidecar Parquet.")
    ] = None,
    playoffs: Annotated[
        Path | None, typer.Option("--playoffs", help="Optional playoffs sidecar Parquet.")
    ] = None,
    award_catalog: Annotated[
        Path | None, typer.Option("--award-catalog", help="Optional award prototype YAML.")
    ] = None,
    openai_cache: Annotated[
        Path, typer.Option("--openai-cache", help="OpenAI embedding cache.")
    ] = OPENAI_EMBEDDING_CACHE_PATH,
    fit_max_week: Annotated[
        int | None, typer.Option("--fit-max-week", help="Optional fold-local week boundary.")
    ] = None,
) -> None:
    """Build experimental frozen target spaces offline."""
    bundle = build_world_model_bundle(
        features,
        output,
        load_world_model_options(config),
        events_path=events,
        rankings_path=rankings,
        selections_path=selections,
        playoffs_path=playoffs,
        award_catalog_path=award_catalog,
        openai_cache_path=openai_cache,
        fit_max_week=fit_max_week,
    )
    typer.echo(
        f"World-model bundle written: {bundle.root} spaces={','.join(sorted(bundle.spaces))}"
    )


@app.command("sync-match-breakdowns")
def sync_match_breakdowns_command(
    start_season: Annotated[
        int, typer.Option("--start-season", help="First TBA season to synchronize.")
    ] = 2015,
    end_season: Annotated[
        int, typer.Option("--end-season", help="Last TBA season to synchronize.")
    ] = 2026,
    include_foc: Annotated[
        bool,
        typer.Option(
            "--include-foc/--exclude-foc",
            help="Include FIRST Festival of Champions event type 6.",
        ),
    ] = False,
    include_remote: Annotated[
        bool,
        typer.Option(
            "--include-remote/--exclude-remote",
            help="Include remote event type 7.",
        ),
    ] = False,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh/--no-refresh",
            help="Conditionally refresh stored endpoint responses using ETags.",
        ),
    ] = False,
    corpus_db: Annotated[
        Path, typer.Option("--corpus-db", help="Durable raw match-breakdown SQLite corpus.")
    ] = MATCH_BREAKDOWN_CORPUS_PATH,
) -> None:
    """Synchronize raw historical TBA match payloads for offline score pretraining."""

    result = sync_match_breakdowns(
        _provider(),
        corpus_db,
        start_season=start_season,
        end_season=end_season,
        include_foc=include_foc,
        include_remote=include_remote,
        refresh=refresh,
    )
    typer.echo(
        f"Match-breakdown sync {result.status}: corpus={corpus_db} run={result.run_id} "
        + " ".join(f"{name}={value}" for name, value in sorted(result.counts.items()))
    )
    if result.errors:
        for error in result.errors:
            typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1)


@app.command("train-match-breakdown-encoder")
def train_match_breakdown_encoder_command(
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Explicit V2 objective YAML; omitted keeps V1 defaults."),
    ] = None,
    start_season: Annotated[
        int | None, typer.Option("--start-season", help="First corpus season to include.")
    ] = None,
    end_season: Annotated[
        int | None, typer.Option("--end-season", help="Last corpus season to include.")
    ] = None,
    epochs: Annotated[
        int | None, typer.Option("--epochs", help="Epochs for each training phase.")
    ] = None,
    seasons_per_step: Annotated[
        int | None,
        typer.Option("--seasons-per-step", help="Season routes sampled per optimizer step."),
    ] = None,
    rows_per_season: Annotated[
        int | None,
        typer.Option("--rows-per-season", help="Alliance rows sampled per season route."),
    ] = None,
    learning_rate: Annotated[
        float | None, typer.Option("--learning-rate", help="AdamW learning rate.")
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Deterministic random seed.")
    ] = None,
    include_foc: Annotated[
        bool | None,
        typer.Option(
            "--include-foc/--exclude-foc",
            help="Include FIRST Festival of Champions event type 6 from the corpus.",
        ),
    ] = None,
    include_remote: Annotated[
        bool | None,
        typer.Option(
            "--include-remote/--exclude-remote",
            help="Include remote event type 7 from the corpus.",
        ),
    ] = None,
    corpus_db: Annotated[
        Path, typer.Option("--corpus-db", help="Durable raw match-breakdown SQLite corpus.")
    ] = MATCH_BREAKDOWN_CORPUS_PATH,
    output: Annotated[
        Path | None, typer.Option("--output", help="Offline encoder artifact directory.")
    ] = None,
    alliance_features: Annotated[
        Path | None,
        typer.Option("--alliance-features", help="Flattened raw alliance Parquet output."),
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help='Torch device, or "auto".')
    ] = None,
) -> None:
    """Train the offline historical match-breakdown encoder."""

    options = load_match_breakdown_training_options(
        config,
        overrides={
            "start_season": start_season,
            "end_season": end_season,
            "epochs": epochs,
            "seasons_per_step": seasons_per_step,
            "rows_per_season": rows_per_season,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": device,
            "include_foc": include_foc,
            "include_remote": include_remote,
        },
    )
    result = train_match_breakdown_encoder(
        corpus_db,
        output
        or match_breakdown_artifact_dir(
            options.start_season, options.end_season, options.artifact_version
        ),
        alliance_features
        or match_breakdown_features_path(options.start_season, options.end_season),
        options,
    )
    typer.echo(
        f"Match-breakdown encoder written: {result.output_dir} "
        f"rows={result.bundle['audit']['alliance_rows']} "
        f"seasons={','.join(str(value) for value in result.bundle['seasons_trained'])} "
        f"batch_size={result.bundle['derived_batch_size']}"
    )


@app.command("inspect-match-breakdown-encoder")
def inspect_match_breakdown_encoder_command(
    artifact_dir: Annotated[
        Path,
        typer.Option("--artifact-dir", help="Trained historical match-breakdown artifact."),
    ] = DEFAULT_MATCH_BREAKDOWN_ARTIFACT_DIR,
    output: Annotated[
        Path | None, typer.Option("--output", help="Standalone static inspection directory.")
    ] = None,
    baseline_artifact_dir: Annotated[
        Path | None,
        typer.Option(
            "--baseline-artifact-dir",
            help="Optional preserved artifact for keyed neighbor and score-sorting drift tables.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic sampling seed.")] = 2026,
    tsne_max_rows: Annotated[
        int, typer.Option("--tsne-max-rows", help="Maximum production t-SNE sample rows.")
    ] = 11_000,
    network_node_limit: Annotated[
        int, typer.Option("--network-node-limit", help="Maximum extreme network seed nodes.")
    ] = 400,
    neighbors_per_node: Annotated[
        int,
        typer.Option("--neighbors-per-node", help="Cross-season neighbors retained per seed."),
    ] = 2,
) -> None:
    """Generate a standalone static inspection report for frozen alliance embeddings."""

    result = write_match_breakdown_inspection(
        artifact_dir,
        output or artifact_dir / "inspection",
        MatchBreakdownInspectionOptions(
            seed=seed,
            tsne_max_rows=tsne_max_rows,
            network_node_limit=network_node_limit,
            neighbors_per_node=neighbors_per_node,
        ),
        baseline_artifact_dir=baseline_artifact_dir,
    )
    typer.echo(
        f"Match-breakdown inspection written: {result.output_dir} "
        f"production_rows={result.manifest['geometry']['production']['rows']} "
        f"holdout_rows={result.manifest['geometry']['holdout']['rows']}"
    )


@app.command("compare-world-model")
def compare_world_model(
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline predictions Parquet.")],
    candidate: Annotated[Path, typer.Option("--candidate", help="Candidate predictions Parquet.")],
    output: Annotated[
        Path | None, typer.Option("--output", help="Optional comparison CSV output.")
    ] = None,
    resamples: Annotated[
        int, typer.Option("--resamples", help="Paired bootstrap sample count.")
    ] = 2_000,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic bootstrap seed.")] = 2026,
) -> None:
    """Evaluate V6 promotion non-inferiority against paired predictions."""
    report = paired_bootstrap_noninferiority(
        pd.read_parquet(baseline, engine="pyarrow"),
        pd.read_parquet(candidate, engine="pyarrow"),
        resamples=resamples,
        seed=seed,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output, index=False)
    typer.echo(report.to_string(index=False))


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
    ] = OPENAI_EMBEDDING_CACHE_PATH,
    max_team_number: Annotated[
        int,
        typer.Option("--max-team-number", help="Largest team number to include."),
    ] = 12_500,
    epa_source_year: Annotated[
        int | None,
        typer.Option(
            "--epa-source-year",
            help="Final completed Statbotics season to use for normalized EPA trajectory.",
        ),
    ] = None,
) -> None:
    """Build the full V5.6 transductive prior feature table."""
    output_path = output or prior_features_path(target_season)
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
        f"norm_epa_source_years={table['norm_epa_source_years'].iloc[0]}"
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
    output_path = output or PRIOR_ARTIFACT_ROOT / f"pretrained_prior_{pd.Timestamp.utcnow().year}"
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
    ] = PRIOR_ARTIFACT_ROOT / "prior_2026",
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
    ] = PRIOR_GRID_ARTIFACT_ROOT / "prior_grid_2026",
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
    checkpoint: Annotated[
        Path, typer.Argument(help="Season checkpoint produced by train-features.")
    ],
    event_key: Annotated[str, typer.Option("--event-key", help="Event key to consolidate.")],
    embedding_db: Annotated[
        Path,
        typer.Option("--embedding-db", help="SQLite store for durable base embeddings."),
    ] = EMBEDDING_DB_PATH,
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


@app.command("write-baseline-manifest")
def write_baseline_manifest_command(
    archive: Annotated[Path, typer.Argument(help="Ignored baseline artifact archive.")],
    output: Annotated[
        Path, typer.Option("--output", help="Tracked baseline manifest JSON.")
    ] = Path("baselines/v5.8-baseline.json"),
    tag: Annotated[
        str, typer.Option("--tag", help="Annotated baseline Git tag.")
    ] = "v5.8-baseline",
    commit_sha: Annotated[
        str | None,
        typer.Option("--commit-sha", help="Tagged commit SHA. Defaults to archive commit.txt."),
    ] = None,
    regeneration_command: Annotated[
        str,
        typer.Option("--regeneration-command", help="Command recorded for baseline replay."),
    ] = "latentstrat validate-walk-forward --help",
) -> None:
    """Write the tracked hash manifest for an ignored empirical baseline archive."""
    resolved_commit = commit_sha or (archive / "commit.txt").read_text(encoding="utf-8").strip()
    path = write_baseline_manifest(
        archive,
        output,
        tag=tag,
        commit_sha=resolved_commit,
        regeneration_commands=[regeneration_command],
    )
    typer.echo(f"Baseline manifest written: {path}")


@app.command("full-season-offline")
def full_season_offline(season: int = 2026, event_limit: int | None = None) -> None:
    """Explain the supported replacement for the removed duplicate runner."""

    _ = (season, event_limit)
    typer.echo(
        "Error: `full-season-offline` was removed in V6.1. Run "
        "`latentstrat season build-features --season 2026` followed by "
        "`latentstrat season train <features.parquet>`.",
        err=True,
    )
    raise typer.Exit(code=2)


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
    out = INSPECTION_ARTIFACT_ROOT / "embeddings"
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
    typer.echo(f"Evidence packet written to {EVIDENCE_ARTIFACT_ROOT / 'evidence_packet'}")


@app.command("clear-cache")
def clear_cache() -> None:
    """Remove Python provider caches."""
    import shutil

    targets = [
        TBA_CACHE_BASE.with_suffix(".sqlite"),
        STATBOTICS_CACHE_PATH,
        OPENAI_EMBEDDING_CACHE_PATH,
        Path("data/prior_cache/openai_embeddings.sqlite"),
        Path("tba_cache.sqlite"),
        Path("statbotics_offline_cache"),
    ]
    removed = 0
    for path in targets:
        for target in (path, Path(f"{path}-shm"), Path(f"{path}-wal")):
            if target.is_dir():
                shutil.rmtree(target)
                removed += 1
            elif target.exists():
                target.unlink()
                removed += 1
    if CACHE_ROOT.exists():
        for sidecar in CACHE_ROOT.glob("*.sqlite-*"):
            sidecar.unlink()
            removed += 1
    for legacy_sidecar in Path(".").glob("tba_cache.sqlite-*"):
        if legacy_sidecar.exists():
            legacy_sidecar.unlink()
            removed += 1
    typer.echo(f"Caches cleared. removed={removed}")


@venue_app.command("train")
def experimental_venue_train(
    ctx: typer.Context,
    input_path: Annotated[Path, typer.Argument(help="Event feature Parquet file.")],
    event_key: Annotated[str, typer.Option("--event-key", help="Event key to isolate.")],
    checkpoint: Annotated[
        Path, typer.Option("--checkpoint", help="Season checkpoint to fine-tune.")
    ],
    output: Annotated[
        Path, typer.Option("--output", help="Directory for venue-mode artifacts.")
    ] = SEASON_ARTIFACT_ROOT / "venue",
) -> None:
    """Train only event-delta embeddings for an explicit experimental venue run."""

    ctx.invoke(
        train_features,
        input_path=input_path,
        event_key=event_key,
        checkpoint=checkpoint,
        output=output,
        venue_mode=True,
    )


@dev_app.command("migrate-layout")
def migrate_layout(
    apply: Annotated[
        bool, typer.Option("--apply/--dry-run", help="Apply verified moves after preview.")
    ] = False,
) -> None:
    """Preview or apply the generated-output storage migration."""

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/organize_local_outputs.ps1",
    ]
    if apply:
        command.append("-Apply")
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise typer.Exit(code=result.returncode)


prior_app.command("build")(build_prior_features)
prior_app.command("train")(train_prior)
prior_app.command("inspect")(inspect_prior)
prior_app.command("grid")(run_prior_grid_command)
match_breakdown_app.command("sync")(sync_match_breakdowns_command)
match_breakdown_app.command("train")(train_match_breakdown_encoder_command)
match_breakdown_app.command("inspect")(inspect_match_breakdown_encoder_command)
season_app.command("build-features")(build_features)
season_app.command("train")(train_features)
season_app.command("validate")(validate_walk_forward)
artifact_app.command("baseline-manifest")(write_baseline_manifest_command)
artifact_app.command("compare-predictions")(compare_world_model)
scouting_app.command("init")(init_scouting_db)
dev_app.command("api-smoke")(api_smoke)
dev_app.command("smoke-test")(smoke_test)
dev_app.command("clear-cache")(clear_cache)
diagnostics_app.command("embeddings")(inspect_embeddings_command)
diagnostics_app.command("evidence")(build_evidence_packet_command)
frozen_targets_app.command("build")(build_world_model)
venue_app.command("consolidate")(consolidate_event)


if __name__ == "__main__":
    app()
