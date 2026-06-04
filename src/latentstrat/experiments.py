"""Controls and evidence-packet generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from latentstrat.baselines import Baselines, fit_baselines
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.inspection import inspect_embeddings
from latentstrat.paths import EVIDENCE_ARTIFACT_ROOT
from latentstrat.season.data import (
    Split,
    TargetStats,
    apply_target_stats,
    fit_target_stats,
    make_split,
)
from latentstrat.season.evaluate import EvaluationReport, evaluate_model
from latentstrat.season.train import TrainingDiagnostics, train_model


@dataclass
class RunRecord:
    name: str
    model: object
    history: pd.DataFrame
    diagnostics: TrainingDiagnostics
    report: EvaluationReport
    inspection: object
    opts: LatentStratOptions


@dataclass
class EvidencePacket:
    scoreboard: pd.DataFrame
    set_attention: pd.DataFrame
    zero_out_diagnostics: pd.DataFrame
    neighbor_coherence: pd.DataFrame
    seed_stability: pd.DataFrame
    run_records: list[RunRecord]


def _split_masks(split: Split) -> list[np.ndarray]:
    return [split.train_mask, split.validation_mask, split.test_mask]


def shuffle_teams_within_split(
    table: pd.DataFrame, split: Split, *, seed: int = 2026
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = table.copy()
    slot_prefixes = [
        "red_team_1",
        "red_team_2",
        "red_team_3",
        "blue_team_1",
        "blue_team_2",
        "blue_team_3",
    ]
    for mask in _split_masks(split):
        rows = np.flatnonzero(mask)
        if len(rows) <= 1:
            continue
        for prefix in slot_prefixes:
            permutation = rng.permutation(rows)
            for suffix in ("key", "idx"):
                column = f"{prefix}_{suffix}"
                out.loc[rows, column] = table.loc[permutation, column].to_numpy()
    return out


def shuffle_targets_within_split(
    table: pd.DataFrame,
    split: Split,
    opts: LatentStratOptions | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    opts = opts or default_options()
    rng = np.random.default_rng(opts.random_seed if seed is None else seed)
    out = table.copy()
    target_names = [mapping.target_name for mapping in opts.target_map] + list(opts.binary_targets)
    target_names = [name for name in target_names if name in table.columns]
    for mask in _split_masks(split):
        rows = np.flatnonzero(mask)
        if len(rows) <= 1:
            continue
        permutation = rng.permutation(rows)
        for target in target_names:
            out.loc[rows, target] = table.loc[permutation, target].to_numpy()
    return out


def make_split_for_seed(table: pd.DataFrame, opts: LatentStratOptions, seed: int) -> Split:
    try:
        split = make_split(
            table,
            opts,
            policy=opts.full_season_split_policy,
            validation_fraction=opts.validation_fraction,
            random_seed=seed,
        )
    except KeyError:
        split = make_split(
            table,
            opts,
            policy="chronological-holdout",
            validation_fraction=opts.validation_fraction,
            random_seed=seed,
        )
        split.fallback_reason = "missing_event_week"
    split.seed = seed
    return split


def _embedding_matrix(team_table: pd.DataFrame) -> np.ndarray:
    return np.vstack(team_table["embedding"].to_numpy())


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    out = values / np.maximum(norms, np.finfo(float).eps)
    out[norms[:, 0] <= np.finfo(float).eps] = 0
    return out


def neighbor_coherence(
    team_table: pd.DataFrame,
    *,
    metric_names: tuple[str, ...] = (
        "avg_alliance_score",
        "avg_point_differential",
        "avg_fouls_drawn",
        "rate_win",
    ),
    top_k: int = 10,
    match_count_tolerance: float = 5,
    seed: int = 2026,
    min_match_count: float = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    embeddings = _embedding_matrix(team_table)
    normalized = _normalize_rows(embeddings)
    rows = []
    for metric in [name for name in metric_names if name in team_table.columns]:
        values = team_table[metric].to_numpy(dtype=float)
        eligible = np.flatnonzero(
            (~np.isnan(values)) & (team_table["match_count"].to_numpy() >= min_match_count)
        )
        neighbor_diffs = []
        random_diffs = []
        for query in eligible:
            similarities = normalized @ normalized[query]
            similarities[query] = -np.inf
            neighbors = np.argsort(-similarities)[: min(top_k, len(team_table) - 1)]
            query_matches = team_table["match_count"].iloc[query]
            tolerance = max(match_count_tolerance, 0.25 * query_matches)
            candidates = eligible[
                np.abs(team_table["match_count"].iloc[eligible].to_numpy() - query_matches)
                <= tolerance
            ]
            candidates = np.setdiff1d(
                candidates, np.concatenate([[query], neighbors]), assume_unique=False
            )
            if len(candidates) < top_k:
                candidates = np.setdiff1d(
                    eligible, np.concatenate([[query], neighbors]), assume_unique=False
                )
            if len(candidates) == 0 or len(neighbors) == 0:
                continue
            random_rows = rng.choice(candidates, size=min(top_k, len(candidates)), replace=False)
            neighbor_diffs.append(float(np.nanmean(np.abs(values[neighbors] - values[query]))))
            random_diffs.append(float(np.nanmean(np.abs(values[random_rows] - values[query]))))
        neighbor_mean = float(np.nanmean(neighbor_diffs)) if neighbor_diffs else np.nan
        random_mean = float(np.nanmean(random_diffs)) if random_diffs else np.nan
        improvement = random_mean - neighbor_mean
        rows.append(
            {
                "metric": metric,
                "top_k": top_k,
                "query_count": len(neighbor_diffs),
                "neighbor_mean_abs_diff": neighbor_mean,
                "random_mean_abs_diff": random_mean,
                "absolute_improvement": improvement,
                "percent_improvement": 100 * improvement / max(random_mean, np.finfo(float).eps)
                if np.isfinite(random_mean)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _scoreboard_variables() -> list[str]:
    return [
        "variant",
        "seed",
        "split",
        "target",
        "target_type",
        "rmse",
        "mae",
        "brier",
        "log_loss",
        "parameter_count",
        "rows_per_parameter",
        "train_rows",
        "validation_rows",
        "split_policy",
        "latent_dim",
        "learning_rate",
        "l2",
        "set_ffn_dim",
        "l2_set",
    ]


def _metric_rows(
    name: str,
    metrics: pd.DataFrame,
    seed: int,
    split: Split,
    opts: LatentStratOptions,
    report: EvaluationReport | None,
    target_type: str,
) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            {
                "variant": name,
                "seed": seed,
                "split": row["split"],
                "target": row["target"],
                "target_type": target_type,
                "rmse": row.get("rmse", np.nan),
                "mae": row.get("mae", np.nan),
                "brier": row.get("brier", np.nan),
                "log_loss": row.get("log_loss", np.nan),
                "parameter_count": np.nan if report is None else report.parameter_count,
                "rows_per_parameter": np.nan if report is None else report.rows_per_parameter,
                "train_rows": int(np.sum(split.train_mask)),
                "validation_rows": int(np.sum(split.validation_mask)),
                "split_policy": split.policy,
                "latent_dim": opts.latent_dim,
                "learning_rate": opts.learning_rate,
                "l2": opts.l2,
                "set_ffn_dim": opts.set_ffn_dim,
                "l2_set": opts.l2_set,
            }
        )
    return pd.DataFrame(rows, columns=_scoreboard_variables())


def _seed_stability(scoreboard: pd.DataFrame) -> pd.DataFrame:
    validation = scoreboard[scoreboard["split"] == "validation"]
    if validation.empty:
        return pd.DataFrame()
    grouped = validation.groupby(["variant", "target", "target_type"], dropna=False)
    return grouped.agg(
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_brier=("brier", "mean"),
        std_brier=("brier", "std"),
    ).reset_index()


def _write_markdown(packet: EvidencePacket, path: Path) -> None:
    validation = packet.scoreboard[
        (packet.scoreboard["split"] == "validation")
        & (packet.scoreboard["target_type"] == "continuous")
    ].head(20)
    lines = [
        "# LatentStrat Evidence Packet",
        "",
        (
            "This packet compares the cross-alliance Set Transformer against "
            "mean/ridge baselines and data-level controls."
        ),
        "",
        (
            "PMA attention and zero-out deltas are diagnostics only. "
            "Validation metrics decide whether the model generalizes."
        ),
        "",
        "## Validation Scoreboard",
        "",
        validation[["variant", "seed", "target", "rmse", "mae"]].to_markdown(index=False)
        if not validation.empty
        else "_No validation rows._",
        "",
        "## Seed Stability",
        "",
        packet.seed_stability.head(20).to_markdown(index=False)
        if not packet.seed_stability.empty
        else "_No seed stability rows._",
        "",
        "## Zero-Out Diagnostics",
        "",
        packet.zero_out_diagnostics.head(20).to_markdown(index=False)
        if not packet.zero_out_diagnostics.empty
        else "_No zero-out rows._",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_evidence_packet(
    table: pd.DataFrame,
    team_index_map: dict[str, int],
    opts: LatentStratOptions | None = None,
    *,
    output_dir: str | Path = EVIDENCE_ARTIFACT_ROOT / "evidence_packet",
    write_files: bool = True,
    verbose: bool = True,
    seeds: tuple[int, ...] | None = None,
) -> EvidencePacket:
    opts = opts or default_options()
    seeds = opts.evidence_seeds if seeds is None else seeds
    scoreboard = []
    set_attention = []
    zero_out = []
    coherence = []
    records = []
    for seed in seeds:
        run_opts = opts.model_copy(update={"random_seed": seed})
        split = make_split_for_seed(table, run_opts, seed)
        target_stats = fit_target_stats(table, split.train_mask, run_opts)
        prepared = apply_target_stats(table, target_stats)
        baselines = fit_baselines(prepared, split, run_opts)

        if verbose:
            print(f"Evidence seed {seed}: training LatentStrat model.")
        official = _train_and_evaluate(
            "latentstrat",
            prepared,
            split,
            target_stats,
            run_opts,
            baselines,
            team_index_map,
        )
        records.append(official)
        scoreboard.append(
            _metric_rows(
                "latentstrat",
                official.report.continuous_metrics,
                seed,
                split,
                run_opts,
                official.report,
                "continuous",
            )
        )
        scoreboard.append(
            _metric_rows(
                "latentstrat",
                official.report.binary_metrics,
                seed,
                split,
                run_opts,
                official.report,
                "binary",
            )
        )
        scoreboard.append(
            _metric_rows(
                "mean_baseline", baselines.mean.metrics, seed, split, run_opts, None, "continuous"
            )
        )
        scoreboard.append(
            _metric_rows(
                "ridge_match_opr",
                baselines.ridge.metrics,
                seed,
                split,
                run_opts,
                None,
                "continuous",
            )
        )
        set_attention.append(official.report.set_attention.assign(variant="latentstrat", seed=seed))
        zero_out.append(
            official.report.zero_out_diagnostics.assign(variant="latentstrat", seed=seed)
        )
        coherence.append(
            neighbor_coherence(official.inspection.team_embedding_table, seed=seed).assign(
                variant="latentstrat", seed=seed
            )
        )

        for name, variant_table in (
            ("shuffled_set_control", shuffle_teams_within_split(table, split, seed=seed)),
            ("null_label_control", shuffle_targets_within_split(table, split, run_opts, seed=seed)),
        ):
            stats = fit_target_stats(variant_table, split.train_mask, run_opts)
            prepared_variant = apply_target_stats(variant_table, stats)
            variant_baselines = fit_baselines(prepared_variant, split, run_opts)
            run = _train_and_evaluate(
                name, prepared_variant, split, stats, run_opts, variant_baselines, team_index_map
            )
            records.append(run)
            scoreboard.append(
                _metric_rows(
                    name,
                    run.report.continuous_metrics,
                    seed,
                    split,
                    run_opts,
                    run.report,
                    "continuous",
                )
            )
            scoreboard.append(
                _metric_rows(
                    name, run.report.binary_metrics, seed, split, run_opts, run.report, "binary"
                )
            )
            set_attention.append(run.report.set_attention.assign(variant=name, seed=seed))
            zero_out.append(run.report.zero_out_diagnostics.assign(variant=name, seed=seed))
            coherence.append(
                neighbor_coherence(run.inspection.team_embedding_table, seed=seed).assign(
                    variant=name, seed=seed
                )
            )
    packet = EvidencePacket(
        scoreboard=pd.concat(scoreboard, ignore_index=True)
        if scoreboard
        else pd.DataFrame(columns=_scoreboard_variables()),
        set_attention=pd.concat(set_attention, ignore_index=True)
        if set_attention
        else pd.DataFrame(),
        zero_out_diagnostics=pd.concat(zero_out, ignore_index=True) if zero_out else pd.DataFrame(),
        neighbor_coherence=pd.concat(coherence, ignore_index=True) if coherence else pd.DataFrame(),
        seed_stability=pd.DataFrame(),
        run_records=records,
    )
    packet.seed_stability = _seed_stability(packet.scoreboard)
    if write_files:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        packet.scoreboard.to_csv(out / "scoreboard.csv", index=False)
        packet.set_attention.to_csv(out / "set_attention.csv", index=False)
        packet.zero_out_diagnostics.to_csv(out / "zero_out_diagnostics.csv", index=False)
        packet.neighbor_coherence.to_csv(out / "neighbor_coherence.csv", index=False)
        packet.seed_stability.to_csv(out / "seed_stability.csv", index=False)
        _write_markdown(packet, out / "evidence_packet.md")
    return packet


def _train_and_evaluate(
    name: str,
    prepared: pd.DataFrame,
    split: Split,
    target_stats: TargetStats,
    opts: LatentStratOptions,
    baselines: Baselines,
    team_index_map: dict[str, int],
) -> RunRecord:
    model, history, diagnostics = train_model(prepared, split, opts, verbose=False)
    report = evaluate_model(model, prepared, split, target_stats, opts, baselines)
    inspection = inspect_embeddings(model, prepared, team_index_map, opts, baselines, top_k=5)
    return RunRecord(name, model, history, diagnostics, report, inspection, opts)
