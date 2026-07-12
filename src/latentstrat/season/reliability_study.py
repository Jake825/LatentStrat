"""Multi-seed reliability experiment for the static 2026 robot-state architecture."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from latentstrat.artifacts import write_artifact_manifest
from latentstrat.config import default_options
from latentstrat.season.features import FeatureTrainingResult, read_feature_table
from latentstrat.season.reference_study import (
    ARCHITECTURES,
    LOWER_IS_BETTER,
    _baseline_frames,
    _filter_sidecars,
    _prediction_frame,
    _read_table,
    _sidecar_validation_metrics,
    _sidecars_with_weeks,
    _train_once,
    _write_checkpoint,
    nested_temporal_split,
    paired_event_bootstrap,
    refit_temporal_split,
    validate_reference_rows,
)
from latentstrat.season.statbotics_baseline import validate_statbotics_prediction_table

RELIABILITY_WORKFLOW = "season.reference-2026-static-reliability"
RELIABILITY_SCHEMA = "2026-static-architecture-reliability-v1"
DEFAULT_SEEDS = (2026, 2027, 2028)
PRIMARY_WEEKS = (6, 8)
STRESS_WEEK = 10
CLIP_CANDIDATES = (2.5, 5.0, 10.0)
EPOCH_CANDIDATES = tuple(range(15, 41))
CALIBRATION_MIN_ROWS = 1_500
PRACTICAL_TOLERANCES = {
    "score_differential_squared_error": 0.02,
    "winner_brier": 0.005,
    "winner_log_loss": 0.01,
}


@dataclass(frozen=True)
class ReliabilityStudyResult:
    output_dir: Path
    manifest_path: Path
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_max_grad_norm: float
    selected_epochs: int
    conclusion_status: str
    tensorboard_logdir: Path | None


def select_gradient_threshold(
    optimization_grid: pd.DataFrame,
    *,
    candidates: Sequence[float] = CLIP_CANDIDATES,
    max_mean_clipping_fraction: float = 0.10,
    warmup_epochs: int = 5,
) -> float:
    """Select the smallest numerically stable threshold that clips at most 10 percent."""

    required = {
        "architecture",
        "max_grad_norm",
        "epoch",
        "train_loss",
        "validation_loss",
        "clipping_fraction",
    }
    missing = sorted(required - set(optimization_grid.columns))
    if missing:
        raise ValueError("Optimization grid is missing: " + ", ".join(missing))
    for threshold in candidates:
        rows = optimization_grid[
            np.isclose(pd.to_numeric(optimization_grid["max_grad_norm"]), float(threshold))
        ]
        if set(rows["architecture"].astype(str)) != set(ARCHITECTURES):
            continue
        numeric = rows[["train_loss", "validation_loss", "clipping_fraction"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(float)).all():
            continue
        post_warmup = rows[pd.to_numeric(rows["epoch"], errors="coerce") > warmup_epochs]
        if post_warmup.empty:
            post_warmup = rows
        clipping = post_warmup.groupby("architecture")["clipping_fraction"].mean()
        if clipping.reindex(ARCHITECTURES).notna().all() and (
            clipping.reindex(ARCHITECTURES) <= max_mean_clipping_fraction
        ).all():
            return float(threshold)
    raise RuntimeError(
        "No max_grad_norm candidate was finite and below the 10% post-warmup clipping gate."
    )


def eligible_epoch_counts(
    optimization_grid: pd.DataFrame,
    *,
    selected_max_grad_norm: float,
    candidates: Sequence[int] = EPOCH_CANDIDATES,
) -> tuple[int, ...]:
    """Return common epoch counts within the predeclared development tolerances."""

    metric_columns = (
        "validation_score_differential_rmse",
        "validation_winner_brier",
        "validation_winner_log_loss",
    )
    required = {"architecture", "max_grad_norm", "epoch", *metric_columns}
    missing = sorted(required - set(optimization_grid.columns))
    if missing:
        raise ValueError("Optimization grid is missing: " + ", ".join(missing))
    rows = optimization_grid[
        np.isclose(
            pd.to_numeric(optimization_grid["max_grad_norm"], errors="coerce"),
            selected_max_grad_norm,
        )
    ].copy()
    best = rows.groupby("architecture")[list(metric_columns)].min()
    eligible: list[int] = []
    for epoch in candidates:
        at_epoch = rows[pd.to_numeric(rows["epoch"], errors="coerce").eq(int(epoch))]
        if set(at_epoch["architecture"].astype(str)) != set(ARCHITECTURES):
            continue
        current = at_epoch.set_index("architecture")
        ok = True
        for architecture in ARCHITECTURES:
            diff = float(current.loc[architecture, metric_columns[0]])
            brier = float(current.loc[architecture, metric_columns[1]])
            log_loss = float(current.loc[architecture, metric_columns[2]])
            ok &= diff <= 1.02 * float(best.loc[architecture, metric_columns[0]])
            ok &= brier <= float(best.loc[architecture, metric_columns[1]]) + 0.005
            ok &= log_loss <= float(best.loc[architecture, metric_columns[2]]) + 0.01
        if ok:
            eligible.append(int(epoch))
    return tuple(eligible)


def confirmation_passes(
    confirmation: pd.DataFrame,
    optimization_grid: pd.DataFrame,
    *,
    selected_max_grad_norm: float,
) -> bool:
    if set(confirmation["architecture"].astype(str)) != set(ARCHITECTURES):
        return False
    grid = optimization_grid[
        np.isclose(
            pd.to_numeric(optimization_grid["max_grad_norm"], errors="coerce"),
            selected_max_grad_norm,
        )
    ]
    best = grid.groupby("architecture")[
        [
            "validation_score_differential_rmse",
            "validation_winner_brier",
            "validation_winner_log_loss",
        ]
    ].min()
    final = confirmation.sort_values("epoch").groupby("architecture").tail(1).set_index(
        "architecture"
    )
    for architecture in ARCHITECTURES:
        row = final.loc[architecture]
        if not np.isfinite(
            row[
                [
                    "train_loss",
                    "validation_loss",
                    "validation_score_differential_rmse",
                    "validation_winner_brier",
                    "validation_winner_log_loss",
                ]
            ].to_numpy(float)
        ).all():
            return False
        if float(row["validation_score_differential_rmse"]) > 1.02 * float(
            best.loc[architecture, "validation_score_differential_rmse"]
        ):
            return False
        if float(row["validation_winner_brier"]) > float(
            best.loc[architecture, "validation_winner_brier"]
        ) + 0.005:
            return False
        if float(row["validation_winner_log_loss"]) > float(
            best.loc[architecture, "validation_winner_log_loss"]
        ) + 0.01:
            return False
        post_warmup = confirmation[
            confirmation["architecture"].astype(str).eq(architecture)
            & pd.to_numeric(confirmation["epoch"], errors="coerce").gt(5)
        ]
        if post_warmup.empty:
            post_warmup = confirmation[confirmation["architecture"].astype(str).eq(architecture)]
        if float(post_warmup["clipping_fraction"].mean()) > 0.10:
            return False
    return True


def minimum_campaign_seconds(
    seconds_per_epoch: float,
    *,
    development_epochs: int,
    confirmation_epochs: int,
    final_epochs: int,
    clip_count: int,
    architecture_count: int,
    seed_count: int,
    fold_count: int,
    safety_factor: float = 1.20,
) -> float:
    """Return a conservative lower-bound runtime before launching the development grid."""

    epoch_work = (
        clip_count * architecture_count * development_epochs
        + architecture_count * confirmation_epochs
        + seed_count * architecture_count * fold_count * final_epochs
    )
    return float(seconds_per_epoch * epoch_work * safety_factor)


def fit_sequential_calibrator(
    prior_predictions: pd.DataFrame,
    *,
    minimum_rows: int = CALIBRATION_MIN_ROWS,
) -> dict[str, Any]:
    """Fit Platt scaling from earlier held-out predictions only."""

    source_weeks = (
        sorted(
            pd.to_numeric(prior_predictions["test_week"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if "test_week" in prior_predictions
        else []
    )
    if prior_predictions.empty:
        return {
            "method": "identity-no-prior-heldout-bank",
            "coefficient": 1.0,
            "intercept": 0.0,
            "source_weeks": source_weeks,
            "row_count": 0,
        }
    probability = pd.to_numeric(
        prior_predictions["raw_pred_red_win_probability"], errors="coerce"
    ).to_numpy(float)
    red = pd.to_numeric(prior_predictions["actual_red_total_score"], errors="coerce").to_numpy(
        float
    )
    blue = pd.to_numeric(
        prior_predictions["actual_blue_total_score"], errors="coerce"
    ).to_numpy(float)
    valid = np.isfinite(probability) & np.isfinite(red) & np.isfinite(blue) & (red != blue)
    outcome = (red > blue).astype(int)
    if valid.sum() < minimum_rows or np.unique(outcome[valid]).size != 2:
        return {
            "method": "identity-insufficient-prior-heldout-bank",
            "coefficient": 1.0,
            "intercept": 0.0,
            "source_weeks": source_weeks,
            "row_count": int(valid.sum()),
        }
    clipped = np.clip(probability[valid], 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))
    model = LogisticRegression(random_state=2026).fit(logits[:, None], outcome[valid])
    return {
        "method": "platt-logit-prior-heldout-weeks",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "source_weeks": source_weeks,
        "row_count": int(valid.sum()),
    }


def apply_sequential_calibrator(
    predictions: pd.DataFrame, calibrator: dict[str, Any]
) -> pd.DataFrame:
    result = predictions.copy()
    raw = np.clip(result["pred_red_win_probability"].to_numpy(float), 1e-7, 1 - 1e-7)
    logits = np.log(raw / (1 - raw))
    calibrated = 1.0 / (
        1.0
        + np.exp(
            -(
                float(calibrator["coefficient"]) * logits
                + float(calibrator["intercept"])
            )
        )
    )
    result["raw_pred_red_win_probability"] = raw
    result["calibrated_pred_red_win_probability"] = calibrated
    result["pred_red_win_probability"] = calibrated
    result["calibration_method"] = str(calibrator["method"])
    result["calibration_source_weeks"] = json.dumps(calibrator.get("source_weeks", []))
    result["calibration_source_rows"] = int(calibrator.get("row_count", 0))
    return result


def _metric_values(frame: pd.DataFrame, probability_column: str) -> dict[str, tuple[float, int]]:
    actual_red = frame["actual_red_total_score"].to_numpy(float)
    actual_blue = frame["actual_blue_total_score"].to_numpy(float)
    predicted_red = frame["pred_red_total_score"].to_numpy(float)
    predicted_blue = frame["pred_blue_total_score"].to_numpy(float)
    alliance_error = np.concatenate([predicted_red - actual_red, predicted_blue - actual_blue])
    differential_error = (predicted_red - predicted_blue) - (actual_red - actual_blue)
    tie = actual_red == actual_blue
    probability = np.clip(frame[probability_column].to_numpy(float), 1e-7, 1 - 1e-7)
    valid = ~tie & np.isfinite(probability)
    outcome = (actual_red > actual_blue).astype(float)
    values = {
        "alliance_score_mae": (float(np.mean(np.abs(alliance_error))), len(alliance_error)),
        "alliance_score_rmse": (
            float(np.sqrt(np.mean(alliance_error**2))),
            len(alliance_error),
        ),
        "score_differential_mae": (
            float(np.mean(np.abs(differential_error))),
            len(differential_error),
        ),
        "score_differential_rmse": (
            float(np.sqrt(np.mean(differential_error**2))),
            len(differential_error),
        ),
    }
    if valid.any():
        y = outcome[valid]
        q = probability[valid]
        values.update(
            {
                "winner_accuracy": (float(np.mean((q >= 0.5) == y)), int(valid.sum())),
                "winner_error_rate": (float(np.mean((q >= 0.5) != y)), int(valid.sum())),
                "winner_brier": (float(np.mean((q - y) ** 2)), int(valid.sum())),
                "winner_log_loss": (
                    float(np.mean(-(y * np.log(q) + (1 - y) * np.log(1 - q)))),
                    int(valid.sum()),
                ),
            }
        )
    return values


def reliability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["seed", "model", "test_week", "evidence_role"]
    groupings: list[tuple[str, Any]] = [("test", predictions.groupby(group_columns))]
    primary = predictions[predictions["test_week"].isin(PRIMARY_WEEKS)]
    groupings.append(("aggregate", primary.groupby(["seed", "model", "evidence_role"])))
    groupings.append(
        (
            "event",
            predictions.groupby([*group_columns, "event_key"]),
        )
    )
    if "observation_slice" in predictions:
        groupings.append(
            (
                "observation_slice",
                predictions.groupby([*group_columns, "observation_slice"], dropna=False),
            )
        )
    for scope, grouping in groupings:
        for keys, frame in grouping:
            keys = keys if isinstance(keys, tuple) else (keys,)
            identity = frame.iloc[0]
            for variant, column in (
                ("raw", "raw_pred_red_win_probability"),
                ("calibrated", "calibrated_pred_red_win_probability"),
            ):
                values = _metric_values(frame, column)
                for metric, (value, count) in values.items():
                    if variant == "calibrated" and not metric.startswith("winner_"):
                        continue
                    rows.append(
                        {
                            "scope": scope,
                            "evidence_role": identity["evidence_role"],
                            "seed": identity["seed"],
                            "model": identity["model"],
                            "test_week": (
                                identity["test_week"] if scope != "aggregate" else np.nan
                            ),
                            "event_key": identity.get("event_key") if scope == "event" else None,
                            "observation_slice": (
                                identity.get("observation_slice")
                                if scope == "observation_slice"
                                else None
                            ),
                            "probability_variant": (
                                variant if metric.startswith("winner_") else "not-applicable"
                            ),
                            "metric": metric,
                            "value": value,
                            "count": count,
                            "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                        }
                    )
    return pd.DataFrame(rows)


def reliability_calibration(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for keys, frame in predictions.groupby(["seed", "model", "test_week", "evidence_role"]):
        actual_red = frame["actual_red_total_score"].to_numpy(float)
        actual_blue = frame["actual_blue_total_score"].to_numpy(float)
        tie = actual_red == actual_blue
        outcome = (actual_red > actual_blue).astype(float)
        for variant, column in (
            ("raw", "raw_pred_red_win_probability"),
            ("calibrated", "calibrated_pred_red_win_probability"),
        ):
            probability = np.clip(frame[column].to_numpy(float), 1e-7, 1 - 1e-7)
            valid = ~tie & np.isfinite(probability)
            which = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, bins - 1)
            total = max(int(valid.sum()), 1)
            bin_rows: list[dict[str, Any]] = []
            for index in range(bins):
                selected = valid & (which == index)
                count = int(selected.sum())
                mean_probability = float(np.mean(probability[selected])) if count else np.nan
                observed_rate = float(np.mean(outcome[selected])) if count else np.nan
                bin_rows.append(
                    {
                        "seed": keys[0],
                        "model": keys[1],
                        "test_week": keys[2],
                        "evidence_role": keys[3],
                        "probability_variant": variant,
                        "bin_low": edges[index],
                        "bin_high": edges[index + 1],
                        "count": count,
                        "mean_probability": mean_probability,
                        "observed_rate": observed_rate,
                        "gap": mean_probability - observed_rate if count else np.nan,
                    }
                )
            ece = float(
                sum(row["count"] * abs(row["gap"]) for row in bin_rows if row["count"])
                / total
            )
            for row in bin_rows:
                row["expected_calibration_error"] = ece
            rows.extend(bin_rows)
    return pd.DataFrame(rows)


def _loss_frame(frame: pd.DataFrame) -> pd.DataFrame:
    actual_red = frame["actual_red_total_score"].to_numpy(float)
    actual_blue = frame["actual_blue_total_score"].to_numpy(float)
    predicted_red = frame["pred_red_total_score"].to_numpy(float)
    predicted_blue = frame["pred_blue_total_score"].to_numpy(float)
    outcome = (actual_red > actual_blue).astype(float)
    tie = actual_red == actual_blue
    probability = np.clip(frame["raw_pred_red_win_probability"].to_numpy(float), 1e-7, 1 - 1e-7)
    return pd.DataFrame(
        {
            "event_key": frame["event_key"].astype(str).to_numpy(),
            "match_key": frame["match_key"].astype(str).to_numpy(),
            "score_differential_squared_error": (
                (predicted_red - predicted_blue) - (actual_red - actual_blue)
            )
            ** 2,
            "winner_brier": np.where(tie, np.nan, (probability - outcome) ** 2),
            "winner_log_loss": np.where(
                tie,
                np.nan,
                -(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability)),
            ),
        }
    )


def hierarchical_seed_event_bootstrap(
    predictions: pd.DataFrame,
    *,
    resamples: int = 5_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Resample seeds, then event clusters within each primary test week."""

    source = predictions[
        predictions["prediction_level"].eq("seed")
        & predictions["test_week"].isin(PRIMARY_WEEKS)
    ]
    available_weeks = sorted(source["test_week"].dropna().astype(int).unique().tolist())
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    comparisons = (("teammate-set", "additive"), ("full-match", "teammate-set"))
    for candidate, baseline in comparisons:
        paired_by_seed_week: dict[tuple[int, int], pd.DataFrame] = {}
        for (seed_value, week), group in source.groupby(["seed", "test_week"]):
            candidate_frame = group[group["model"].eq(candidate)]
            baseline_frame = group[group["model"].eq(baseline)]
            left = _loss_frame(candidate_frame).rename(
                columns={metric: f"candidate_{metric}" for metric in PRACTICAL_TOLERANCES}
            )
            right = _loss_frame(baseline_frame).rename(
                columns={metric: f"baseline_{metric}" for metric in PRACTICAL_TOLERANCES}
            )
            paired_by_seed_week[(int(seed_value), int(week))] = left.merge(
                right, on=["event_key", "match_key"], validate="one_to_one"
            )
        seed_values = sorted({key[0] for key in paired_by_seed_week})
        if not seed_values:
            continue
        for metric in PRACTICAL_TOLERANCES:
            point_candidate = np.nanmean(
                np.concatenate(
                    [
                        frame[f"candidate_{metric}"].to_numpy(float)
                        for frame in paired_by_seed_week.values()
                    ]
                )
            )
            point_baseline = np.nanmean(
                np.concatenate(
                    [
                        frame[f"baseline_{metric}"].to_numpy(float)
                        for frame in paired_by_seed_week.values()
                    ]
                )
            )
            distribution = np.empty(resamples, dtype=float)
            for index in range(resamples):
                sampled_seed_values = rng.choice(seed_values, size=len(seed_values), replace=True)
                sampled_deltas: list[np.ndarray] = []
                for sampled_seed in sampled_seed_values:
                    for week in available_weeks:
                        frame = paired_by_seed_week.get((int(sampled_seed), int(week)))
                        if frame is None or frame.empty:
                            continue
                        events = frame["event_key"].drop_duplicates().to_numpy()
                        sampled_events = rng.choice(events, size=len(events), replace=True)
                        sampled = pd.concat(
                            [frame[frame["event_key"].eq(event)] for event in sampled_events],
                            ignore_index=True,
                        )
                        sampled_deltas.append(
                            sampled[f"candidate_{metric}"].to_numpy(float)
                            - sampled[f"baseline_{metric}"].to_numpy(float)
                        )
                distribution[index] = (
                    float(np.nanmean(np.concatenate(sampled_deltas)))
                    if sampled_deltas
                    else math.nan
                )
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "candidate_loss_mean": float(point_candidate),
                    "baseline_loss_mean": float(point_baseline),
                    "delta_mean": float(point_candidate - point_baseline),
                    "ci_low": float(np.nanquantile(distribution, 0.025)),
                    "ci_high": float(np.nanquantile(distribution, 0.975)),
                    "probability_of_improvement": float(np.mean(distribution < 0)),
                    "resamples": resamples,
                    "bootstrap_seed": seed,
                    "cluster": "seed-then-event_key",
                    "primary_weeks": json.dumps(available_weeks),
                }
            )
    return pd.DataFrame(rows)


def classify_interactions(
    hierarchical: pd.DataFrame, paired: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, baseline, component in (
        ("teammate-set", "additive", "teammate-interaction"),
        ("full-match", "teammate-set", "opponent-interaction"),
    ):
        comparison = hierarchical[
            hierarchical["candidate"].eq(candidate) & hierarchical["baseline"].eq(baseline)
        ]
        metric_support: list[str] = []
        harmful_metrics: list[str] = []
        for metric, tolerance in PRACTICAL_TOLERANCES.items():
            row = comparison[comparison["metric"].eq(metric)]
            if row.empty:
                continue
            item = row.iloc[0]
            seed_week = paired[
                paired["candidate"].eq(candidate)
                & paired["baseline"].eq(baseline)
                & paired["metric"].eq(metric)
                & paired["test_week"].isin(PRIMARY_WEEKS)
            ]
            favorable_both_weeks = all(
                int((seed_week[seed_week["test_week"].eq(week)]["delta_mean"] < 0).sum()) >= 2
                for week in PRIMARY_WEEKS
            )
            harmful_both_weeks = False
            if metric == "score_differential_squared_error":
                practical = math.sqrt(max(float(item["candidate_loss_mean"]), 0)) <= (
                    1 - tolerance
                ) * math.sqrt(max(float(item["baseline_loss_mean"]), 0))
                harmful_both_weeks = all(
                    int(
                        (
                            np.sqrt(
                                seed_week[seed_week["test_week"].eq(week)][
                                    "candidate_loss_mean"
                                ].clip(lower=0)
                            )
                            > (1 + tolerance)
                            * np.sqrt(
                                seed_week[seed_week["test_week"].eq(week)][
                                    "baseline_loss_mean"
                                ].clip(lower=0)
                            )
                        ).sum()
                    )
                    >= 2
                    for week in PRIMARY_WEEKS
                )
            else:
                practical = float(item["delta_mean"]) <= -tolerance
                harmful_both_weeks = all(
                    int(
                        (
                            seed_week[seed_week["test_week"].eq(week)]["delta_mean"]
                            > tolerance
                        ).sum()
                    )
                    >= 2
                    for week in PRIMARY_WEEKS
                )
            if (
                float(item["probability_of_improvement"]) >= 0.90
                and favorable_both_weeks
                and practical
            ):
                metric_support.append(metric)
            if harmful_both_weeks:
                harmful_metrics.append(metric)
        classification = (
            "supported"
            if metric_support and not harmful_metrics
            else ("harmful" if harmful_metrics else "uncertain")
        )
        rows.append(
            {
                "component": component,
                "candidate": candidate,
                "baseline": baseline,
                "classification": classification,
                "supporting_metrics": json.dumps(metric_support),
                "harmful_metrics": json.dumps(harmful_metrics),
                "formal_weeks": json.dumps(list(PRIMARY_WEEKS)),
                "stress_week": STRESS_WEEK,
                "provisional": True,
            }
        )
    return pd.DataFrame(rows)


def external_statbotics_comparison(
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    resamples: int = 5_000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare the best internally supported ensemble with Statbotics."""

    columns = [
        "candidate",
        "baseline",
        "metric",
        "candidate_loss_mean",
        "baseline_loss_mean",
        "delta_mean",
        "ci_low",
        "ci_high",
        "probability_of_improvement",
        "resamples",
        "bootstrap_seed",
        "cluster",
    ]
    if not predictions["model"].eq("statbotics").any():
        return pd.DataFrame(columns=columns), {
            "status": "awaiting-statbotics",
            "candidate": None,
            "verdict": "awaiting-statbotics",
        }
    decision_map = decisions.set_index("component")["classification"].to_dict()
    candidate = (
        "full-match"
        if decision_map.get("opponent-interaction") == "supported"
        else (
            "teammate-set"
            if decision_map.get("teammate-interaction") == "supported"
            else "additive"
        )
    )
    candidate_frame = predictions[
        predictions["prediction_level"].eq("ensemble")
        & predictions["model"].eq(candidate)
        & predictions["test_week"].isin(PRIMARY_WEEKS)
    ]
    baseline_frame = predictions[
        predictions["model"].eq("statbotics")
        & predictions["test_week"].isin(PRIMARY_WEEKS)
    ]
    left = _loss_frame(candidate_frame).rename(
        columns={metric: f"candidate_{metric}" for metric in PRACTICAL_TOLERANCES}
    )
    right = _loss_frame(baseline_frame).rename(
        columns={metric: f"baseline_{metric}" for metric in PRACTICAL_TOLERANCES}
    )
    paired = left.merge(right, on=["event_key", "match_key"], validate="one_to_one")
    if paired.empty:
        raise ValueError("The ensemble and Statbotics primary subsets do not overlap.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for metric in PRACTICAL_TOLERANCES:
        candidate_loss = paired[f"candidate_{metric}"].to_numpy(float)
        baseline_loss = paired[f"baseline_{metric}"].to_numpy(float)
        distribution = np.empty(resamples, dtype=float)
        for index in range(resamples):
            sampled_deltas: list[np.ndarray] = []
            for week in PRIMARY_WEEKS:
                week_keys = set(
                    candidate_frame[candidate_frame["test_week"].eq(week)]["match_key"].astype(
                        str
                    )
                )
                week_frame = paired[paired["match_key"].astype(str).isin(week_keys)]
                events = week_frame["event_key"].drop_duplicates().to_numpy()
                sampled_events = rng.choice(events, size=len(events), replace=True)
                sampled = pd.concat(
                    [week_frame[week_frame["event_key"].eq(event)] for event in sampled_events],
                    ignore_index=True,
                )
                sampled_deltas.append(
                    sampled[f"candidate_{metric}"].to_numpy(float)
                    - sampled[f"baseline_{metric}"].to_numpy(float)
                )
            distribution[index] = float(np.nanmean(np.concatenate(sampled_deltas)))
        rows.append(
            {
                "candidate": candidate,
                "baseline": "statbotics",
                "metric": metric,
                "candidate_loss_mean": float(np.nanmean(candidate_loss)),
                "baseline_loss_mean": float(np.nanmean(baseline_loss)),
                "delta_mean": float(np.nanmean(candidate_loss - baseline_loss)),
                "ci_low": float(np.nanquantile(distribution, 0.025)),
                "ci_high": float(np.nanquantile(distribution, 0.975)),
                "probability_of_improvement": float(np.mean(distribution < 0)),
                "resamples": resamples,
                "bootstrap_seed": seed,
                "cluster": "test_week-then-event_key",
            }
        )
    comparison = pd.DataFrame(rows, columns=columns)
    improvement = []
    harmful = []
    noninferior = []
    for row in comparison.itertuples():
        tolerance = PRACTICAL_TOLERANCES[row.metric]
        if row.metric == "score_differential_squared_error":
            candidate_rmse = math.sqrt(max(row.candidate_loss_mean, 0))
            baseline_rmse = math.sqrt(max(row.baseline_loss_mean, 0))
            practical_improvement = candidate_rmse <= (1 - tolerance) * baseline_rmse
            practical_harm = candidate_rmse > (1 + tolerance) * baseline_rmse
            within_tolerance = candidate_rmse <= (1 + tolerance) * baseline_rmse
        else:
            practical_improvement = row.delta_mean <= -tolerance
            practical_harm = row.delta_mean > tolerance
            within_tolerance = row.delta_mean <= tolerance
        improvement.append(row.probability_of_improvement >= 0.80 and practical_improvement)
        harmful.append(practical_harm)
        noninferior.append(within_tolerance)
    verdict = (
        "competitive"
        if any(improvement) and all(noninferior)
        else ("not-competitive" if sum(harmful) >= 2 else "mixed")
    )
    return comparison, {"status": "complete", "candidate": candidate, "verdict": verdict}


def _seed_event_bootstrap(predictions: pd.DataFrame, *, resamples: int) -> pd.DataFrame:
    rows = []
    for seed_value, frame in predictions[
        predictions["prediction_level"].eq("seed")
    ].groupby("seed"):
        raw = frame.copy()
        raw["pred_red_win_probability"] = raw["raw_pred_red_win_probability"]
        result = paired_event_bootstrap(raw, resamples=resamples, seed=2026)
        result.insert(0, "experiment_seed", seed_value)
        result["probability_variant"] = "raw"
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    seed_rows = predictions[predictions["prediction_level"].eq("seed")]
    frames = []
    for _, group in seed_rows.groupby(["model", "test_week", "match_key"], sort=False):
        row = group.iloc[0].copy()
        row["seed"] = "ensemble"
        row["prediction_level"] = "ensemble"
        for column in (
            "pred_red_total_score",
            "pred_blue_total_score",
            "raw_pred_red_win_probability",
            "calibrated_pred_red_win_probability",
        ):
            row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        row["pred_red_win_probability"] = row["calibrated_pred_red_win_probability"]
        row["calibration_method"] = "mean-of-seed-level-probabilities"
        frames.append(row)
    return pd.DataFrame(frames)


def _parameter_rows(
    result: FeatureTrainingResult,
    *,
    seed: int,
    architecture: str,
    fold_number: int,
    test_week: int,
) -> list[dict[str, Any]]:
    final = result.history.iloc[-1]
    base = {
        "seed": seed,
        "model": architecture,
        "fold_number": fold_number,
        "test_week": test_week,
        "active_z_base_rows": int(final["active_team_rows"]),
    }
    rows = [
        {
            **base,
            "module": "<all>",
            "nominal_parameters": int(final["nominal_parameter_count"]),
            "trainable_parameters": int(final["trainable_parameter_count"]),
            "gradient_receiving_parameters": int(
                final["gradient_receiving_parameter_count"]
            ),
        }
    ]
    suffix = "_nominal_parameter_count"
    modules = {
        str(column)[len("module_") : -len(suffix)]
        for column in final.index
        if str(column).startswith("module_") and str(column).endswith(suffix)
    }
    for module in sorted(modules):
        rows.append(
            {
                **base,
                "module": module,
                "nominal_parameters": int(final[f"module_{module}_nominal_parameter_count"]),
                "trainable_parameters": int(final[f"module_{module}_trainable_parameter_count"]),
                "gradient_receiving_parameters": int(
                    final[f"module_{module}_gradient_receiving_parameter_count"]
                ),
            }
        )
    return rows


def _initial_manifest(
    output: Path,
    *,
    sources: dict[str, Path],
    tests: Sequence[int],
    seeds: Sequence[int],
    smoke: bool,
) -> Path:
    return write_artifact_manifest(
        output,
        workflow=RELIABILITY_WORKFLOW,
        artifact_version="v1",
        resolved_config={
            "study_id": output.name,
            "artifact_schema": RELIABILITY_SCHEMA,
            "season": 2026,
            "state_model": "static-z-base",
            "complete": False,
            "conclusion_ready": False,
            "stage": "development",
            "smoke": smoke,
            "test_weeks": list(tests),
            "seeds": list(seeds),
        },
        source_files=sources,
        promotion_eligible=False,
        promotion_note="Incomplete reliability matrix; conclusions are suppressed.",
    )


def _write_development_rejection(
    output: Path,
    *,
    sources: dict[str, Path],
    tests: Sequence[int],
    seeds: Sequence[int],
    reason_code: str,
    reason: str,
    budget_minutes: float,
    selected_clip: float | None = None,
) -> Path:
    rejection = {
        "status": reason_code,
        "selected_max_grad_norm": selected_clip,
        "eligible_epoch_counts": [],
        "budget_seconds": budget_minutes * 60,
        "complete": False,
        "conclusion_ready": False,
        "promotion_eligible": False,
        "reason": reason,
    }
    (output / "development_selection.json").write_text(
        json.dumps(rejection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return write_artifact_manifest(
        output,
        workflow=RELIABILITY_WORKFLOW,
        artifact_version="v1",
        resolved_config={
            "study_id": output.name,
            "artifact_schema": RELIABILITY_SCHEMA,
            "season": 2026,
            "state_model": "static-z-base",
            "latent_dim": 16,
            "architectures": list(ARCHITECTURES),
            "initialization": "pre-2026-prior",
            "development_week": 4,
            "stage": "development-rejected",
            "rejection": rejection,
            "seeds": list(seeds),
            "test_weeks": list(tests),
            "primary_weeks": list(PRIMARY_WEEKS),
            "stress_week": STRESS_WEEK,
            "known_as_of_policy": "development selection uses completed matches through week 3",
            "complete": False,
            "conclusion_ready": False,
            "promotion_eligible": False,
        },
        source_files=sources,
        promotion_eligible=False,
        promotion_note="Development gate rejected the matrix; no test evidence was produced.",
    )


def _add_statbotics_rows(
    predictions: pd.DataFrame, statbotics_path: Path | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    without_stat = predictions[predictions["model"].ne("statbotics")].copy()
    if statbotics_path is None or not statbotics_path.exists():
        return without_stat, {"status": "not-provided"}
    statbotics = validate_statbotics_prediction_table(
        pd.read_parquet(statbotics_path, engine="pyarrow"), 2026
    )
    truth = without_stat[
        without_stat["prediction_level"].eq("ensemble")
        & without_stat["model"].eq("full-match")
    ].drop_duplicates("match_key")
    candidate_keys = set(truth["match_key"].astype(str))
    baseline_keys = set(statbotics["match_key"].astype(str))
    joined = truth.merge(statbotics, on="match_key", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("Reliability predictions and Statbotics have no matching match keys.")
    joined["model"] = "statbotics"
    joined["architecture"] = "external-control"
    joined["seed"] = "control"
    joined["prediction_level"] = "control"
    joined["pred_red_total_score"] = joined["statbotics_pred_red_score"]
    joined["pred_blue_total_score"] = joined["statbotics_pred_blue_score"]
    joined["raw_pred_red_win_probability"] = joined[
        "statbotics_pred_red_win_probability"
    ]
    joined["calibrated_pred_red_win_probability"] = joined[
        "statbotics_pred_red_win_probability"
    ]
    joined["pred_red_win_probability"] = joined["statbotics_pred_red_win_probability"]
    joined["calibration_method"] = "statbotics-pre-match-pred"
    joined["calibration_source_weeks"] = "[]"
    joined["calibration_source_rows"] = 0
    joined["evidence_role"] = np.where(
        joined["test_week"].eq(STRESS_WEEK), "championship-stress", "primary"
    )
    stat_rows = joined[without_stat.columns]
    return (
        pd.concat([without_stat, stat_rows], ignore_index=True),
        {
            "status": "provided",
            "candidate_rows": len(candidate_keys),
            "statbotics_rows": len(baseline_keys),
            "matched_rows": len(candidate_keys & baseline_keys),
            "candidate_only_rows": len(candidate_keys - baseline_keys),
            "statbotics_only_rows": len(baseline_keys - candidate_keys),
        },
    )


def _write_evaluation(
    output: Path,
    predictions: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    statbotics_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    prediction_table, statbotics_coverage = _add_statbotics_rows(predictions, statbotics_path)
    # Parquet requires one stable physical type; neural seeds, controls, and ensembles share this
    # identifier column by design.
    prediction_table["seed"] = prediction_table["seed"].astype(str)
    metrics = reliability_metrics(prediction_table)
    calibration = reliability_calibration(prediction_table)
    paired = _seed_event_bootstrap(prediction_table, resamples=bootstrap_resamples)
    hierarchical = hierarchical_seed_event_bootstrap(
        prediction_table, resamples=bootstrap_resamples, seed=2026
    )
    decisions = classify_interactions(hierarchical, paired)
    external, external_verdict = external_statbotics_comparison(
        prediction_table, decisions, resamples=bootstrap_resamples, seed=2026
    )
    conclusion_status = (
        "ready" if statbotics_coverage["status"] == "provided" else "awaiting-statbotics"
    )
    decisions["provisional"] = conclusion_status != "ready"
    prediction_table.to_parquet(output / "walk_forward_predictions.parquet", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    paired.to_csv(output / "paired_bootstrap.csv", index=False)
    hierarchical.to_csv(output / "hierarchical_bootstrap.csv", index=False)
    decisions.to_csv(output / "interaction_decisions.csv", index=False)
    external.to_csv(output / "external_comparison.csv", index=False)
    (output / "external_verdict.json").write_text(
        json.dumps(external_verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics, statbotics_coverage, conclusion_status


def run_reliability_study(
    features_path: str | Path,
    event_metadata_path: str | Path,
    output_dir: str | Path,
    *,
    prior_checkpoint: str | Path,
    sidecar_paths: dict[str, str | Path] | None = None,
    statbotics_path: str | Path | None = None,
    tensorboard_root: str | Path | None = "runs/reference-2026-static-reliability",
    max_epochs: int = 40,
    budget_minutes: float = 240,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    test_weeks: Iterable[int] = (*PRIMARY_WEEKS, STRESS_WEEK),
    bootstrap_resamples: int = 5_000,
    smoke: bool = False,
) -> ReliabilityStudyResult:
    """Run the development-frozen multi-seed static architecture experiment."""

    started = time.perf_counter()
    features_path = Path(features_path)
    event_metadata_path = Path(event_metadata_path)
    prior_path = Path(prior_checkpoint)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table, filtering = validate_reference_rows(
        read_feature_table(features_path), _read_table(event_metadata_path)
    )
    sidecar_paths = sidecar_paths or {}
    raw_sidecars = {name: _read_table(path) for name, path in sidecar_paths.items()}
    sidecars = _sidecars_with_weeks(
        {name: frame for name, frame in raw_sidecars.items() if frame is not None}, table
    )
    tests = tuple(int(week) for week in test_weeks)
    seed_values = tuple(int(value) for value in seeds)
    if smoke:
        tests = tests[:1]
        seed_values = seed_values[:1]
        max_epochs = min(max_epochs, 2)
        bootstrap_resamples = min(bootstrap_resamples, 100)
        clip_candidates = (5.0,)
        epoch_candidates = (max_epochs,)
    else:
        clip_candidates = CLIP_CANDIDATES
        epoch_candidates = tuple(value for value in EPOCH_CANDIDATES if value <= max_epochs)
    if not epoch_candidates:
        raise ValueError("max_epochs does not permit any predeclared epoch candidate.")
    sources = {
        "features": features_path,
        "event_metadata": event_metadata_path,
        "prior_checkpoint": prior_path,
        **{name: Path(path) for name, path in sidecar_paths.items()},
    }
    if statbotics_path is not None and Path(statbotics_path).exists():
        sources["statbotics"] = Path(statbotics_path)
    manifest_path = _initial_manifest(
        output, sources=sources, tests=tests, seeds=seed_values, smoke=smoke
    )
    for week in tests:
        rows = table[pd.to_numeric(table["event_week"], errors="coerce").eq(week)]
        if rows.empty:
            raise ValueError(f"Test week {week} has no played official matches.")
        if not smoke and (len(rows) < 500 or rows["event_key"].nunique() < 5):
            raise ValueError(f"Test week {week} fails the 500-match/five-event gate.")
    tensorboard_dir = Path(tensorboard_root) / output.name if tensorboard_root else None
    development_split = nested_temporal_split(table, train_max_week=3, holdout_week=4)
    base_opts = default_options(2026).model_copy(
        update={
            "state_model": "static-z-base",
            "latent_dim": 16,
            "device": "cpu",
            "use_early_stopping": False,
            "restore_best_validation_model": False,
        }
    )
    if not smoke:
        pilot_epochs = min(2, max_epochs)
        pilot_opts = base_opts.model_copy(
            update={
                "match_architecture": "additive",
                "random_seed": 2026,
                "epochs": pilot_epochs,
                "max_grad_norm": 5.0,
            }
        )
        pilot = _train_once(
            table,
            pilot_opts,
            development_split,
            prior_checkpoint=prior_path,
            sidecars=_filter_sidecars(sidecars, max_week=3),
            logdir=(
                tensorboard_dir / "development" / "timing-pilot"
                if tensorboard_dir
                else None
            ),
            contract={
                "study_id": output.name,
                "artifact_schema": RELIABILITY_SCHEMA,
                "architecture": "additive",
                "initialization": "pre-2026-prior",
                "seed": 2026,
                "train_max_week": 3,
                "holdout_week": 4,
                "phase": "timing-pilot",
                "known_as_of": "end-of-week-3",
                "options": pilot_opts.model_dump(mode="json"),
            },
        )
        pilot_history = pilot.history.copy()
        pilot_history.to_csv(output / "timing_pilot.csv", index=False)
        seconds_per_epoch = float(pilot_history["epoch_seconds"].mean())
        lower_bound_seconds = minimum_campaign_seconds(
            seconds_per_epoch,
            development_epochs=max_epochs,
            confirmation_epochs=min(epoch_candidates),
            final_epochs=min(epoch_candidates),
            clip_count=len(clip_candidates),
            architecture_count=len(ARCHITECTURES),
            seed_count=len(seed_values),
            fold_count=len(tests),
        )
        preflight = {
            "pilot_epochs": pilot_epochs,
            "seconds_per_epoch": seconds_per_epoch,
            "minimum_buffered_campaign_seconds": lower_bound_seconds,
            "budget_seconds": budget_minutes * 60,
            "budget_passed": lower_bound_seconds <= budget_minutes * 60,
            "estimate_kind": "conservative-lower-bound",
        }
        (output / "timing_preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not preflight["budget_passed"]:
            raise RuntimeError(
                "The two-epoch timing pilot shows that even the conservative lower-bound "
                f"campaign estimate ({lower_bound_seconds / 60:.1f} minutes) exceeds the "
                f"{budget_minutes:.1f}-minute CPU budget. No development grid was started."
            )
    optimization_histories: list[pd.DataFrame] = []
    development_started = time.perf_counter()
    for threshold in clip_candidates:
        for architecture in ARCHITECTURES:
            opts = base_opts.model_copy(
                update={
                    "match_architecture": architecture,
                    "random_seed": 2026,
                    "epochs": max_epochs,
                    "max_grad_norm": float(threshold),
                }
            )
            contract = {
                "study_id": output.name,
                "artifact_schema": RELIABILITY_SCHEMA,
                "architecture": architecture,
                "initialization": "pre-2026-prior",
                "seed": 2026,
                "train_max_week": 3,
                "holdout_week": 4,
                "phase": "development-grid",
                "known_as_of": "end-of-week-3",
                "max_grad_norm_candidate": float(threshold),
                "options": opts.model_dump(mode="json"),
            }
            result = _train_once(
                table,
                opts,
                development_split,
                prior_checkpoint=prior_path,
                sidecars=_filter_sidecars(sidecars, max_week=3),
                logdir=(
                    tensorboard_dir
                    / "development"
                    / architecture
                    / f"clip_{str(threshold).replace('.', '_')}"
                    if tensorboard_dir
                    else None
                ),
                contract=contract,
            )
            history = result.history.copy()
            history.insert(0, "max_grad_norm", float(threshold))
            history.insert(0, "architecture", architecture)
            optimization_histories.append(history)
    optimization_grid = pd.concat(optimization_histories, ignore_index=True)
    # Persist development evidence before any mechanical selection gate can reject the study.
    optimization_grid.to_csv(output / "optimization_grid.csv", index=False)
    try:
        selected_clip = (
            float(clip_candidates[0])
            if smoke
            else select_gradient_threshold(
                optimization_grid,
                candidates=clip_candidates,
                warmup_epochs=min(5, max_epochs),
            )
        )
    except RuntimeError as exc:
        _write_development_rejection(
            output,
            sources=sources,
            tests=tests,
            seeds=seed_values,
            reason_code="rejected-no-gradient-threshold",
            reason=str(exc),
            budget_minutes=budget_minutes,
        )
        raise
    eligible_epochs = (
        (max_epochs,)
        if smoke
        else eligible_epoch_counts(
            optimization_grid,
            selected_max_grad_norm=selected_clip,
            candidates=epoch_candidates,
        )
    )
    if not eligible_epochs:
        _write_development_rejection(
            output,
            sources=sources,
            tests=tests,
            seeds=seed_values,
            reason_code="rejected-no-common-epoch",
            reason=(
                "No common epoch count satisfied the predeclared development metric tolerances."
            ),
            budget_minutes=budget_minutes,
            selected_clip=selected_clip,
        )
        raise RuntimeError("No common epoch count satisfies the development metric tolerances.")
    confirmation_frames: list[pd.DataFrame] = []
    confirmation_results: dict[str, FeatureTrainingResult] = {}
    selected_epochs: int | None = None
    for candidate_epochs in eligible_epochs:
        trial_frames = []
        trial_results = {}
        for architecture in ARCHITECTURES:
            opts = base_opts.model_copy(
                update={
                    "match_architecture": architecture,
                    "random_seed": 2026,
                    "epochs": int(candidate_epochs),
                    "max_grad_norm": selected_clip,
                }
            )
            contract = {
                "study_id": output.name,
                "artifact_schema": RELIABILITY_SCHEMA,
                "architecture": architecture,
                "initialization": "pre-2026-prior",
                "seed": 2026,
                "train_max_week": 3,
                "holdout_week": 4,
                "phase": "development-confirmation",
                "known_as_of": "end-of-week-3",
                "selected_max_grad_norm": selected_clip,
                "candidate_epoch_count": int(candidate_epochs),
                "options": opts.model_dump(mode="json"),
            }
            result = _train_once(
                table,
                opts,
                development_split,
                prior_checkpoint=prior_path,
                sidecars=_filter_sidecars(sidecars, max_week=3),
                logdir=(
                    tensorboard_dir
                    / "development"
                    / "confirmation"
                    / architecture
                    / f"epochs_{candidate_epochs}"
                    if tensorboard_dir
                    else None
                ),
                contract=contract,
            )
            history = result.history.copy()
            history.insert(0, "candidate_epochs", int(candidate_epochs))
            history.insert(0, "max_grad_norm", selected_clip)
            history.insert(0, "architecture", architecture)
            trial_frames.append(history)
            trial_results[architecture] = result
        trial = pd.concat(trial_frames, ignore_index=True).copy()
        trial["confirmation_passed"] = smoke or confirmation_passes(
            trial, optimization_grid, selected_max_grad_norm=selected_clip
        )
        confirmation_frames.append(trial)
        if bool(trial["confirmation_passed"].iloc[0]):
            selected_epochs = int(candidate_epochs)
            confirmation_results = trial_results
            break
    confirmation = pd.concat(confirmation_frames, ignore_index=True)
    if selected_epochs is None:
        raise RuntimeError("No development confirmation run satisfied the frozen-config gates.")
    development_elapsed = time.perf_counter() - development_started
    final_seconds = 0.0
    development_rows = max(int(development_split.train_mask.sum()), 1)
    for _architecture, result in confirmation_results.items():
        seconds_per_epoch = float(result.history["epoch_seconds"].mean())
        for week in tests:
            train_rows = int(
                pd.to_numeric(table["event_week"], errors="coerce").le(week - 1).sum()
            )
            final_seconds += (
                seconds_per_epoch
                * selected_epochs
                * max(train_rows / development_rows, 1.0)
                * len(seed_values)
            )
    buffered_seconds = 1.20 * (development_elapsed + final_seconds)
    selection = {
        "selected_max_grad_norm": selected_clip,
        "selected_epochs": selected_epochs,
        "eligible_epoch_counts": list(eligible_epochs),
        "development_elapsed_seconds": development_elapsed,
        "estimated_final_training_seconds": final_seconds,
        "buffered_total_seconds": buffered_seconds,
        "budget_seconds": budget_minutes * 60,
        "budget_passed": smoke or buffered_seconds <= budget_minutes * 60,
    }
    confirmation.to_csv(output / "development_confirmation.csv", index=False)
    (output / "development_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not selection["budget_passed"]:
        raise RuntimeError(
            "The measured development runs predict that the fixed three-seed matrix cannot fit "
            f"within {budget_minutes:.1f} minutes with the required 20% safety margin."
        )

    predictions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    auxiliary_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    calibration_banks: dict[tuple[int, str], list[pd.DataFrame]] = {
        (seed_value, architecture): []
        for seed_value in seed_values
        for architecture in ARCHITECTURES
    }
    baseline_weeks: set[int] = set()
    for seed_value in seed_values:
        for architecture in ARCHITECTURES:
            for fold_number, test_week in enumerate(tests, start=1):
                opts = base_opts.model_copy(
                    update={
                        "match_architecture": architecture,
                        "random_seed": seed_value,
                        "epochs": selected_epochs,
                        "max_grad_norm": selected_clip,
                    }
                )
                prediction_split = nested_temporal_split(
                    table, train_max_week=test_week - 1, holdout_week=test_week
                )
                refit_split = refit_temporal_split(table, train_max_week=test_week - 1)
                role = "championship-stress" if test_week == STRESS_WEEK else "primary"
                contract = {
                    "study_id": output.name,
                    "artifact_schema": RELIABILITY_SCHEMA,
                    "architecture": architecture,
                    "initialization": "pre-2026-prior",
                    "seed": seed_value,
                    "train_max_week": test_week - 1,
                    "holdout_week": test_week,
                    "test_week": test_week,
                    "phase": "final-fixed-duration",
                    "evidence_role": role,
                    "known_as_of": f"end-of-week-{test_week - 1}",
                    "selected_on_development_week": 4,
                    "selected_max_grad_norm": selected_clip,
                    "selected_epoch_count": selected_epochs,
                    "options": opts.model_dump(mode="json"),
                }
                result = _train_once(
                    table,
                    opts,
                    refit_split,
                    prior_checkpoint=prior_path,
                    sidecars=_filter_sidecars(sidecars, max_week=test_week - 1),
                    logdir=(
                        tensorboard_dir
                        / f"seed_{seed_value}"
                        / architecture
                        / f"fold_{test_week}"
                        if tensorboard_dir
                        else None
                    ),
                    contract=contract,
                )
                raw = _prediction_frame(
                    result,
                    prediction_split,
                    opts,
                    model_name=architecture,
                    initialization="pre-2026-prior",
                    fold_number=fold_number,
                    test_week=test_week,
                    train_max_week=test_week - 1,
                    phase="test",
                )
                raw["seed"] = seed_value
                raw["prediction_level"] = "seed"
                raw["evidence_role"] = role
                bank = calibration_banks[(seed_value, architecture)]
                prior = pd.concat(bank, ignore_index=True) if bank else pd.DataFrame()
                calibrator = fit_sequential_calibrator(prior)
                calibrated = apply_sequential_calibrator(raw, calibrator)
                predictions.append(calibrated)
                bank.append(calibrated.copy())
                if seed_value == seed_values[0] and architecture == "full-match" and (
                    test_week not in baseline_weeks
                ):
                    for control in _baseline_frames(
                        result,
                        prediction_split,
                        calibrated,
                        initialization="pre-2026-prior",
                        fold_number=fold_number,
                        test_week=test_week,
                    ):
                        control["seed"] = "control"
                        control["prediction_level"] = "control"
                        control["evidence_role"] = role
                        control["raw_pred_red_win_probability"] = control[
                            "pred_red_win_probability"
                        ]
                        control["calibrated_pred_red_win_probability"] = control[
                            "pred_red_win_probability"
                        ]
                        control["calibration_method"] = "control-native"
                        control["calibration_source_weeks"] = "[]"
                        control["calibration_source_rows"] = 0
                        predictions.append(control)
                    baseline_weeks.add(test_week)
                auxiliary = _sidecar_validation_metrics(
                    result, _filter_sidecars(sidecars, exact_week=test_week), opts
                )
                auxiliary_rows.extend(
                    {
                        "scope": "test",
                        "seed": seed_value,
                        "model": architecture,
                        "fold_number": fold_number,
                        "test_week": test_week,
                        "evidence_role": role,
                        "metric": metric,
                        "value": value,
                        "direction": "lower",
                    }
                    for metric, value in auxiliary.items()
                )
                history = result.history.copy()
                history.insert(0, "phase", "final-fixed-duration")
                history.insert(0, "test_week", test_week)
                history.insert(0, "fold_number", fold_number)
                history.insert(0, "model", architecture)
                history.insert(0, "seed", seed_value)
                histories.append(history)
                parameter_rows.extend(
                    _parameter_rows(
                        result,
                        seed=seed_value,
                        architecture=architecture,
                        fold_number=fold_number,
                        test_week=test_week,
                    )
                )
                checkpoint = (
                    output
                    / "checkpoints"
                    / f"seed_{seed_value}"
                    / architecture
                    / f"fold_{test_week}.pt"
                )
                _write_checkpoint(
                    result,
                    checkpoint,
                    architecture=architecture,
                    initialization="pre-2026-prior",
                    fold=fold_number,
                    test_week=test_week,
                    calibrator=calibrator,
                    study_contract=RELIABILITY_SCHEMA,
                    seed=seed_value,
                    extra_payload={
                        "known_as_of": f"end-of-week-{test_week - 1}",
                        "evidence_role": role,
                        "development_selection": selection,
                    },
                )
    seed_predictions = pd.concat(predictions, ignore_index=True)
    ensemble = _ensemble_predictions(seed_predictions)
    all_predictions = pd.concat([seed_predictions, ensemble], ignore_index=True)
    metrics, statbotics_coverage, conclusion_status = _write_evaluation(
        output,
        all_predictions,
        bootstrap_resamples=bootstrap_resamples,
        statbotics_path=Path(statbotics_path) if statbotics_path is not None else None,
    )
    pd.DataFrame(auxiliary_rows).to_csv(output / "auxiliary_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / "training_history.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "parameter_utilization.csv", index=False)
    complete = (
        not smoke
        and set(tests) == {*PRIMARY_WEEKS, STRESS_WEEK}
        and set(seed_values) == set(DEFAULT_SEEDS)
        and all(
            (
                (all_predictions["prediction_level"].eq("seed"))
                & all_predictions["seed"].eq(seed_value)
                & all_predictions["model"].eq(architecture)
                & all_predictions["test_week"].eq(week)
            ).any()
            for seed_value in DEFAULT_SEEDS
            for architecture in ARCHITECTURES
            for week in (*PRIMARY_WEEKS, STRESS_WEEK)
        )
    )
    coverage = {
        **filtering,
        "complete": complete,
        "conclusion_ready": complete and conclusion_status == "ready",
        "conclusion_status": conclusion_status,
        "prediction_rows": int(len(all_predictions)),
        "neural_seed_rows": int(all_predictions["prediction_level"].eq("seed").sum()),
        "ensemble_rows": int(all_predictions["prediction_level"].eq("ensemble").sum()),
        "test_match_rows": int(all_predictions["match_key"].nunique()),
        "seeds": list(seed_values),
        "test_weeks": list(tests),
        "primary_weeks": list(PRIMARY_WEEKS),
        "stress_week": STRESS_WEEK,
        "statbotics": statbotics_coverage,
    }
    (output / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = write_artifact_manifest(
        output,
        workflow=RELIABILITY_WORKFLOW,
        artifact_version="v1",
        resolved_config={
            "study_id": output.name,
            "artifact_schema": RELIABILITY_SCHEMA,
            "season": 2026,
            "state_model": "static-z-base",
            "latent_dim": 16,
            "architectures": list(ARCHITECTURES),
            "initialization": "pre-2026-prior",
            "development_week": 4,
            "selected_max_grad_norm": selected_clip,
            "selected_epochs": selected_epochs,
            "seeds": list(seed_values),
            "test_weeks": list(tests),
            "primary_weeks": list(PRIMARY_WEEKS),
            "stress_week": STRESS_WEEK,
            "cpu_budget_minutes": budget_minutes,
            "known_as_of_policy": (
                "select optimizer and duration on week 4; train fixed duration through T-1; "
                "freeze for T"
            ),
            "probability_evidence": "raw-primary; prior-heldout-sequential-calibration-secondary",
            "complete": complete,
            "conclusion_ready": complete and conclusion_status == "ready",
            "conclusion_status": conclusion_status,
            "smoke": smoke,
            "promotion_eligible": False,
            "elapsed_seconds": time.perf_counter() - started,
            "resolved_options": base_opts.model_dump(mode="json"),
        },
        source_files=sources,
        promotion_eligible=False,
        promotion_note=(
            "One-season static architecture reliability evidence; production promotion remains "
            "ineligible."
        ),
    )
    return ReliabilityStudyResult(
        output,
        manifest_path,
        metrics,
        all_predictions,
        selected_clip,
        selected_epochs,
        conclusion_status,
        tensorboard_dir,
    )


def finalize_reliability_statbotics(
    study_dir: str | Path,
    statbotics_path: str | Path,
    *,
    bootstrap_resamples: int = 5_000,
) -> Path:
    """Finalize external evidence from saved predictions without loading a PyTorch model."""

    output = Path(study_dir)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("workflow") != RELIABILITY_WORKFLOW:
        raise ValueError("Expected a static reliability study manifest.")
    predictions = pd.read_parquet(output / "walk_forward_predictions.parquet", engine="pyarrow")
    predictions = predictions[predictions["model"].ne("statbotics")]
    _, statbotics_coverage, conclusion_status = _write_evaluation(
        output,
        predictions,
        bootstrap_resamples=bootstrap_resamples,
        statbotics_path=Path(statbotics_path),
    )
    coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
    coverage["statbotics"] = statbotics_coverage
    coverage["conclusion_status"] = conclusion_status
    coverage["conclusion_ready"] = bool(coverage.get("complete"))
    (output / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resolved = dict(manifest.get("resolved_config") or {})
    resolved["conclusion_status"] = conclusion_status
    resolved["conclusion_ready"] = bool(resolved.get("complete"))
    source_files: dict[str, Path] = {}
    for name, value in (manifest.get("sources") or {}).items():
        if isinstance(value, dict) and value.get("path"):
            source_files[name] = Path(str(value["path"]))
    source_files["statbotics"] = Path(statbotics_path)
    return write_artifact_manifest(
        output,
        workflow=RELIABILITY_WORKFLOW,
        artifact_version="v1",
        resolved_config=resolved,
        source_files=source_files,
        promotion_eligible=False,
        promotion_note=(
            "One-season static architecture reliability evidence; production promotion remains "
            "ineligible."
        ),
    )


__all__ = [
    "CALIBRATION_MIN_ROWS",
    "CLIP_CANDIDATES",
    "DEFAULT_SEEDS",
    "EPOCH_CANDIDATES",
    "PRIMARY_WEEKS",
    "RELIABILITY_SCHEMA",
    "RELIABILITY_WORKFLOW",
    "ReliabilityStudyResult",
    "apply_sequential_calibrator",
    "classify_interactions",
    "confirmation_passes",
    "eligible_epoch_counts",
    "finalize_reliability_statbotics",
    "fit_sequential_calibrator",
    "hierarchical_seed_event_bootstrap",
    "minimum_campaign_seconds",
    "reliability_calibration",
    "reliability_metrics",
    "run_reliability_study",
    "select_gradient_threshold",
]
