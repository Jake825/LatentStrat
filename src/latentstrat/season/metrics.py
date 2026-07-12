"""Pure NumPy/Pandas evaluation of saved match-prediction artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latentstrat.artifacts import write_artifact_manifest
from latentstrat.season.statbotics_baseline import validate_statbotics_prediction_table

CANDIDATE_REQUIRED_COLUMNS = (
    "event_key",
    "match_key",
    "pred_red_total_score",
    "actual_red_total_score",
    "pred_blue_total_score",
    "actual_blue_total_score",
    "pred_red_win_probability",
)
PROBABILITY_EPSILON = 1e-7


@dataclass(frozen=True)
class PredictionEvaluation:
    paired_predictions: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    paired_bootstrap: pd.DataFrame
    coverage: dict[str, Any]


def validate_candidate_prediction_table(table: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CANDIDATE_REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Candidate prediction table is missing columns: {', '.join(missing)}")
    out = table.copy()
    if out["match_key"].isna().any() or (out["match_key"].astype(str).str.len() == 0).any():
        raise ValueError("Candidate prediction table contains empty match keys.")
    duplicates = int(out["match_key"].duplicated(keep=False).sum())
    if duplicates:
        raise ValueError(
            f"Candidate prediction table contains {duplicates} duplicate match-key rows."
        )
    for column in CANDIDATE_REQUIRED_COLUMNS[2:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    probabilities = out["pred_red_win_probability"].to_numpy(dtype=float)
    finite = np.isfinite(probabilities)
    if np.any((probabilities[finite] < 0) | (probabilities[finite] > 1)):
        raise ValueError("Candidate win probabilities must be within [0, 1] when present.")
    return out


def pair_prediction_tables(
    candidate: pd.DataFrame, statbotics: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidate = validate_candidate_prediction_table(candidate)
    if statbotics.empty:
        raise ValueError("Statbotics prediction table is empty.")
    season_values = pd.to_numeric(statbotics.get("season"), errors="coerce").dropna()
    if season_values.empty or season_values.nunique() != 1:
        raise ValueError("Statbotics predictions must contain exactly one season.")
    statbotics = validate_statbotics_prediction_table(statbotics, int(season_values.iloc[0]))
    candidate_keys = set(candidate["match_key"].astype(str))
    statbotics_keys = set(statbotics["match_key"].astype(str))
    matched_keys = candidate_keys & statbotics_keys
    coverage = {
        "candidate_rows": len(candidate),
        "statbotics_rows": len(statbotics),
        "matched_rows": len(matched_keys),
        "candidate_only_rows": len(candidate_keys - statbotics_keys),
        "statbotics_only_rows": len(statbotics_keys - candidate_keys),
        "candidate_duplicate_rows": 0,
        "statbotics_duplicate_rows": 0,
        "candidate_missing_predictions": {
            column: int(candidate[column].isna().sum())
            for column in CANDIDATE_REQUIRED_COLUMNS[2:]
        },
    }
    paired = candidate.merge(
        statbotics,
        on="match_key",
        how="inner",
        suffixes=("", "_statbotics"),
        validate="one_to_one",
    )
    if "event_key_statbotics" in paired:
        mismatch = paired["event_key"].astype(str) != paired["event_key_statbotics"].astype(str)
        if mismatch.any():
            raise ValueError(f"{int(mismatch.sum())} paired matches have conflicting event keys.")
        paired = paired.drop(columns=["event_key_statbotics"])
    if paired.empty:
        raise ValueError("Candidate and Statbotics artifacts have no matching match keys.")
    paired["is_tie"] = paired["actual_red_total_score"] == paired["actual_blue_total_score"]
    paired["actual_red_win"] = np.where(
        paired["is_tie"],
        np.nan,
        (paired["actual_red_total_score"] > paired["actual_blue_total_score"]).astype(float),
    )
    return paired.sort_values("match_key", kind="mergesort").reset_index(drop=True), coverage


def _model_columns(model: str) -> tuple[str, str, str]:
    if model == "latentstrat":
        return (
            "pred_red_total_score",
            "pred_blue_total_score",
            "pred_red_win_probability",
        )
    if model == "statbotics":
        return (
            "statbotics_pred_red_score",
            "statbotics_pred_blue_score",
            "statbotics_pred_red_win_probability",
        )
    raise ValueError(f"Unknown model {model!r}.")


def _metric_rows(table: pd.DataFrame, *, scope: str, week: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actual_red = table["actual_red_total_score"].to_numpy(dtype=float)
    actual_blue = table["actual_blue_total_score"].to_numpy(dtype=float)
    actual_diff = actual_red - actual_blue
    actual_win = table["actual_red_win"].to_numpy(dtype=float)
    for model in ("latentstrat", "statbotics"):
        red_column, blue_column, probability_column = _model_columns(model)
        pred_red = table[red_column].to_numpy(dtype=float)
        pred_blue = table[blue_column].to_numpy(dtype=float)
        pred_diff = pred_red - pred_blue
        alliance_errors = np.concatenate([pred_red - actual_red, pred_blue - actual_blue])
        diff_errors = pred_diff - actual_diff
        finite_alliance = np.isfinite(alliance_errors)
        finite_diff = np.isfinite(diff_errors)
        probabilities = table[probability_column].to_numpy(dtype=float)
        valid_win = np.isfinite(probabilities) & np.isfinite(actual_win)
        metrics: dict[str, tuple[float, int]] = {
            "alliance_score_mae": (
                float(np.mean(np.abs(alliance_errors[finite_alliance]))),
                int(finite_alliance.sum()),
            ),
            "alliance_score_mse": (
                float(np.mean(alliance_errors[finite_alliance] ** 2)),
                int(finite_alliance.sum()),
            ),
            "score_differential_mae": (
                float(np.mean(np.abs(diff_errors[finite_diff]))),
                int(finite_diff.sum()),
            ),
            "score_differential_mse": (
                float(np.mean(diff_errors[finite_diff] ** 2)),
                int(finite_diff.sum()),
            ),
        }
        metrics["alliance_score_rmse"] = (
            metrics["alliance_score_mse"][0] ** 0.5,
            metrics["alliance_score_mse"][1],
        )
        metrics["score_differential_rmse"] = (
            metrics["score_differential_mse"][0] ** 0.5,
            metrics["score_differential_mse"][1],
        )
        if valid_win.any():
            probs = np.clip(probabilities[valid_win], PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
            outcomes = actual_win[valid_win]
            correct = (probs >= 0.5) == outcomes
            brier = (probs - outcomes) ** 2
            log_loss = -(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
            metrics.update(
                {
                    "winner_accuracy": (float(np.mean(correct)), int(valid_win.sum())),
                    "winner_error_rate": (float(np.mean(~correct)), int(valid_win.sum())),
                    "winner_brier": (float(np.mean(brier)), int(valid_win.sum())),
                    "winner_log_loss": (float(np.mean(log_loss)), int(valid_win.sum())),
                }
            )
        for metric, (value, count) in metrics.items():
            rows.append(
                {
                    "scope": scope,
                    "historical_week": week,
                    "model": model,
                    "metric": metric,
                    "value": value,
                    "count": count,
                }
            )
    return rows


def prediction_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    rows = _metric_rows(paired, scope="aggregate", week=None)
    if "val_week" in paired.columns:
        weeks = pd.to_numeric(paired["val_week"], errors="coerce")
        for week in sorted(weeks.dropna().unique()):
            rows.extend(
                _metric_rows(
                    paired.loc[weeks == week], scope="historical_week", week=float(week)
                )
            )
    return pd.DataFrame(rows)


def prediction_calibration(paired: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    if bins <= 1:
        raise ValueError("bins must be greater than 1.")
    frames = [("aggregate", None, paired)]
    if "val_week" in paired.columns:
        weeks = pd.to_numeric(paired["val_week"], errors="coerce")
        frames.extend(
            ("historical_week", float(week), paired.loc[weeks == week])
            for week in sorted(weeks.dropna().unique())
        )
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for scope, week, frame in frames:
        actual = frame["actual_red_win"].to_numpy(dtype=float)
        for model in ("latentstrat", "statbotics"):
            probability_column = _model_columns(model)[2]
            probabilities = frame[probability_column].to_numpy(dtype=float)
            valid = np.isfinite(probabilities) & np.isfinite(actual)
            probs = probabilities[valid]
            outcomes = actual[valid]
            indices = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, bins - 1)
            bin_rows = []
            for idx in range(bins):
                mask = indices == idx
                mean_probability = float(np.mean(probs[mask])) if mask.any() else np.nan
                observed_rate = float(np.mean(outcomes[mask])) if mask.any() else np.nan
                gap = mean_probability - observed_rate
                bin_rows.append((int(mask.sum()), mean_probability, observed_rate, gap))
            total = max(len(probs), 1)
            ece = float(sum(count / total * abs(gap) for count, _, _, gap in bin_rows if count))
            for idx, (count, mean_probability, observed_rate, gap) in enumerate(bin_rows):
                rows.append(
                    {
                        "scope": scope,
                        "historical_week": week,
                        "model": model,
                        "bin_low": edges[idx],
                        "bin_high": edges[idx + 1],
                        "count": count,
                        "mean_probability": mean_probability,
                        "observed_rate": observed_rate,
                        "gap": gap,
                        "expected_calibration_error": ece,
                    }
                )
    return pd.DataFrame(rows)


def _per_match_losses(paired: pd.DataFrame, model: str) -> pd.DataFrame:
    red_column, blue_column, probability_column = _model_columns(model)
    actual_red = paired["actual_red_total_score"].to_numpy(dtype=float)
    actual_blue = paired["actual_blue_total_score"].to_numpy(dtype=float)
    pred_red = paired[red_column].to_numpy(dtype=float)
    pred_blue = paired[blue_column].to_numpy(dtype=float)
    red_error = pred_red - actual_red
    blue_error = pred_blue - actual_blue
    diff_error = (pred_red - pred_blue) - (actual_red - actual_blue)
    probabilities = np.clip(
        paired[probability_column].to_numpy(dtype=float),
        PROBABILITY_EPSILON,
        1 - PROBABILITY_EPSILON,
    )
    actual_win = paired["actual_red_win"].to_numpy(dtype=float)
    valid_win = np.isfinite(actual_win) & np.isfinite(probabilities)
    return pd.DataFrame(
        {
            "alliance_score_absolute_error": 0.5 * (np.abs(red_error) + np.abs(blue_error)),
            "alliance_score_squared_error": 0.5 * (red_error**2 + blue_error**2),
            "score_differential_absolute_error": np.abs(diff_error),
            "score_differential_squared_error": diff_error**2,
            "winner_brier": np.where(valid_win, (probabilities - actual_win) ** 2, np.nan),
            "winner_log_loss": np.where(
                valid_win,
                -(
                    actual_win * np.log(probabilities)
                    + (1 - actual_win) * np.log(1 - probabilities)
                ),
                np.nan,
            ),
            "winner_error": np.where(valid_win, (probabilities >= 0.5) != actual_win, np.nan),
        }
    )


def event_cluster_paired_bootstrap(
    paired: pd.DataFrame,
    *,
    resamples: int = 2_000,
    seed: int = 2026,
) -> pd.DataFrame:
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    candidate = _per_match_losses(paired, "latentstrat")
    baseline = _per_match_losses(paired, "statbotics")
    deltas = candidate - baseline
    events = paired["event_key"].astype(str).to_numpy()
    unique_events = np.unique(events)
    rng = np.random.default_rng(seed)
    draws = {column: [] for column in deltas.columns}
    for _ in range(resamples):
        selected = rng.choice(unique_events, size=len(unique_events), replace=True)
        indices = np.concatenate([np.flatnonzero(events == event) for event in selected])
        for column in deltas.columns:
            values = deltas[column].to_numpy(dtype=float)[indices]
            draws[column].append(float(np.nanmean(values)))
    rows = []
    for column, values in draws.items():
        observed = deltas[column].to_numpy(dtype=float)
        samples = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": column,
                "delta_mean": float(np.nanmean(observed)),
                "ci_low": float(np.nanquantile(samples, 0.025)),
                "ci_high": float(np.nanquantile(samples, 0.975)),
                "probability_of_improvement": float(np.nanmean(samples < 0)),
                "cluster": "event_key",
                "resamples": int(resamples),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def evaluate_prediction_tables(
    candidate: pd.DataFrame,
    statbotics: pd.DataFrame,
    *,
    resamples: int = 2_000,
    seed: int = 2026,
) -> PredictionEvaluation:
    paired, coverage = pair_prediction_tables(candidate, statbotics)
    return PredictionEvaluation(
        paired_predictions=paired,
        metrics=prediction_metrics(paired),
        calibration=prediction_calibration(paired),
        paired_bootstrap=event_cluster_paired_bootstrap(paired, resamples=resamples, seed=seed),
        coverage=coverage,
    )


def write_prediction_evaluation(
    candidate_path: str | Path,
    statbotics_path: str | Path,
    output_dir: str | Path,
    *,
    resamples: int = 2_000,
    seed: int = 2026,
) -> PredictionEvaluation:
    candidate_path = Path(candidate_path)
    statbotics_path = Path(statbotics_path)
    result = evaluate_prediction_tables(
        pd.read_parquet(candidate_path, engine="pyarrow"),
        pd.read_parquet(statbotics_path, engine="pyarrow"),
        resamples=resamples,
        seed=seed,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.paired_predictions.to_parquet(output / "paired_predictions.parquet", index=False)
    result.metrics.to_csv(output / "metrics.csv", index=False)
    result.calibration.to_csv(output / "calibration.csv", index=False)
    result.paired_bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    (output / "coverage.json").write_text(
        json.dumps(result.coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_artifact_manifest(
        output,
        workflow="artifacts.evaluate-predictions",
        artifact_version="v1",
        resolved_config={
            "resamples": resamples,
            "seed": seed,
            "comparison_complete": True,
            "winner_ties_excluded": True,
        },
        source_files={"candidate": candidate_path, "statbotics": statbotics_path},
        promotion_eligible=False,
        promotion_note=(
            "Historical V5.8 development evidence; fold checkpoints selected epochs on the same "
            "weeks represented by these predictions."
        ),
    )
    return result


def paired_bootstrap_noninferiority(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    resamples: int = 2_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Compatibility implementation for historical V6-Lite comparisons."""
    keys = ["fold_number", "match_key"]
    joined = baseline.merge(
        candidate, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one"
    )
    if "total_score_squared_error_baseline" not in joined:
        for suffix in ("baseline", "candidate"):
            joined[f"total_score_squared_error_{suffix}"] = 0.5 * (
                (
                    joined[f"pred_red_total_score_{suffix}"]
                    - joined[f"actual_red_total_score_{suffix}"]
                )
                ** 2
                + (
                    joined[f"pred_blue_total_score_{suffix}"]
                    - joined[f"actual_blue_total_score_{suffix}"]
                )
                ** 2
            )
    specs = {
        "brier": ("win_brier", "absolute", 0.002),
        "log_loss": ("win_log_loss", "absolute", 0.01),
        "total_score_mse": ("total_score_squared_error", "relative", 0.01),
    }
    rng = np.random.default_rng(seed)
    groups = [group for _, group in joined.groupby("fold_number", sort=True)]
    draws = {name: [] for name in specs}
    for _ in range(resamples):
        sampled = pd.concat(
            [group.iloc[rng.integers(0, len(group), size=len(group))] for group in groups],
            ignore_index=True,
        )
        for name, (column, mode, _) in specs.items():
            base = float(pd.to_numeric(sampled[f"{column}_baseline"], errors="coerce").mean())
            cand = float(pd.to_numeric(sampled[f"{column}_candidate"], errors="coerce").mean())
            delta = cand - base
            if mode == "relative":
                delta = delta / base if abs(base) > np.finfo(float).eps else np.inf
            draws[name].append(delta)
    rows = []
    for name, (_, _, epsilon) in specs.items():
        values = np.asarray(draws[name])
        rows.append(
            {
                "metric": name,
                "delta_mean": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "epsilon": epsilon,
                "passes_noninferiority": bool(np.quantile(values, 0.975) <= epsilon),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "PredictionEvaluation",
    "evaluate_prediction_tables",
    "event_cluster_paired_bootstrap",
    "pair_prediction_tables",
    "paired_bootstrap_noninferiority",
    "prediction_calibration",
    "prediction_metrics",
    "validate_candidate_prediction_table",
    "write_prediction_evaluation",
]
