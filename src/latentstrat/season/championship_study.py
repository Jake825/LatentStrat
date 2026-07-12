"""Fixed-100-epoch static Championship diagnostic orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from latentstrat.artifacts import sha256_file, write_artifact_manifest
from latentstrat.season.features import read_feature_table
from latentstrat.season.reference_study import validate_reference_rows
from latentstrat.season.static_core import (
    CHAMPIONSHIP_EVENTS,
    EINSTEIN_EVENT,
    StaticStudySpec,
    apply_calibrators,
    build_split_manifest,
    fit_calibrators,
    manifest_keys,
    prediction_frame,
    train_core_leaf,
)

WORKFLOW = "season.2026-static-championship-core"
ARTIFACT_SCHEMA = "2026-static-championship-core-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARCHITECTURES = ("additive", "teammate-set", "full-match")
SCORE_MODES = ("phase-core", "official-total-core")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return not result.stdout.strip()


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    ).hexdigest()


def _key_hash(keys: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def _compact_team_keys(table: pd.DataFrame) -> tuple[str, ...]:
    columns = [f"{color}_team_{slot}_key" for color in ("red", "blue") for slot in (1, 2, 3)]
    return tuple(sorted(set(table[columns].astype(str).to_numpy().ravel())))


def _alliance_design(
    table: pd.DataFrame, team_keys: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = {key: idx for idx, key in enumerate(team_keys)}
    rows = []
    targets = []
    weeks = []
    colors = []
    for row in table.itertuples(index=False):
        for color in ("red", "blue"):
            vector = np.zeros(len(team_keys), dtype=float)
            for slot in (1, 2, 3):
                key = str(getattr(row, f"{color}_team_{slot}_key"))
                if key in index:
                    vector[index[key]] = 1.0
            rows.append(vector)
            targets.append(float(getattr(row, f"{color}_total_score")))
            weeks.append(float(row.event_week))
            colors.append(color)
    return np.asarray(rows), np.asarray(targets), np.asarray(weeks), np.asarray(colors)


def _baseline_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    development_train: pd.DataFrame,
    development_holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_keys = _compact_team_keys(pd.concat([train, evaluation], ignore_index=True))

    def fit_predict(
        name: str,
        fit_table: pd.DataFrame,
        predict_table: pd.DataFrame,
        *,
        recency: bool,
    ) -> np.ndarray:
        clean = fit_table[
            ~(fit_table["red_has_dq"].astype(bool) | fit_table["blue_has_dq"].astype(bool))
        ]
        x, y, weeks, _ = _alliance_design(clean, team_keys)
        px, _, _, _ = _alliance_design(predict_table, team_keys)
        if name == "mean-total":
            return np.full(len(px), float(np.mean(y)))
        weights = np.power(0.7, np.maximum(np.nanmax(weeks) - weeks, 0)) if recency else None
        model = Ridge(alpha=1.0, fit_intercept=True, solver="lsqr").fit(x, y, sample_weight=weights)
        return np.asarray(model.predict(px), dtype=float)

    baseline_rows = []
    calibration_rows = []
    for name, recency in (
        ("mean-total", False),
        ("direct-total-ridge", False),
        ("direct-total-pridge", True),
    ):
        development_scores = fit_predict(
            name, development_train, development_holdout, recency=recency
        ).reshape(-1, 2)
        dev = development_holdout.reset_index(drop=True)
        dev_diff = development_scores[:, 0] - development_scores[:, 1]
        dev_actual_red = dev["red_total_score"].to_numpy(float)
        dev_actual_blue = dev["blue_total_score"].to_numpy(float)
        clean = ~(
            dev["red_has_dq"].astype(bool).to_numpy()
            | dev["blue_has_dq"].astype(bool).to_numpy()
            | np.isclose(dev_actual_red, dev_actual_blue)
        )
        calibrator = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, random_state=2026
        ).fit(dev_diff[clean, None], (dev_actual_red[clean] > dev_actual_blue[clean]).astype(int))
        scores = fit_predict(name, train, evaluation, recency=recency).reshape(-1, 2)
        diff = scores[:, 0] - scores[:, 1]
        probability = calibrator.predict_proba(diff[:, None])[:, 1]
        baseline_rows.append(
            pd.DataFrame(
                {
                    "phase": "test",
                    "supervision_mode": "baseline",
                    "architecture": name,
                    "seed": "control",
                    "epoch": 0,
                    "event_key": evaluation["event_key"].astype(str).to_numpy(),
                    "match_key": evaluation["match_key"].astype(str).to_numpy(),
                    "pred_red_total_score": scores[:, 0],
                    "pred_blue_total_score": scores[:, 1],
                    "actual_red_total_score": evaluation["red_total_score"].to_numpy(float),
                    "actual_blue_total_score": evaluation["blue_total_score"].to_numpy(float),
                    "raw_pred_red_win_probability": probability,
                    "transferred_platt_red_win_probability": probability,
                    "transferred_score_red_win_probability": probability,
                    "red_has_dq": evaluation["red_has_dq"].astype(bool).to_numpy(),
                    "blue_has_dq": evaluation["blue_has_dq"].astype(bool).to_numpy(),
                    "red_has_surrogate": evaluation["red_has_surrogate"].astype(bool).to_numpy(),
                    "blue_has_surrogate": evaluation["blue_has_surrogate"].astype(bool).to_numpy(),
                }
            )
        )
        calibration_rows.append(
            {
                "baseline": name,
                "coefficient": float(calibrator.coef_[0, 0]),
                "intercept": float(calibrator.intercept_[0]),
                "alpha": 1.0 if name != "mean-total" else np.nan,
                "fit_intercept": True,
                "recency_decay": 0.7 if recency else np.nan,
                "design": "alliance three-team binary incidence",
                "target": "official alliance total",
                "dq_masked": True,
            }
        )
    return pd.concat(baseline_rows, ignore_index=True), pd.DataFrame(calibration_rows)


def _attach_evaluation_context(
    predictions: pd.DataFrame, training: pd.DataFrame, evaluation: pd.DataFrame
) -> pd.DataFrame:
    team_columns = [f"{color}_team_{slot}_key" for color in ("red", "blue") for slot in range(1, 4)]
    appearances = pd.Series(
        training[team_columns].astype(str).to_numpy().reshape(-1)
    ).value_counts()
    q33, q67 = appearances.quantile([1 / 3, 2 / 3]).to_numpy(float)
    context = evaluation[["match_key", "event_week", "event_type", *team_columns]].copy()
    observation_count = (
        context[team_columns]
        .astype(str)
        .apply(lambda column: column.map(appearances).fillna(0))
        .mean(axis=1)
    )
    context["mean_prior_team_observations"] = observation_count
    context["observation_slice"] = np.where(
        observation_count <= q33,
        "low",
        np.where(observation_count <= q67, "medium", "high"),
    )
    context_columns = [column for column in context if column != "match_key"]
    renamed = context.rename(columns={column: f"{column}__context" for column in context_columns})
    out = predictions.merge(
        renamed,
        on="match_key",
        how="left",
        validate="many_to_one",
    )
    for column in context_columns:
        context_column = f"{column}__context"
        if column in out:
            out[column] = out[column].where(out[column].notna(), out[context_column])
        else:
            out[column] = out[context_column]
        out = out.drop(columns=context_column)
    return out


def _metric_values(frame: pd.DataFrame, probability_column: str) -> dict[str, tuple[float, int]]:
    actual_red = frame["actual_red_total_score"].to_numpy(float)
    actual_blue = frame["actual_blue_total_score"].to_numpy(float)
    pred_red = frame["pred_red_total_score"].to_numpy(float)
    pred_blue = frame["pred_blue_total_score"].to_numpy(float)
    red_error = pred_red - actual_red
    blue_error = pred_blue - actual_blue
    alliance_error = np.r_[red_error, blue_error]
    diff_error = (pred_red - pred_blue) - (actual_red - actual_blue)
    probability = np.clip(frame[probability_column].to_numpy(float), 1e-7, 1 - 1e-7)
    tie = np.isclose(actual_red, actual_blue)
    valid = ~tie & np.isfinite(probability)
    outcome = (actual_red > actual_blue).astype(float)
    actual_mean = float(np.mean(np.r_[actual_red, actual_blue]))
    bias = float(np.mean(alliance_error))
    result = {
        "alliance_score_mae": (float(np.mean(np.abs(alliance_error))), len(alliance_error)),
        "alliance_score_rmse": (float(np.sqrt(np.mean(alliance_error**2))), len(alliance_error)),
        "score_differential_mae": (float(np.mean(np.abs(diff_error))), len(diff_error)),
        "score_differential_rmse": (float(np.sqrt(np.mean(diff_error**2))), len(diff_error)),
        "mean_bias": (bias, len(alliance_error)),
        "relative_bias": (bias / actual_mean, len(alliance_error)),
    }
    if valid.any():
        p = probability[valid]
        y = outcome[valid]
        result.update(
            {
                "winner_accuracy": (float(np.mean((p >= 0.5) == y)), int(valid.sum())),
                "winner_brier": (float(np.mean((p - y) ** 2)), int(valid.sum())),
                "winner_log_loss": (
                    float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))),
                    int(valid.sum()),
                ),
            }
        )
        bins = np.clip(np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=True), 0, 9)
        ece = sum(
            float((bins == idx).mean())
            * abs(float(p[bins == idx].mean()) - float(y[bins == idx].mean()))
            for idx in range(10)
            if np.any(bins == idx)
        )
        result["winner_ece"] = (float(ece), int(valid.sum()))
    return result


def compute_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    calibration = []
    probability_columns = (
        "raw_pred_red_win_probability",
        "transferred_platt_red_win_probability",
        "transferred_score_red_win_probability",
    )
    groups = ["supervision_mode", "architecture", "seed", "epoch"]
    for identity, model_rows in predictions.groupby(groups, dropna=False):
        dq = model_rows["red_has_dq"].astype(bool) | model_rows["blue_has_dq"].astype(bool)
        scopes = {
            "primary-clean": model_rows[~dq],
            "all-official": model_rows,
            "DQ-only": model_rows[dq],
            "surrogate-only": model_rows[
                model_rows["red_has_surrogate"].astype(bool)
                | model_rows["blue_has_surrogate"].astype(bool)
            ],
        }
        scopes.update(
            {f"division:{event}": frame for event, frame in model_rows.groupby("event_key")}
        )
        if "observation_slice" in model_rows:
            scopes.update(
                {
                    f"observation:{name}": frame
                    for name, frame in model_rows.groupby("observation_slice", dropna=False)
                }
            )
        for scope, frame in scopes.items():
            if frame.empty:
                continue
            for probability_column in probability_columns:
                if probability_column not in frame:
                    continue
                for metric, (value, count) in _metric_values(frame, probability_column).items():
                    rows.append(
                        {
                            **dict(zip(groups, identity, strict=True)),
                            "scope": scope,
                            "probability_variant": probability_column,
                            "metric": metric,
                            "value": value,
                            "count": count,
                            "direction": "higher" if metric in {"winner_accuracy"} else "lower",
                        }
                    )
                clean = frame[
                    ~(frame["red_has_dq"].astype(bool) | frame["blue_has_dq"].astype(bool))
                ]
                actual_red = clean["actual_red_total_score"].to_numpy(float)
                actual_blue = clean["actual_blue_total_score"].to_numpy(float)
                p = np.clip(clean[probability_column].to_numpy(float), 1e-7, 1 - 1e-7)
                valid = ~np.isclose(actual_red, actual_blue) & np.isfinite(p)
                y = (actual_red > actual_blue).astype(float)
                index = np.clip(np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=True), 0, 9)
                for bin_index in range(10):
                    selected = valid & (index == bin_index)
                    calibration.append(
                        {
                            **dict(zip(groups, identity, strict=True)),
                            "scope": scope,
                            "probability_variant": probability_column,
                            "bin": bin_index,
                            "count": int(selected.sum()),
                            "mean_prediction": float(np.mean(p[selected]))
                            if selected.any()
                            else np.nan,
                            "observed_rate": float(np.mean(y[selected]))
                            if selected.any()
                            else np.nan,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(calibration)


def _per_match_losses(frame: pd.DataFrame) -> pd.DataFrame:
    actual_red = frame["actual_red_total_score"].to_numpy(float)
    actual_blue = frame["actual_blue_total_score"].to_numpy(float)
    pred_red = frame["pred_red_total_score"].to_numpy(float)
    pred_blue = frame["pred_blue_total_score"].to_numpy(float)
    p = np.clip(frame["raw_pred_red_win_probability"].to_numpy(float), 1e-7, 1 - 1e-7)
    y = (actual_red > actual_blue).astype(float)
    tie = np.isclose(actual_red, actual_blue)
    return pd.DataFrame(
        {
            "event_key": frame["event_key"].astype(str),
            "match_key": frame["match_key"].astype(str),
            "score_squared_error": 0.5
            * ((pred_red - actual_red) ** 2 + (pred_blue - actual_blue) ** 2),
            "differential_squared_error": ((pred_red - pred_blue) - (actual_red - actual_blue))
            ** 2,
            "winner_brier": np.where(tie, np.nan, (p - y) ** 2),
            "winner_log_loss": np.where(tie, np.nan, -(y * np.log(p) + (1 - y) * np.log(1 - p))),
        }
    )


def paired_comparisons(
    predictions: pd.DataFrame, *, resamples: int = 5_000, seed: int = 2026
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neural = predictions[
        predictions["supervision_mode"].isin(SCORE_MODES)
        & ~(predictions["red_has_dq"].astype(bool) | predictions["blue_has_dq"].astype(bool))
    ]
    pairs = []
    for architecture in ARCHITECTURES:
        pairs.append(
            (
                f"supervision:{architecture}",
                ("phase-core", architecture),
                ("official-total-core", architecture),
            )
        )
    for mode in SCORE_MODES:
        pairs.extend(
            [
                (f"teammate-v-additive:{mode}", (mode, "additive"), (mode, "teammate-set")),
                (f"full-v-teammate:{mode}", (mode, "teammate-set"), (mode, "full-match")),
                (f"full-v-additive:{mode}", (mode, "additive"), (mode, "full-match")),
            ]
        )
    rng = np.random.default_rng(seed)
    summary_rows = []
    bootstrap_rows = []
    loo_rows = []
    for name, left_key, right_key in pairs:
        left = neural[
            neural["supervision_mode"].eq(left_key[0]) & neural["architecture"].eq(left_key[1])
        ]
        right = neural[
            neural["supervision_mode"].eq(right_key[0]) & neural["architecture"].eq(right_key[1])
        ]
        joined = _per_match_losses(left).merge(
            _per_match_losses(right),
            on=["event_key", "match_key"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        events = sorted(joined["event_key"].unique())
        for metric in (
            "score_squared_error",
            "differential_squared_error",
            "winner_brier",
            "winner_log_loss",
        ):
            left_values = joined[f"{metric}_left"]
            right_values = joined[f"{metric}_right"]
            delta = right_values - left_values
            observed = float(np.nanmean(delta))
            left_mean = float(np.nanmean(left_values))
            right_mean = float(np.nanmean(right_values))
            if metric.endswith("squared_error"):
                left_effect = math.sqrt(left_mean)
                right_effect = math.sqrt(right_mean)
            else:
                left_effect = left_mean
                right_effect = right_mean
            samples = []
            for _ in range(resamples):
                selected_events = rng.choice(events, size=len(events), replace=True)
                selected = pd.concat(
                    [joined[joined["event_key"].eq(event)] for event in selected_events],
                    ignore_index=True,
                )
                samples.append(
                    float(np.nanmean(selected[f"{metric}_right"] - selected[f"{metric}_left"]))
                )
            samples_array = np.asarray(samples)
            summary_rows.append(
                {
                    "comparison": name,
                    "left": f"{left_key[0]}:{left_key[1]}",
                    "right": f"{right_key[0]}:{right_key[1]}",
                    "metric": metric,
                    "delta_right_minus_left": observed,
                    "left_value": left_effect,
                    "right_value": right_effect,
                    "relative_delta": (
                        (right_effect - left_effect) / left_effect if left_effect else np.nan
                    ),
                    "ci_low": float(np.quantile(samples_array, 0.025)),
                    "ci_high": float(np.quantile(samples_array, 0.975)),
                    "probability_improvement": float(np.mean(samples_array < 0)),
                }
            )
            bootstrap_rows.extend(
                {"comparison": name, "metric": metric, "resample": idx, "delta": value}
                for idx, value in enumerate(samples)
            )
            for omitted in events:
                subset = joined[~joined["event_key"].eq(omitted)]
                left_loo = float(np.nanmean(subset[f"{metric}_left"]))
                right_loo = float(np.nanmean(subset[f"{metric}_right"]))
                if metric.endswith("squared_error"):
                    left_loo = math.sqrt(left_loo)
                    right_loo = math.sqrt(right_loo)
                loo_rows.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "omitted_event": omitted,
                        "delta_right_minus_left": float(
                            np.nanmean(subset[f"{metric}_right"] - subset[f"{metric}_left"])
                        ),
                        "left_value": left_loo,
                        "right_value": right_loo,
                        "relative_delta": (
                            (right_loo - left_loo) / left_loo if left_loo else np.nan
                        ),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_rows), pd.DataFrame(loo_rows)


def classify_comparisons(comparisons: pd.DataFrame, leave_one_out: pd.DataFrame) -> pd.DataFrame:
    """Apply the predeclared practical-effect and stability rules."""

    rows = []
    for comparison, frame in comparisons.groupby("comparison"):
        values = frame.set_index("metric")

        brier_delta = float(values.loc["winner_brier", "delta_right_minus_left"])
        log_delta = float(values.loc["winner_log_loss", "delta_right_minus_left"])
        score_relative = float(values.loc["score_squared_error", "relative_delta"])
        differential_relative = float(values.loc["differential_squared_error", "relative_delta"])
        loo = leave_one_out[leave_one_out["comparison"].eq(comparison)]
        if comparison.startswith("supervision:"):
            supported = (
                score_relative <= -0.02
                and float(values.loc["score_squared_error", "probability_improvement"]) >= 0.80
                and differential_relative <= 0.02
                and brier_delta <= 0.005
                and log_delta <= 0.01
            )
            harmful = (
                score_relative >= 0.02
                and float(values.loc["score_squared_error", "probability_improvement"]) <= 0.20
                and differential_relative >= -0.02
                and brier_delta >= -0.005
                and log_delta >= -0.01
            )
            stable_supported = not (
                loo[loo["metric"].eq("score_squared_error")]["relative_delta"] > 0.02
            ).any()
            stable_harmful = not (
                loo[loo["metric"].eq("score_squared_error")]["relative_delta"] < -0.02
            ).any()
        else:
            differential_supported = (
                differential_relative <= -0.02
                and float(values.loc["differential_squared_error", "probability_improvement"])
                >= 0.80
                and brier_delta <= 0.005
                and log_delta <= 0.01
            )
            probability_supported = (
                brier_delta <= -0.005
                and log_delta <= -0.01
                and float(values.loc["winner_brier", "probability_improvement"]) >= 0.80
                and float(values.loc["winner_log_loss", "probability_improvement"]) >= 0.80
                and differential_relative <= 0.02
            )
            supported = (differential_supported or probability_supported) and score_relative <= 0.02
            differential_harmful = (
                differential_relative >= 0.02
                and float(values.loc["differential_squared_error", "probability_improvement"])
                <= 0.20
                and brier_delta >= -0.005
                and log_delta >= -0.01
            )
            probability_harmful = (
                brier_delta >= 0.005
                and log_delta >= 0.01
                and float(values.loc["winner_brier", "probability_improvement"]) <= 0.20
                and float(values.loc["winner_log_loss", "probability_improvement"]) <= 0.20
                and differential_relative >= -0.02
            )
            harmful = (differential_harmful or probability_harmful) and score_relative >= -0.02
            supported_reversal = (
                (loo[loo["metric"].eq("differential_squared_error")]["relative_delta"] > 0.02).any()
                or (loo[loo["metric"].eq("winner_brier")]["delta_right_minus_left"] > 0.005).any()
                or (loo[loo["metric"].eq("winner_log_loss")]["delta_right_minus_left"] > 0.01).any()
            )
            harmful_reversal = (
                (
                    loo[loo["metric"].eq("differential_squared_error")]["relative_delta"] < -0.02
                ).any()
                or (loo[loo["metric"].eq("winner_brier")]["delta_right_minus_left"] < -0.005).any()
                or (
                    loo[loo["metric"].eq("winner_log_loss")]["delta_right_minus_left"] < -0.01
                ).any()
            )
            stable_supported = not supported_reversal
            stable_harmful = not harmful_reversal
        if supported and stable_supported:
            classification = "diagnostic-supported"
        elif harmful and stable_harmful:
            classification = "harmful"
        else:
            classification = "uncertain"
        rows.append(
            {
                "comparison": comparison,
                "classification": classification,
                "score_rmse_relative_delta": score_relative,
                "differential_rmse_relative_delta": differential_relative,
                "brier_delta": brier_delta,
                "log_loss_delta": log_delta,
                "leave_one_division_out_stable": (
                    stable_supported if supported else stable_harmful if harmful else False
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_report(
    *,
    metrics: pd.DataFrame,
    history: pd.DataFrame,
    comparisons: pd.DataFrame,
    classifications: pd.DataFrame,
    coverage: dict[str, object],
    complete: bool,
) -> str:
    neural = metrics[metrics["supervision_mode"].isin(SCORE_MODES)]
    primary = neural[
        neural["scope"].eq("primary-clean")
        & neural["probability_variant"].eq("raw_pred_red_win_probability")
    ]
    classification_map = (
        classifications.set_index("comparison")["classification"].to_dict()
        if not classifications.empty
        else {}
    )
    lines = [
        "# 2026 Static Championship Core Diagnostic",
        "",
        "> Under a fixed 100-epoch, fixed-loss static `Z_base` model, does direct "
        "official-total supervision materially reduce Championship score bias without "
        "harming relative-strength or probability quality, and do teammate or opponent "
        "interaction modules add useful information beyond additive team strength?",
        "",
        "## Executive Verdict",
        "",
    ]
    if not complete:
        lines.append("The matrix is incomplete, so no matrix-wide conclusion is reported.")
    else:
        pivot = primary.pivot_table(
            index=["supervision_mode", "architecture"], columns="metric", values="value"
        )
        direct_labels = {
            architecture: classification_map.get(f"supervision:{architecture}", "unavailable")
            for architecture in ARCHITECTURES
        }
        teammate_labels = {
            mode: classification_map.get(f"teammate-v-additive:{mode}", "unavailable")
            for mode in SCORE_MODES
        }
        opponent_labels = {
            mode: classification_map.get(f"full-v-teammate:{mode}", "unavailable")
            for mode in SCORE_MODES
        }
        lines.extend(
            [
                "All six development and six refit runs completed. The tables below are "
                "fixed-100-epoch diagnostic evidence, not a best-achievable architecture ranking.",
                "",
                "Direct official-total supervision was classified as "
                + ", ".join(
                    f"`{name}` for {architecture}" for architecture, name in direct_labels.items()
                )
                + ".",
                "Teammate interaction versus additive strength was "
                + ", ".join(f"`{name}` under {mode}" for mode, name in teammate_labels.items())
                + "; opponent interaction versus teammate-only was "
                + ", ".join(f"`{name}` under {mode}" for mode, name in opponent_labels.items())
                + ".",
                "These labels answer the declared question only at epoch 100 under this static, "
                "one-seed contract.",
                "",
                classifications.to_markdown(index=False, floatfmt=".4f"),
                "",
                pivot.reset_index().to_markdown(index=False, floatfmt=".4f"),
            ]
        )
    lines.extend(
        ["", "## Data and Provenance", "", "```json", json.dumps(coverage, indent=2), "```"]
    )
    lines.extend(["", "## Development Runs", ""])
    for (mode, architecture), frame in history[history["phase"].eq("development")].groupby(
        ["supervision_mode", "architecture"]
    ):
        final = frame.sort_values("epoch").iloc[-1]
        lines.extend(
            [
                f"### {mode} / {architecture}",
                "",
                f"Completed {int(final['epoch'])} epochs. Final validation score RMSE was "
                f"`{final.get('validation_score_rmse', math.nan):.3f}`, differential RMSE "
                f"`{final.get('validation_score_differential_rmse', math.nan):.3f}`, Brier "
                f"`{final.get('validation_winner_brier', math.nan):.4f}`, and log loss "
                f"`{final.get('validation_winner_log_loss', math.nan):.4f}`.",
                f"Training loss moved from "
                f"`{frame.sort_values('epoch').iloc[0].get('train_loss', math.nan):.4f}` "
                f"to `{final.get('train_loss', math.nan):.4f}`; mean `Z_base` update distance "
                f"ended at `{final.get('z_base_update_norm_mean', math.nan):.4f}`.",
                "",
            ]
        )
        for metric in (
            "validation_score_rmse",
            "validation_score_differential_rmse",
            "validation_winner_brier",
            "validation_winner_log_loss",
        ):
            if metric in frame and frame[metric].notna().any():
                best = frame.loc[frame[metric].idxmin()]
                lines.append(
                    f"- Best {metric.removeprefix('validation_')} occurred at epoch "
                    f"{int(best['epoch'])}: `{best[metric]:.4f}`; epoch 100: "
                    f"`{final[metric]:.4f}`."
                )
        lines.append("")
    lines.extend(["## Final Refit Runs", ""])
    for mode in SCORE_MODES:
        for architecture in ARCHITECTURES:
            frame = primary[
                primary["supervision_mode"].eq(mode) & primary["architecture"].eq(architecture)
            ]
            lines.extend([f"### {mode} / {architecture}", ""])
            if frame.empty:
                lines.extend(["No complete Championship evidence.", ""])
                continue
            values = dict(zip(frame["metric"], frame["value"], strict=False))
            lines.extend(
                [
                    f"At epoch 100: score RMSE "
                    f"`{values.get('alliance_score_rmse', math.nan):.3f}`, mean bias "
                    f"`{values.get('mean_bias', math.nan):.3f}`, relative bias "
                    f"`{values.get('relative_bias', math.nan):.3%}`, differential RMSE "
                    f"`{values.get('score_differential_rmse', math.nan):.3f}`, Brier "
                    f"`{values.get('winner_brier', math.nan):.4f}`, and log loss "
                    f"`{values.get('winner_log_loss', math.nan):.4f}`.",
                    "",
                ]
            )
    development_diagnostics = []
    for (mode, architecture), frame in history[history["phase"].eq("development")].groupby(
        ["supervision_mode", "architecture"]
    ):
        ordered = frame.sort_values("epoch")
        final = ordered.iloc[-1]
        row = {
            "supervision_mode": mode,
            "architecture": architecture,
            "first_train_loss": float(ordered.iloc[0]["train_loss"]),
            "final_train_loss": float(final["train_loss"]),
            "final_z_base_update": float(final["z_base_update_norm_mean"]),
        }
        for short, metric in (
            ("score", "validation_score_rmse"),
            ("differential", "validation_score_differential_rmse"),
            ("brier", "validation_winner_brier"),
            ("log_loss", "validation_winner_log_loss"),
        ):
            best = ordered.loc[ordered[metric].idxmin()]
            row[f"best_{short}_epoch"] = int(best["epoch"])
            row[f"best_{short}"] = float(best[metric])
            row[f"final_{short}"] = float(final[metric])
        development_diagnostics.append(row)
    diagnostics_table = pd.DataFrame(development_diagnostics)
    lines.extend(
        [
            "## What 100 Epochs Changed",
            "",
            "This table separates continued loss reduction from the epoch of each development "
            "forecast optimum. An optimum before epoch 100 is evidence of fixed-horizon "
            "sensitivity, not proof that the architecture is intrinsically harmful.",
            "",
            diagnostics_table.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    )
    calibration_table = neural[
        neural["scope"].eq("primary-clean")
        & neural["metric"].isin(["winner_brier", "winner_log_loss", "winner_ece"])
    ].pivot_table(
        index=["supervision_mode", "architecture", "metric"],
        columns="probability_variant",
        values="value",
        aggfunc="first",
    )
    lines.extend(
        [
            "## Raw and Transferred Calibration",
            "",
            "Platt and score-derived mappings were fitted on the sealed development holdout and "
            "transferred to separately initialized full pre-Championship refits. They are not "
            "guaranteed Championship calibration.",
            "",
            calibration_table.reset_index().to_markdown(index=False, floatfmt=".5f"),
            "",
        ]
    )
    controls = metrics[
        metrics["scope"].eq("primary-clean")
        & metrics["probability_variant"].eq("raw_pred_red_win_probability")
        & ~metrics["supervision_mode"].isin(SCORE_MODES)
    ]
    if not controls.empty:
        control_table = controls.pivot_table(
            index=["supervision_mode", "architecture"], columns="metric", values="value"
        )
        lines.extend(
            [
                "## Frozen Baselines and Historical Context",
                "",
                "Mean, ridge, and rolling pRidge controls use the same non-DQ training rows and "
                "exact Championship keys. The historical static reference is descriptive only.",
                "",
                control_table.reset_index().to_markdown(index=False, floatfmt=".5f"),
                "",
            ]
        )
    bias_slices = neural[
        neural["probability_variant"].eq("raw_pred_red_win_probability")
        & neural["metric"].isin(["alliance_score_rmse", "mean_bias", "relative_bias"])
        & neural["scope"].str.startswith(("DQ-only", "surrogate-only", "observation:"))
    ]
    lines.extend(
        [
            "## Bias and Availability Slices",
            "",
            bias_slices[
                [
                    "supervision_mode",
                    "architecture",
                    "scope",
                    "metric",
                    "value",
                    "count",
                ]
            ].to_markdown(index=False, floatfmt=".5f"),
            "",
        ]
    )
    lines.extend(
        [
            "## Paired Comparisons and Sensitivity",
            "",
            comparisons.to_markdown(index=False, floatfmt=".5f")
            if not comparisons.empty
            else "No complete paired comparisons.",
            "",
            "Every classification also passed through the saved leave-one-division-out veto. "
            "See `leave_one_division_out.csv` for all omitted-division rows.",
            "",
            "## Findings, Explanations, and Open Questions",
            "",
            "### Confirmed by this diagnostic",
            "",
            "- All final comparisons use the same exact primary-clean Championship match set.",
            "- The predeclared classifications above describe supervision and interaction effects "
            "at the fixed 100-epoch horizon.",
            "- The development table records whether score, relative-strength, or probability "
            "metrics peaked before epoch 100 even while training loss continued downward.",
            "- The recorded `Z_base` update distances quantify state movement without assigning "
            "semantic meaning to latent coordinates.",
            "",
            "### Plausible explanations requiring another experiment",
            "",
            "- Remaining score bias may reflect static state, target construction, distribution "
            "shift, or optimization. This matrix isolates supervision and interaction only.",
            "- A transferred calibrator can shift after the full-data refit; raw and transferred "
            "probabilities must therefore be interpreted together.",
            "- Any interaction benefit may depend on training duration because the architectures "
            "have different development optima.",
            "",
            "### Unanswered",
            "",
            "- Whether explicit week or event context reduces the remaining out-of-time bias.",
            "- Whether the fixed-horizon effects reproduce across seeds.",
            "- Each architecture's best achievable performance under a separately sealed "
            "selection protocol.",
            "- Whether these forecast changes improve alliance-selection decisions or tournament "
            "outcomes.",
            "",
            "## Interpretation Boundaries",
            "",
            "- This study does not compare static and temporal state.",
            "- Championship divisions are reused diagnostic evidence.",
            "- One seed cannot establish optimization reliability.",
            "- Earlier development optima are diagnostic only; Championship predictions "
            "use epoch 100 exclusively.",
            "",
            "## Next Decision",
            "",
            "Use the supported parts of this matrix as the fixed static control for one next "
            "experiment. If material score bias remains, vary only time/event context while "
            "holding the selected supervision, interaction architecture, split, fixed loss, and "
            "100-epoch horizon constant. Repeat across seeds before any promotion claim. "
            "Persistent bias motivates that test but does not prove a temporal model will "
            "solve it.",
            "",
        ]
    )
    return "\n".join(lines)


def run_championship_study(
    features_path: str | Path,
    event_metadata_path: str | Path,
    prior_checkpoint: str | Path,
    historical_reference: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 100,
    budget_minutes: float = 360,
    tensorboard_root: str | Path | None = "runs/2026-static-championship-core",
    bootstrap_resamples: int = 5_000,
    smoke: bool = False,
) -> Path:
    started = time.perf_counter()
    features_path = Path(features_path)
    event_metadata_path = Path(event_metadata_path)
    prior_checkpoint = Path(prior_checkpoint)
    historical_reference = Path(historical_reference)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = read_feature_table(features_path)
    events = pd.read_parquet(event_metadata_path)
    table, filtering = validate_reference_rows(raw, events)
    manifest = build_split_manifest(raw, events)
    manifest.to_parquet(output / "split_manifest.parquet", index=False)
    manifest.to_csv(output / "split_manifest.csv", index=False)
    epochs = min(epochs, 2) if smoke else epochs
    if not smoke and epochs != 100:
        raise ValueError("The formal diagnostic requires exactly 100 epochs.")
    if not smoke and not _git_clean():
        raise RuntimeError("Formal training requires a clean Git working tree.")
    study_id = output.name
    tensorboard_dir = Path(tensorboard_root) / study_id if tensorboard_root else None
    dev_train_keys = manifest_keys(manifest, "development_train")
    dev_eval_keys = manifest_keys(manifest, "development_holdout")
    final_train_keys = manifest_keys(manifest, "final_train")
    test_keys = manifest_keys(manifest, "championship_test")
    dev_table = table[table["match_key"].astype(str).isin((*dev_train_keys, *dev_eval_keys))].copy()
    final_table = table[table["match_key"].astype(str).isin((*final_train_keys, *test_keys))].copy()
    if not smoke:
        pilot_rows = []
        for mode in SCORE_MODES:
            for architecture in ARCHITECTURES:
                pilot_spec = StaticStudySpec(
                    mode,
                    architecture,
                    2,
                    2026,
                    ("score", "win", "embedding"),
                    dev_train_keys,
                    dev_eval_keys,
                    1.0,
                    1.0,
                    1.0,
                )
                pilot_result = train_core_leaf(
                    dev_table,
                    pilot_spec,
                    prior_checkpoint=prior_checkpoint,
                    output_dir=output / "timing-pilot" / mode / architecture,
                    log_dir=None,
                    phase="development",
                )
                seconds = pd.to_numeric(
                    pilot_result.history["epoch_seconds"], errors="coerce"
                ).dropna()
                pilot_rows.append(
                    {
                        "supervision_mode": mode,
                        "architecture": architecture,
                        "mean_epoch_seconds": float(seconds.mean()),
                        "measured_epochs": int(len(seconds)),
                    }
                )
        row_ratio = len(final_train_keys) / len(dev_train_keys)
        base_seconds = sum(row["mean_epoch_seconds"] for row in pilot_rows)
        projected_seconds = base_seconds * epochs * (1.0 + row_ratio)
        projected_with_margin = projected_seconds * 1.2
        timing = {
            "leaves": pilot_rows,
            "development_rows": len(dev_train_keys),
            "refit_rows": len(final_train_keys),
            "row_ratio": row_ratio,
            "projected_seconds": projected_seconds,
            "safety_margin": 1.2,
            "projected_seconds_with_margin": projected_with_margin,
            "budget_seconds": budget_minutes * 60,
            "passes": projected_with_margin <= budget_minutes * 60,
        }
        (output / "timing_pilot.json").write_text(
            json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not timing["passes"]:
            raise RuntimeError(
                "Timing pilot projects "
                f"{projected_with_margin / 60:.1f} minutes with margin, exceeding "
                f"the {budget_minutes:.1f}-minute budget."
            )
    development_predictions = []
    all_history = []
    calibrators = []
    development_results = {}
    for mode in SCORE_MODES:
        for architecture in ARCHITECTURES:
            spec = StaticStudySpec(
                mode,
                architecture,
                epochs,
                2026,
                ("score", "win", "embedding"),
                dev_train_keys,
                dev_eval_keys,
                1.0,
                1.0,
                1.0,
            )
            leaf = output / "development" / mode / architecture
            result = train_core_leaf(
                dev_table,
                spec,
                prior_checkpoint=prior_checkpoint,
                output_dir=leaf,
                log_dir=(
                    tensorboard_dir / "development" / mode / architecture
                    if tensorboard_dir
                    else None
                ),
                phase="development",
            )
            development_results[(mode, architecture)] = result
            history = result.history.copy()
            history.insert(0, "architecture", architecture)
            history.insert(0, "supervision_mode", mode)
            history.insert(0, "phase", "development")
            all_history.append(history)
            predictions = prediction_frame(result, spec, dev_eval_keys, phase="development")
            development_predictions.append(predictions)
            checkpoint_hash = sha256_file(leaf / "checkpoint.pt")
            calibrators.append(fit_calibrators(predictions, checkpoint_hash))
    development_prediction_table = pd.concat(development_predictions, ignore_index=True)
    development_prediction_table.to_parquet(output / "development_predictions.parquet", index=False)
    calibrator_table = pd.concat(calibrators, ignore_index=True)
    calibrator_table.to_csv(output / "calibrators.csv", index=False)

    clean_dev_train = table[table["match_key"].astype(str).isin(dev_train_keys)]
    clean_dev_holdout = table[table["match_key"].astype(str).isin(dev_eval_keys)]
    clean_final_train = table[table["match_key"].astype(str).isin(final_train_keys)]
    championship = table[table["match_key"].astype(str).isin(test_keys)].reset_index(drop=True)
    baseline_predictions, baseline_contract = _baseline_predictions(
        clean_final_train, championship, clean_dev_train, clean_dev_holdout
    )
    baseline_contract.to_csv(output / "baseline_contract.csv", index=False)
    frozen = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "season": 2026,
        "state_model": "static-z-base",
        "latent_dim": 16,
        "evidence_role": "reused-diagnostic-holdout",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "git_clean": _git_clean(),
        "epochs": epochs,
        "seed": 2026,
        "configurations": [
            {"supervision_mode": mode, "architecture": architecture}
            for mode in SCORE_MODES
            for architecture in ARCHITECTURES
        ],
        "loss_weights": {"score": 1.0, "win": 1.0, "embedding": 1.0},
        "optimizer": {
            "name": "AdamW",
            "batch_size": 256,
            "max_grad_norm": 5.0,
            "scheduler": "cosine",
            "scheduler_horizon_epochs": epochs,
            "early_stopping": False,
            "restore_best": False,
        },
        "match_key_hashes": {
            "development_train": _key_hash(dev_train_keys),
            "development_holdout": _key_hash(dev_eval_keys),
            "final_train": _key_hash(final_train_keys),
            "championship_test": _key_hash(test_keys),
        },
        "split_manifest_sha256": sha256_file(output / "split_manifest.parquet"),
        "development_predictions_sha256": sha256_file(output / "development_predictions.parquet"),
        "calibrators_sha256": sha256_file(output / "calibrators.csv"),
        "calibrators": calibrator_table.to_dict(orient="records"),
        "prior_checkpoint_sha256": sha256_file(prior_checkpoint),
        "features_sha256": sha256_file(features_path),
        "events_sha256": sha256_file(event_metadata_path),
        "historical_reference_sha256": sha256_file(historical_reference),
        "lockfile_sha256": sha256_file(REPOSITORY_ROOT / "uv.lock"),
        "relevant_source_hashes": {
            str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path)
            for path in (
                REPOSITORY_ROOT / "src/latentstrat/season/championship_study.py",
                REPOSITORY_ROOT / "src/latentstrat/season/static_core.py",
                REPOSITORY_ROOT / "src/latentstrat/season/train.py",
                REPOSITORY_ROOT / "src/latentstrat/season/model.py",
                REPOSITORY_ROOT / "src/latentstrat/config.py",
            )
        },
        "championship_event_keys": list(CHAMPIONSHIP_EVENTS),
        "einstein_event_key": EINSTEIN_EVENT,
        "budget_minutes": budget_minutes,
        "promotion_eligible": False,
        "baseline_settings": {
            "mean-total": {"target": "official alliance total"},
            "direct-total-ridge": {"alpha": 1.0, "fit_intercept": True},
            "direct-total-pridge": {
                "alpha": 1.0,
                "fit_intercept": True,
                "recency_decay": 0.7,
            },
        },
        "decision_rules": (
            "comparison-specific practical tolerances with event bootstrap and "
            "leave-one-division-out veto"
        ),
    }
    frozen_path = output / "frozen_final_contract.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    frozen_hash_before_predictions = sha256_file(frozen_path)

    final_predictions = []
    final_results = {}
    failed = []
    for mode in SCORE_MODES:
        for architecture in ARCHITECTURES:
            spec = StaticStudySpec(
                mode,
                architecture,
                epochs,
                2026,
                ("score", "win", "embedding"),
                final_train_keys,
                test_keys,
                1.0,
                1.0,
                1.0,
            )
            leaf = output / "refit" / mode / architecture
            try:
                result = train_core_leaf(
                    final_table,
                    spec,
                    prior_checkpoint=prior_checkpoint,
                    output_dir=leaf,
                    log_dir=(
                        tensorboard_dir / "refit" / mode / architecture if tensorboard_dir else None
                    ),
                    phase="refit",
                )
            except (FloatingPointError, RuntimeError) as exc:
                failed.append(
                    {"supervision_mode": mode, "architecture": architecture, "error": str(exc)}
                )
                continue
            history = result.history.copy()
            history.insert(0, "architecture", architecture)
            history.insert(0, "supervision_mode", mode)
            history.insert(0, "phase", "refit")
            all_history.append(history)
            final_results[(mode, architecture)] = (result, spec)
    for (mode, architecture), (result, spec) in final_results.items():
        predictions = prediction_frame(result, spec, test_keys, phase="test")
        leaf_calibrators = calibrator_table[
            calibrator_table["supervision_mode"].eq(mode)
            & calibrator_table["architecture"].eq(architecture)
        ]
        final_predictions.append(apply_calibrators(predictions, leaf_calibrators))
    if sha256_file(frozen_path) != frozen_hash_before_predictions:
        raise RuntimeError("The frozen final contract changed during refit.")
    complete = not failed and len(final_predictions) == 6
    neural_predictions = (
        pd.concat(final_predictions, ignore_index=True) if final_predictions else pd.DataFrame()
    )
    prediction_table = pd.concat([neural_predictions, baseline_predictions], ignore_index=True)
    if historical_reference.exists():
        historical = pd.read_parquet(historical_reference)
        historical = historical[
            historical["match_key"].astype(str).isin(test_keys)
            & historical["scope"].astype(str).eq("test")
        ].copy()
        if not historical.empty:
            historical["phase"] = "test"
            historical["supervision_mode"] = "historical-uncontrolled-reference"
            historical["architecture"] = historical["model"].astype(str)
            historical["seed"] = "historical"
            historical["epoch"] = np.nan
            historical["raw_pred_red_win_probability"] = historical["pred_red_win_probability"]
            historical["transferred_platt_red_win_probability"] = historical[
                "pred_red_win_probability"
            ]
            historical["transferred_score_red_win_probability"] = historical[
                "pred_red_win_probability"
            ]
            prediction_table = pd.concat(
                [prediction_table, historical.reindex(columns=prediction_table.columns)],
                ignore_index=True,
            )
    prediction_table = _attach_evaluation_context(prediction_table, clean_final_train, championship)
    prediction_table.to_parquet(output / "championship_predictions.parquet", index=False)
    history_table = pd.concat(all_history, ignore_index=True)
    history_table.to_csv(output / "training_history.csv", index=False)
    parameter_columns = [
        column
        for column in history_table
        if column.endswith("parameter_count") or column == "active_team_rows"
    ]
    history_table[
        ["phase", "supervision_mode", "architecture", "epoch", *parameter_columns]
    ].to_csv(output / "parameter_utilization.csv", index=False)
    phase_residuals = {}
    for phase in ("development", "refit"):
        for mode in SCORE_MODES:
            for architecture in ARCHITECTURES:
                contract_path = output / phase / mode / architecture / "contract.json"
                if contract_path.exists():
                    contract = json.loads(contract_path.read_text(encoding="utf-8"))
                    phase_residuals[f"{phase}/{mode}/{architecture}"] = contract["phase_residual"]
    (output / "phase_residuals.json").write_text(
        json.dumps(phase_residuals, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics, calibration = compute_metrics(prediction_table)
    metrics.to_csv(output / "metrics.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    score_bias_slices = metrics[
        metrics["metric"].isin(["alliance_score_rmse", "mean_bias", "relative_bias"])
        & metrics["scope"].str.startswith(
            (
                "primary-clean",
                "all-official",
                "DQ-only",
                "surrogate-only",
                "division:",
                "observation:",
            )
        )
    ].copy()
    score_bias_slices.to_csv(output / "score_bias_slices.csv", index=False)
    comparisons, bootstrap, loo = (
        paired_comparisons(prediction_table, resamples=(100 if smoke else bootstrap_resamples))
        if complete
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    loo.to_csv(output / "leave_one_division_out.csv", index=False)
    classifications = classify_comparisons(comparisons, loo) if complete else pd.DataFrame()
    classifications.to_csv(output / "decision_classifications.csv", index=False)
    coverage = {
        "filtering": filtering,
        "complete": complete,
        "failed_leaves": failed,
        "development_train_rows": len(dev_train_keys),
        "development_holdout_rows": len(dev_eval_keys),
        "final_train_rows": len(final_train_keys),
        "championship_rows": len(test_keys),
        "einstein_rows": int(manifest["split_role"].eq("excluded_einstein").sum()),
        "frozen_final_contract_sha256": frozen_hash_before_predictions,
        "elapsed_seconds": time.perf_counter() - started,
        "promotion_eligible": False,
        "evidence_role": "reused-diagnostic-holdout",
    }
    (output / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_data = {
        "question": (
            "Under a fixed 100-epoch, fixed-loss static Z_base model, does direct official-total "
            "supervision materially reduce Championship score bias without harming "
            "relative-strength or probability quality, and do teammate or opponent interaction "
            "modules add useful information beyond additive team strength?"
        ),
        "coverage": coverage,
        "complete": complete,
        "metric_rows": int(len(metrics)),
        "comparison_rows": int(len(comparisons)),
        "development_history": history_table[history_table["phase"].eq("development")].to_dict(
            orient="records"
        ),
        "final_history": history_table[history_table["phase"].eq("refit")].to_dict(
            orient="records"
        ),
        "metrics": metrics.to_dict(orient="records"),
        "paired_comparisons": comparisons.to_dict(orient="records"),
        "leave_one_division_out": loo.to_dict(orient="records"),
        "classifications": classifications.to_dict(orient="records"),
    }
    (output / "report_data.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _render_report(
        metrics=metrics,
        history=history_table,
        comparisons=comparisons,
        classifications=classifications,
        coverage=coverage,
        complete=complete,
    )
    report_bytes = report.encode("utf-8")
    artifact_report = output / "report.md"
    documentation_report = REPOSITORY_ROOT / "docs/2026-static-championship-core-report.md"
    artifact_report.write_bytes(report_bytes)
    documentation_report.write_bytes(report_bytes)
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    (output / "report_hashes.json").write_text(
        json.dumps(
            {
                "artifact_report": str(artifact_report),
                "documentation_report": str(documentation_report),
                "artifact_sha256": report_hash,
                "documentation_sha256": sha256_file(documentation_report),
                "matching": report_hash == sha256_file(documentation_report),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(
        output,
        workflow=WORKFLOW,
        artifact_version="v1",
        resolved_config=frozen,
        source_files={
            "features": features_path,
            "events": event_metadata_path,
            "prior_checkpoint": prior_checkpoint,
            "historical_reference": historical_reference,
        },
        promotion_eligible=False,
        promotion_note="One-seed fixed-100-epoch reused Championship diagnostic.",
    )
    return output
