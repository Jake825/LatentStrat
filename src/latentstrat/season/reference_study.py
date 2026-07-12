"""Nested temporal 2026 static robot-state reference study."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge

from latentstrat.artifacts import sha256_file, write_artifact_manifest
from latentstrat.baselines import match_design_matrix
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.experimental.frozen_targets import WorldModelOptions
from latentstrat.season.data import Split
from latentstrat.season.evaluate import _phase_score_predictions, _predict_batched, sigmoid
from latentstrat.season.features import (
    FeatureTrainingResult,
    read_feature_table,
    train_feature_table,
)
from latentstrat.season.statbotics_baseline import validate_statbotics_prediction_table
from latentstrat.season.train import create_tensorboard_writer, match_v5_matrices
from latentstrat.season.walk_forward import (
    _filter_sidecars,
    _sidecar_validation_metrics,
    _sidecars_with_weeks,
)

REFERENCE_WORKFLOW = "season.reference-2026-static"
REFERENCE_SCHEMA = "2026-static-robot-state-v1"
ARCHITECTURES = ("additive", "teammate-set", "full-match")
INITIALIZATIONS = ("pre-2026-prior", "random")
OFFICIAL_EVENT_TYPES = frozenset({0, 1, 2, 3, 4, 5})
LOWER_IS_BETTER = {
    "alliance_score_mae",
    "alliance_score_mse",
    "alliance_score_rmse",
    "score_differential_mae",
    "score_differential_mse",
    "score_differential_rmse",
    "winner_error_rate",
    "winner_brier",
    "winner_log_loss",
    "expected_calibration_error",
}


@dataclass(frozen=True)
class ReferenceStudyResult:
    output_dir: Path
    manifest_path: Path
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_initialization: str
    tensorboard_logdir: Path | None


def _read_table(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source, engine="pyarrow")
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported table format: {source}")


def validate_reference_rows(
    table: pd.DataFrame, event_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the fail-closed 2026 official-event contract."""

    required = {"season", "event_key", "match_key", "event_week", "red_score_raw", "blue_score_raw"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError("Reference feature table is missing: " + ", ".join(missing))
    metadata = event_metadata.rename(columns={"key": "event_key"}).copy()
    if not {"event_key", "event_type"}.issubset(metadata.columns):
        raise ValueError("Event metadata must contain event_key/key and event_type.")
    if metadata["event_key"].duplicated().any():
        raise ValueError("Event metadata contains duplicate event keys.")
    rows = table.copy()
    if "event_type" not in rows.columns:
        rows = rows.merge(
            metadata[["event_key", "event_type"]],
            on="event_key",
            how="left",
            validate="many_to_one",
        )
    season = pd.to_numeric(rows["season"], errors="coerce")
    weeks = pd.to_numeric(rows["event_week"], errors="coerce")
    event_types = pd.to_numeric(rows["event_type"], errors="coerce")
    scores = rows[["red_score_raw", "blue_score_raw"]].apply(pd.to_numeric, errors="coerce")
    played = scores.notna().all(axis=1) & (scores >= 0).all(axis=1)
    official = event_types.isin(OFFICIAL_EVENT_TYPES)
    in_season = weeks.notna() & (weeks > 0)
    is_2026 = season.eq(2026)
    standard_event = ~rows["event_key"].astype(str).str.contains("week0", case=False, regex=False)
    standard_match = (
        rows["comp_level"].astype(str).isin({"qm", "ef", "qf", "sf", "f"})
        if "comp_level" in rows.columns
        else pd.Series(True, index=rows.index)
    )
    keep = is_2026 & official & in_season & standard_event & standard_match & played
    report = {
        "input_rows": int(len(rows)),
        "included_rows": int(keep.sum()),
        "excluded_non_2026": int((~is_2026).sum()),
        "excluded_non_official": int((is_2026 & ~official).sum()),
        "excluded_missing_week": int((is_2026 & official & ~in_season).sum()),
        "excluded_week_zero_or_nonstandard": int(
            (is_2026 & official & in_season & ~(standard_event & standard_match)).sum()
        ),
        "excluded_unplayed": int((is_2026 & official & in_season & ~played).sum()),
    }
    result = rows.loc[keep].sort_values(["event_week", "sort_ordinal", "match_key"])
    if result["match_key"].duplicated().any():
        raise ValueError("Reference rows contain duplicate match keys.")
    return result.reset_index(drop=True), report


def nested_temporal_split(table: pd.DataFrame, *, train_max_week: int, holdout_week: int) -> Split:
    weeks = pd.to_numeric(table["event_week"], errors="coerce")
    train = weeks.le(train_max_week).fillna(False).to_numpy(bool)
    holdout = weeks.eq(holdout_week).fillna(False).to_numpy(bool)
    if np.any(train & holdout) or not np.any(train) or not np.any(holdout):
        raise ValueError(
            f"Invalid temporal split: train<=week {train_max_week}, holdout week {holdout_week}."
        )
    return Split(train, holdout, ~(train | holdout), "nested-temporal-reference")


def refit_temporal_split(table: pd.DataFrame, *, train_max_week: int) -> Split:
    weeks = pd.to_numeric(table["event_week"], errors="coerce")
    train = weeks.le(train_max_week).fillna(False).to_numpy(bool)
    if not np.any(train):
        raise ValueError(f"Refit through week {train_max_week} has no training rows.")
    return Split(
        train_mask=train,
        validation_mask=np.zeros(len(table), dtype=bool),
        test_mask=~train,
        policy="nested-temporal-refit-no-test-access",
    )


def _prediction_frame(
    result: FeatureTrainingResult,
    split: Split,
    opts: LatentStratOptions,
    *,
    model_name: str,
    initialization: str,
    fold_number: int,
    test_week: int,
    train_max_week: int,
    phase: str,
) -> pd.DataFrame:
    red, blue, red_event, blue_event, red_missing, blue_missing = match_v5_matrices(result.prepared)
    prediction = _predict_batched(
        result.model,
        red,
        blue,
        opts,
        red_event=red_event,
        blue_event=blue_event,
        red_missing=red_missing,
        blue_missing=blue_missing,
        return_aux=True,
    )
    predicted_phase, actual_phase = _phase_score_predictions(
        result.prepared, prediction["cont"], result.target_stats
    )
    rows = np.flatnonzero(split.validation_mask)
    source = result.prepared.iloc[rows]
    actual_total = source[["red_total_score", "blue_total_score"]].to_numpy(float)
    train_residual = (
        result.prepared[["red_total_score", "blue_total_score"]].to_numpy(float) - actual_phase
    )[split.train_mask]
    residual_mean = np.nanmean(train_residual, axis=0)
    predicted_total = predicted_phase[rows] + residual_mean
    frame = pd.DataFrame(
        {
            "scope": phase,
            "model": model_name,
            "architecture": opts.match_architecture,
            "initialization": initialization,
            "fold_number": fold_number,
            "test_week": test_week,
            "val_week": test_week,
            "train_max_week": train_max_week,
            "event_key": source["event_key"].astype(str).to_numpy(),
            "match_key": source["match_key"].astype(str).to_numpy(),
            "pred_red_total_score": predicted_total[:, 0],
            "actual_red_total_score": actual_total[:, 0],
            "pred_blue_total_score": predicted_total[:, 1],
            "actual_blue_total_score": actual_total[:, 1],
            "pred_red_win_probability": sigmoid(prediction["bin"][rows, 0]),
            "red_has_dq": source.get("red_has_dq", False),
            "blue_has_dq": source.get("blue_has_dq", False),
            "red_has_surrogate": source.get("red_has_surrogate", False),
            "blue_has_surrogate": source.get("blue_has_surrogate", False),
            "known_as_of": f"end-of-week-{train_max_week}",
        }
    )
    team_columns = [f"{color}_team_{slot}_key" for color in ("red", "blue") for slot in range(1, 4)]
    if all(column in result.prepared.columns for column in team_columns):
        for column in team_columns:
            frame[column] = source[column].astype(str).to_numpy()
        training_teams = (
            result.prepared.loc[split.train_mask, team_columns].astype(str).to_numpy().reshape(-1)
        )
        appearances = pd.Series(training_teams).value_counts()
        q33, q67 = appearances.quantile([1 / 3, 2 / 3]).to_numpy(float)
        observation_count = (
            source[team_columns]
            .astype(str)
            .apply(lambda column: column.map(appearances).fillna(0))
            .mean(axis=1)
            .to_numpy(float)
        )
        frame["mean_prior_team_observations"] = observation_count
        frame["observation_slice"] = np.where(
            observation_count <= q33,
            "low",
            np.where(observation_count <= q67, "medium", "high"),
        )
    for column in result.prepared.columns:
        if column.startswith(("red_atomic_", "blue_atomic_", "red_bonus_", "blue_bonus_")):
            frame[f"actual_{column}"] = source[column].to_numpy()
    return frame


def _baseline_frames(
    result: FeatureTrainingResult,
    split: Split,
    reference: pd.DataFrame,
    *,
    initialization: str,
    fold_number: int,
    test_week: int,
) -> list[pd.DataFrame]:
    target_index = {name: idx for idx, name in enumerate(result.report.baselines.target_names)}
    indices = [
        target_index[name]
        for name in ("red_auto_pts", "red_teleop_pts", "blue_auto_pts", "blue_teleop_pts")
    ]
    actual_phase = result.prepared[
        ["red_auto_pts", "red_teleop_pts", "blue_auto_pts", "blue_teleop_pts"]
    ].to_numpy(float)
    train_actual_total = result.prepared.loc[
        split.train_mask, ["red_total_score", "blue_total_score"]
    ].to_numpy(float)
    train_phase = np.column_stack(
        [
            actual_phase[split.train_mask, :2].sum(axis=1),
            actual_phase[split.train_mask, 2:].sum(axis=1),
        ]
    )
    residual = np.nanmean(train_actual_total - train_phase, axis=0)
    design = match_design_matrix(result.prepared)
    phase_targets = result.prepared[
        ["red_auto_pts", "red_teleop_pts", "blue_auto_pts", "blue_teleop_pts"]
    ].to_numpy(float)
    train_weeks = pd.to_numeric(
        result.prepared.loc[split.train_mask, "event_week"], errors="coerce"
    ).to_numpy(float)
    max_week = float(np.nanmax(train_weeks))
    recency_weight = np.power(0.7, np.maximum(max_week - train_weeks, 0))
    rolling_model = Ridge(
        alpha=float(result.model.opts.l2), fit_intercept=False, solver="lsqr"
    ).fit(
        design[split.train_mask],
        phase_targets[split.train_mask],
        sample_weight=recency_weight,
    )
    rolling_predictions = np.asarray(rolling_model.predict(design))
    frames = []
    for model_name, predictions in (
        ("mean", result.report.baselines.mean.predictions),
        ("ridge", result.report.baselines.ridge.predictions),
        ("rolling-pridge", rolling_predictions),
    ):
        selected = predictions[split.validation_mask][:, indices]
        total = np.column_stack([selected[:, :2].sum(axis=1), selected[:, 2:].sum(axis=1)])
        total += residual
        frame = reference.copy()
        frame["model"] = model_name
        frame["architecture"] = "additive-control"
        frame["initialization"] = initialization
        frame["pred_red_total_score"] = total[:, 0]
        frame["pred_blue_total_score"] = total[:, 1]
        train_predictions = predictions[split.train_mask][:, indices]
        train_diff = train_predictions[:, :2].sum(axis=1) - train_predictions[:, 2:].sum(axis=1)
        actual_win = pd.to_numeric(
            result.prepared.loc[split.train_mask, "red_win"], errors="coerce"
        )
        valid = np.isfinite(train_diff) & np.isfinite(actual_win)
        if valid.sum() and np.unique(actual_win[valid]).size == 2:
            calibrator = LogisticRegression(random_state=2026).fit(
                train_diff[valid, None], actual_win[valid].astype(int)
            )
            probability = calibrator.predict_proba((total[:, 0] - total[:, 1])[:, None])[:, 1]
        else:
            probability = np.full(len(total), 0.5)
        frame["pred_red_win_probability"] = probability
        frames.append(frame)
    return frames


def _fit_probability_calibrator(predictions: pd.DataFrame) -> dict[str, float | str]:
    probability = predictions["pred_red_win_probability"].to_numpy(float)
    actual_red = predictions["actual_red_total_score"].to_numpy(float)
    actual_blue = predictions["actual_blue_total_score"].to_numpy(float)
    outcome = (actual_red > actual_blue).astype(int)
    valid = (
        np.isfinite(probability)
        & np.isfinite(actual_red)
        & np.isfinite(actual_blue)
        & (actual_red != actual_blue)
    )
    if valid.sum() < 2 or np.unique(outcome[valid]).size != 2:
        return {
            "method": "identity-insufficient-selection-classes",
            "coefficient": 1.0,
            "intercept": 0.0,
        }
    clipped = np.clip(probability[valid], 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped))
    model = LogisticRegression(random_state=2026).fit(logits[:, None], outcome[valid])
    return {
        "method": "platt-logit-week-t-minus-1",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def _apply_probability_calibrator(
    predictions: pd.DataFrame, calibrator: dict[str, float | str]
) -> pd.DataFrame:
    result = predictions.copy()
    probability = np.clip(result["pred_red_win_probability"].to_numpy(float), 1e-7, 1 - 1e-7)
    logits = np.log(probability / (1 - probability))
    calibrated = sigmoid(float(calibrator["coefficient"]) * logits + float(calibrator["intercept"]))
    result["uncalibrated_pred_red_win_probability"] = probability
    result["pred_red_win_probability"] = calibrated
    result["calibration_method"] = str(calibrator["method"])
    return result


def _metric_rows(frame: pd.DataFrame, *, scope: str) -> list[dict[str, Any]]:
    actual_red = frame["actual_red_total_score"].to_numpy(float)
    actual_blue = frame["actual_blue_total_score"].to_numpy(float)
    predicted_red = frame["pred_red_total_score"].to_numpy(float)
    predicted_blue = frame["pred_blue_total_score"].to_numpy(float)
    alliance_error = np.concatenate([predicted_red - actual_red, predicted_blue - actual_blue])
    differential_error = (predicted_red - predicted_blue) - (actual_red - actual_blue)
    tie = actual_red == actual_blue
    actual_win = (actual_red > actual_blue).astype(float)
    probabilities = frame["pred_red_win_probability"].to_numpy(float)
    valid_win = ~tie & np.isfinite(probabilities)
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    values: dict[str, tuple[float, int]] = {
        "alliance_score_mae": (
            float(np.nanmean(np.abs(alliance_error))),
            int(np.isfinite(alliance_error).sum()),
        ),
        "alliance_score_mse": (
            float(np.nanmean(alliance_error**2)),
            int(np.isfinite(alliance_error).sum()),
        ),
        "score_differential_mae": (
            float(np.nanmean(np.abs(differential_error))),
            int(np.isfinite(differential_error).sum()),
        ),
        "score_differential_mse": (
            float(np.nanmean(differential_error**2)),
            int(np.isfinite(differential_error).sum()),
        ),
    }
    values["alliance_score_rmse"] = (
        math.sqrt(values["alliance_score_mse"][0]),
        values["alliance_score_mse"][1],
    )
    values["score_differential_rmse"] = (
        math.sqrt(values["score_differential_mse"][0]),
        values["score_differential_mse"][1],
    )
    if valid_win.any():
        outcome = actual_win[valid_win]
        probability = probabilities[valid_win]
        correct = (probability >= 0.5) == outcome
        values.update(
            {
                "winner_accuracy": (float(np.mean(correct)), int(valid_win.sum())),
                "winner_error_rate": (float(np.mean(~correct)), int(valid_win.sum())),
                "winner_brier": (
                    float(np.mean((probability - outcome) ** 2)),
                    int(valid_win.sum()),
                ),
                "winner_log_loss": (
                    float(
                        np.mean(
                            -(
                                outcome * np.log(probability)
                                + (1 - outcome) * np.log(1 - probability)
                            )
                        )
                    ),
                    int(valid_win.sum()),
                ),
            }
        )
    identity = frame.iloc[0]
    return [
        {
            "scope": scope,
            "model": identity["model"],
            "initialization": identity["initialization"],
            "fold_number": identity.get("fold_number"),
            "test_week": identity.get("test_week"),
            "event_key": identity.get("event_key") if scope == "event" else None,
            "metric": metric,
            "value": value,
            "count": count,
            "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
        }
        for metric, (value, count) in values.items()
    ]


def reference_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, frame in predictions.groupby(
        ["scope", "model", "initialization", "fold_number", "test_week"], dropna=False
    ):
        rows.extend(_metric_rows(frame, scope=str(frame["scope"].iloc[0])))
        for _, event in frame.groupby("event_key"):
            rows.extend(_metric_rows(event, scope="event"))
    final = predictions[predictions["scope"] == "test"]
    for _, frame in final.groupby(["model", "initialization"]):
        rows.extend(_metric_rows(frame, scope="aggregate"))
    if "observation_slice" in final.columns:
        for _, frame in final.groupby(
            ["model", "initialization", "observation_slice"], dropna=False
        ):
            slice_rows = _metric_rows(frame, scope="observation_slice")
            for row in slice_rows:
                row["observation_slice"] = frame["observation_slice"].iloc[0]
            rows.extend(slice_rows)
    return pd.DataFrame(rows)


def reference_calibration(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for keys, frame in predictions.groupby(["scope", "model", "test_week"], dropna=False):
        actual_red = frame["actual_red_total_score"].to_numpy(float)
        actual_blue = frame["actual_blue_total_score"].to_numpy(float)
        probability = frame["pred_red_win_probability"].to_numpy(float)
        valid = (actual_red != actual_blue) & np.isfinite(probability)
        outcome = (actual_red[valid] > actual_blue[valid]).astype(float)
        probability = probability[valid]
        which = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, bins - 1)
        bins_out = []
        for index in range(bins):
            mask = which == index
            mean = float(np.mean(probability[mask])) if mask.any() else np.nan
            observed = float(np.mean(outcome[mask])) if mask.any() else np.nan
            bins_out.append((int(mask.sum()), mean, observed, mean - observed))
        ece = sum(
            count / max(len(probability), 1) * abs(gap) for count, _, _, gap in bins_out if count
        )
        for index, (count, mean, observed, gap) in enumerate(bins_out):
            rows.append(
                {
                    "scope": keys[0],
                    "model": keys[1],
                    "test_week": keys[2],
                    "bin_low": edges[index],
                    "bin_high": edges[index + 1],
                    "count": count,
                    "mean_probability": mean,
                    "observed_rate": observed,
                    "gap": gap,
                    "expected_calibration_error": ece,
                }
            )
    return pd.DataFrame(rows)


def _loss_table(frame: pd.DataFrame) -> pd.DataFrame:
    actual_red = frame["actual_red_total_score"].to_numpy(float)
    actual_blue = frame["actual_blue_total_score"].to_numpy(float)
    pred_red = frame["pred_red_total_score"].to_numpy(float)
    pred_blue = frame["pred_blue_total_score"].to_numpy(float)
    probability = np.clip(frame["pred_red_win_probability"].to_numpy(float), 1e-7, 1 - 1e-7)
    actual_win = (actual_red > actual_blue).astype(float)
    valid = actual_red != actual_blue
    red_error = pred_red - actual_red
    blue_error = pred_blue - actual_blue
    differential_error = (pred_red - pred_blue) - (actual_red - actual_blue)
    return pd.DataFrame(
        {
            "alliance_score_absolute_error": 0.5 * (np.abs(red_error) + np.abs(blue_error)),
            "alliance_score_squared_error": 0.5 * (red_error**2 + blue_error**2),
            "score_differential_absolute_error": np.abs(differential_error),
            "score_differential_squared_error": differential_error**2,
            "winner_brier": np.where(valid, (probability - actual_win) ** 2, np.nan),
            "winner_log_loss": np.where(
                valid,
                -(actual_win * np.log(probability) + (1 - actual_win) * np.log(1 - probability)),
                np.nan,
            ),
            "winner_error": np.where(valid, (probability >= 0.5) != actual_win, np.nan),
        }
    )


def paired_event_bootstrap(
    predictions: pd.DataFrame, *, resamples: int = 2_000, seed: int = 2026
) -> pd.DataFrame:
    rows = []
    comparisons = (
        ("teammate-set", "additive"),
        ("full-match", "teammate-set"),
        ("full-match", "additive"),
        ("full-match", "statbotics"),
    )
    for week, week_rows in predictions[predictions["scope"] == "test"].groupby("test_week"):
        for candidate_name, baseline_name in comparisons:
            candidate = week_rows[week_rows["model"] == candidate_name]
            baseline = week_rows[week_rows["model"] == baseline_name]
            if candidate.empty or baseline.empty:
                continue
            paired = candidate.merge(
                baseline,
                on="match_key",
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            if paired.empty:
                continue
            candidate_loss = _loss_table(
                pd.DataFrame(
                    {
                        column: paired[f"{column}_candidate"]
                        for column in (
                            "actual_red_total_score",
                            "actual_blue_total_score",
                            "pred_red_total_score",
                            "pred_blue_total_score",
                            "pred_red_win_probability",
                        )
                    }
                )
            )
            baseline_loss = _loss_table(
                pd.DataFrame(
                    {
                        column: paired[f"{column}_baseline"]
                        for column in (
                            "actual_red_total_score",
                            "actual_blue_total_score",
                            "pred_red_total_score",
                            "pred_blue_total_score",
                            "pred_red_win_probability",
                        )
                    }
                )
            )
            delta = candidate_loss - baseline_loss
            events = paired["event_key_candidate"].astype(str).to_numpy()
            unique = np.unique(events)
            rng = np.random.default_rng(seed + int(week))
            samples = {column: [] for column in delta.columns}
            for _ in range(resamples):
                selected = rng.choice(unique, len(unique), replace=True)
                indices = np.concatenate([np.flatnonzero(events == event) for event in selected])
                for column in delta:
                    samples[column].append(float(np.nanmean(delta[column].to_numpy()[indices])))
            for metric, values in samples.items():
                distribution = np.asarray(values)
                rows.append(
                    {
                        "test_week": int(week),
                        "candidate": candidate_name,
                        "baseline": baseline_name,
                        "metric": metric,
                        "candidate_loss_mean": float(np.nanmean(candidate_loss[metric])),
                        "baseline_loss_mean": float(np.nanmean(baseline_loss[metric])),
                        "delta_mean": float(np.nanmean(delta[metric])),
                        "ci_low": float(np.nanquantile(distribution, 0.025)),
                        "ci_high": float(np.nanquantile(distribution, 0.975)),
                        "probability_of_improvement": float(np.nanmean(distribution < 0)),
                        "resamples": resamples,
                        "seed": seed,
                        "cluster": "event_key",
                    }
                )
    return pd.DataFrame(rows)


def _write_checkpoint(
    result: FeatureTrainingResult,
    path: Path,
    *,
    architecture: str,
    initialization: str,
    fold: int,
    test_week: int,
    calibrator: dict[str, float | str],
    study_contract: str = REFERENCE_SCHEMA,
    seed: int | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_columns = [column for column in result.prepared.columns if column.endswith("_base_idx")]
    active_rows = sorted(
        {
            int(value)
            for value in result.prepared[base_columns].to_numpy().ravel()
            if pd.notna(value) and int(value) != 0
        }
    )
    payload = {
        "checkpoint_schema_version": 7,
        "study_contract": study_contract,
        "model_state_dict": result.model.state_dict(),
        "options": result.model.opts.model_dump(mode="json"),
        "world_model": WorldModelOptions(enabled=False).model_dump(mode="json"),
        "team_base_index_map": result.team_index_map,
        "team_event_index_map": {},
        "target_stats": {
            "target_names": result.target_stats.target_names,
            "mu": result.target_stats.mu.tolist(),
            "sigma": result.target_stats.sigma.tolist(),
        },
        "architecture": architecture,
        "initialization": initialization,
        "fold_number": fold,
        "test_week": test_week,
        "active_z_base_rows": active_rows,
        "probability_calibration": calibrator,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if extra_payload:
        payload.update(extra_payload)
    torch.save(payload, path)
    (path.with_suffix(".manifest.json")).write_text(
        json.dumps(
            {
                "schema": study_contract,
                "checkpoint": path.name,
                "sha256": sha256_file(path),
                "architecture": architecture,
                "initialization": initialization,
                "fold_number": fold,
                "test_week": test_week,
                "seed": seed,
                "probability_calibration": calibrator,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _tensorboard_contract(writer: Any, contract: dict[str, Any]) -> None:
    writer.add_text(
        "Study/ResolvedContract", "```json\n" + json.dumps(contract, indent=2) + "\n```", 0
    )
    options = contract.get("options") or {}
    hparams = {
        "architecture": str(contract["architecture"]),
        "initialization": str(contract["initialization"]),
        "state_model": "static-z-base",
        "latent_dim": 16,
        "seed": int(contract["seed"]),
        "train_max_week": int(contract["train_max_week"]),
        "holdout_week": int(contract["holdout_week"]),
        "batch_size": int(options.get("mini_batch_size", 0)),
        "gradient_accumulation_steps": int(options.get("gradient_accumulation_steps", 1)),
        "scheduler": str(options.get("scheduler", "unknown")),
        "max_grad_norm": float(options.get("max_grad_norm", 0)),
        "weight_decay": float(options.get("l2", 0)),
        "team_dropout_rate": float(options.get("team_dropout_rate", 0)),
    }
    writer.add_hparams(hparams, {"hparam/contract_written": 1.0}, run_name="contract")


def _train_once(
    table: pd.DataFrame,
    opts: LatentStratOptions,
    split: Split,
    *,
    prior_checkpoint: Path | None,
    sidecars: dict[str, pd.DataFrame],
    logdir: Path | None,
    contract: dict[str, Any],
) -> FeatureTrainingResult:
    writer = create_tensorboard_writer(str(logdir)) if logdir is not None else None
    try:
        if writer is not None:
            _tensorboard_contract(writer, contract)
        result = train_feature_table(
            table,
            opts,
            verbose=False,
            prior_checkpoint=prior_checkpoint,
            sidecar_tables=sidecars,
            split_override=split,
            tensorboard_writer=writer,
            tensorboard_logdir=logdir,
        )
        if writer is not None and not result.history.empty:
            final = result.history.iloc[-1]
            hparams = {
                "architecture": str(contract["architecture"]),
                "initialization": str(contract["initialization"]),
                "phase": str(contract["phase"]),
                "test_week": int(contract.get("test_week", contract["holdout_week"])),
                "nominal_parameters": int(final.get("nominal_parameter_count", 0)),
                "trainable_parameters": int(final.get("trainable_parameter_count", 0)),
                "gradient_receiving_parameters": int(
                    final.get("gradient_receiving_parameter_count", 0)
                ),
            }
            outcomes = {
                f"final/{name.removeprefix('validation_')}": float(value)
                for name, value in final.items()
                if str(name).startswith("validation_")
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
            }
            writer.add_hparams(hparams, outcomes or {"final/completed": 1.0}, run_name="outcomes")
        return result
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def _selection_tuple(predictions: pd.DataFrame, auxiliary: dict[str, float]) -> tuple[float, ...]:
    metrics = reference_metrics(predictions)
    values = metrics[metrics["scope"] == "development"].set_index("metric")["value"]
    auxiliary_values = [value for value in auxiliary.values() if np.isfinite(value)]
    return (
        float(values.get("score_differential_rmse", np.inf)),
        float(values.get("winner_brier", np.inf)),
        float(values.get("winner_log_loss", np.inf)),
        float(np.mean(auxiliary_values)) if auxiliary_values else np.inf,
    )


def run_reference_study(
    features_path: str | Path,
    event_metadata_path: str | Path,
    output_dir: str | Path,
    *,
    prior_checkpoint: str | Path | None = None,
    sidecar_paths: dict[str, str | Path] | None = None,
    statbotics_path: str | Path | None = None,
    tensorboard_root: str | Path | None = "runs/reference-2026-static",
    epochs: int = 20,
    budget_minutes: float = 100,
    test_weeks: Iterable[int] = (6, 8, 10),
    bootstrap_resamples: int = 2_000,
    smoke: bool = False,
) -> ReferenceStudyResult:
    """Run the guarded static reference matrix; use ``smoke`` before the full campaign."""

    features_path = Path(features_path)
    event_metadata_path = Path(event_metadata_path)
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
    if smoke:
        tests = tests[:1]
        epochs = min(epochs, 2)
    preliminary_sources = {"features": features_path, "event_metadata": event_metadata_path}
    if prior_checkpoint:
        preliminary_sources["prior_checkpoint"] = Path(prior_checkpoint)
    if statbotics_path:
        preliminary_sources["statbotics"] = Path(statbotics_path)
    preliminary_sources.update({name: Path(path) for name, path in sidecar_paths.items()})
    preliminary_coverage = {
        **filtering,
        "complete": False,
        "stage": "initialization",
        "test_weeks": list(tests),
    }
    (output / "coverage.json").write_text(
        json.dumps(preliminary_coverage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preliminary_manifest_path = write_artifact_manifest(
        output,
        workflow=REFERENCE_WORKFLOW,
        artifact_version="v1",
        resolved_config={
            "study_id": output.name,
            "artifact_schema": REFERENCE_SCHEMA,
            "season": 2026,
            "state_model": "static-z-base",
            "complete": False,
            "smoke": smoke,
            "test_weeks": list(tests),
        },
        source_files=preliminary_sources,
        promotion_eligible=False,
        promotion_note="Incomplete reference matrix; architectural conclusions are unavailable.",
    )
    preliminary_manifest = json.loads(preliminary_manifest_path.read_text(encoding="utf-8"))
    study_provenance = {
        "git_commit": preliminary_manifest.get("git_commit"),
        "sources": preliminary_manifest.get("sources") or {},
        "target_set": "native-2026-match-breakdown-and-official-outcomes",
        "auxiliary_tasks": sorted(sidecars),
    }
    for week in tests:
        week_rows = table[pd.to_numeric(table["event_week"], errors="coerce") == week]
        if week_rows.empty:
            raise ValueError(f"Test week {week} has no played official matches.")
        if not smoke and (len(week_rows) < 500 or week_rows["event_key"].nunique() < 5):
            raise ValueError(f"Test week {week} fails the 500-match/five-event evidence gate.")
    study_id = output.name
    tensorboard_dir = Path(tensorboard_root) / study_id if tensorboard_root else None
    base_opts = default_options(2026).model_copy(
        update={
            "state_model": "static-z-base",
            "latent_dim": 16,
            "random_seed": 2026,
            "device": "cpu",
            "epochs": epochs,
            "use_early_stopping": True,
            "restore_best_validation_model": True,
        }
    )
    timing_pilot_seconds = 0.0
    common_epoch_cap = epochs
    if not smoke:
        pilot_holdout_week = max(tests) - 1
        pilot_split = nested_temporal_split(
            table,
            train_max_week=pilot_holdout_week - 1,
            holdout_week=pilot_holdout_week,
        )
        pilot_opts = base_opts.model_copy(
            update={
                "match_architecture": "full-match",
                "epochs": 2,
                "use_early_stopping": False,
                "restore_best_validation_model": False,
            }
        )
        pilot_contract = {
            "study_id": study_id,
            "architecture": "full-match",
            "initialization": "random",
            "seed": 2026,
            "train_max_week": pilot_holdout_week - 1,
            "holdout_week": pilot_holdout_week,
            "phase": "timing-pilot",
            "known_as_of": f"end-of-week-{pilot_holdout_week - 1}",
            "options": pilot_opts.model_dump(mode="json"),
            **study_provenance,
        }
        pilot_started = time.perf_counter()
        _train_once(
            table,
            pilot_opts,
            pilot_split,
            prior_checkpoint=None,
            sidecars=_filter_sidecars(sidecars, max_week=pilot_holdout_week - 1),
            logdir=None,
            contract=pilot_contract,
        )
        timing_pilot_seconds = time.perf_counter() - pilot_started
        pilot_seconds_per_epoch = timing_pilot_seconds / 2
        pilot_train_rows = max(int(pilot_split.train_mask.sum()), 1)
        max_train_rows = int(
            (pd.to_numeric(table["event_week"], errors="coerce") <= max(tests) - 1).sum()
        )
        scaled_seconds_per_epoch = pilot_seconds_per_epoch * max(
            max_train_rows / pilot_train_rows, 1.0
        )
        initialization_count = 2 if prior_checkpoint else 1
        phase_count = initialization_count + 2 * len(ARCHITECTURES) * len(tests)
        available_seconds = budget_minutes * 60 - timing_pilot_seconds
        common_epoch_cap = min(
            epochs,
            int(available_seconds // max(scaled_seconds_per_epoch * phase_count, 1e-9)),
        )
        if common_epoch_cap < 1:
            raise RuntimeError(
                "The two-epoch timing pilot shows that even a one-epoch complete matrix "
                f"cannot fit within the {budget_minutes:.1f}-minute CPU budget."
            )
        base_opts = base_opts.model_copy(update={"epochs": common_epoch_cap})
    development_predictions = []
    development_rows = []
    prior_path = Path(prior_checkpoint) if prior_checkpoint else None
    for initialization in INITIALIZATIONS:
        if initialization == "pre-2026-prior" and prior_path is None:
            continue
        split = nested_temporal_split(table, train_max_week=3, holdout_week=4)
        opts = base_opts.model_copy(update={"match_architecture": "full-match"})
        contract = {
            "study_id": study_id,
            "architecture": "full-match",
            "initialization": initialization,
            "seed": 2026,
            "train_max_week": 3,
            "holdout_week": 4,
            "phase": "development",
            "known_as_of": "end-of-week-3",
            "options": opts.model_dump(mode="json"),
            **study_provenance,
        }
        train_sidecars = _filter_sidecars(sidecars, max_week=3)
        result = _train_once(
            table,
            opts,
            split,
            prior_checkpoint=prior_path if initialization == "pre-2026-prior" else None,
            sidecars=train_sidecars,
            logdir=(tensorboard_dir / "development" / initialization) if tensorboard_dir else None,
            contract=contract,
        )
        prediction = _prediction_frame(
            result,
            split,
            opts,
            model_name="full-match",
            initialization=initialization,
            fold_number=0,
            test_week=4,
            train_max_week=3,
            phase="development",
        )
        auxiliary = _sidecar_validation_metrics(
            result, _filter_sidecars(sidecars, exact_week=4), opts
        )
        selection = _selection_tuple(prediction, auxiliary)
        development_rows.append(
            {
                "initialization": initialization,
                "score_differential_rmse": selection[0],
                "winner_brier": selection[1],
                "winner_log_loss": selection[2],
                "auxiliary_probe_loss": selection[3],
                "best_epoch": result.diagnostics.best_epoch or len(result.history),
            }
        )
        development_predictions.append(prediction)
    if not development_rows:
        raise ValueError("No initialization candidate could be evaluated.")
    initialization_table = pd.DataFrame(development_rows).sort_values(
        ["score_differential_rmse", "winner_brier", "winner_log_loss", "auxiliary_probe_loss"]
    )
    selected_initialization = str(initialization_table.iloc[0]["initialization"])
    initialization_table["selected"] = initialization_table["initialization"].eq(
        selected_initialization
    )

    predictions = list(development_predictions)
    histories = []
    auxiliary_rows = []
    parameter_rows = []
    fold_number = 0
    for architecture in ARCHITECTURES:
        for test_week in tests:
            fold_number += 1
            selection_split = nested_temporal_split(
                table, train_max_week=test_week - 2, holdout_week=test_week - 1
            )
            selection_opts = base_opts.model_copy(update={"match_architecture": architecture})
            init_checkpoint = prior_path if selected_initialization == "pre-2026-prior" else None
            contract = {
                "study_id": study_id,
                "architecture": architecture,
                "initialization": selected_initialization,
                "seed": 2026,
                "train_max_week": test_week - 2,
                "holdout_week": test_week - 1,
                "test_week": test_week,
                "phase": "selection",
                "known_as_of": f"end-of-week-{test_week - 2}",
                "options": selection_opts.model_dump(mode="json"),
                **study_provenance,
            }
            selection_result = _train_once(
                table,
                selection_opts,
                selection_split,
                prior_checkpoint=init_checkpoint,
                sidecars=_filter_sidecars(sidecars, max_week=test_week - 2),
                logdir=(tensorboard_dir / architecture / f"fold_{test_week}" / "selection")
                if tensorboard_dir
                else None,
                contract=contract,
            )
            calibration_predictions = _prediction_frame(
                selection_result,
                selection_split,
                selection_opts,
                model_name=architecture,
                initialization=selected_initialization,
                fold_number=fold_number,
                test_week=test_week - 1,
                train_max_week=test_week - 2,
                phase="selection",
            )
            calibrator = _fit_probability_calibrator(calibration_predictions)
            selected_epoch = selection_result.diagnostics.best_epoch or len(
                selection_result.history
            )
            refit_opts = selection_opts.model_copy(
                update={
                    "epochs": int(selected_epoch),
                    "use_early_stopping": False,
                    "restore_best_validation_model": False,
                }
            )
            prediction_split = nested_temporal_split(
                table, train_max_week=test_week - 1, holdout_week=test_week
            )
            refit_split = refit_temporal_split(table, train_max_week=test_week - 1)
            refit_contract = {
                **contract,
                "train_max_week": test_week - 1,
                "holdout_week": test_week,
                "phase": "refit",
                "selected_epoch_count": int(selected_epoch),
                "known_as_of": f"end-of-week-{test_week - 1}",
                "options": refit_opts.model_dump(mode="json"),
            }
            refit_result = _train_once(
                table,
                refit_opts,
                refit_split,
                prior_checkpoint=init_checkpoint,
                sidecars=_filter_sidecars(sidecars, max_week=test_week - 1),
                logdir=(tensorboard_dir / architecture / f"fold_{test_week}" / "refit")
                if tensorboard_dir
                else None,
                contract=refit_contract,
            )
            test_predictions = _prediction_frame(
                refit_result,
                prediction_split,
                refit_opts,
                model_name=architecture,
                initialization=selected_initialization,
                fold_number=fold_number,
                test_week=test_week,
                train_max_week=test_week - 1,
                phase="test",
            )
            test_predictions = _apply_probability_calibrator(test_predictions, calibrator)
            predictions.append(test_predictions)
            if architecture == "full-match":
                predictions.extend(
                    _baseline_frames(
                        refit_result,
                        prediction_split,
                        test_predictions,
                        initialization=selected_initialization,
                        fold_number=fold_number,
                        test_week=test_week,
                    )
                )
            auxiliary = _sidecar_validation_metrics(
                refit_result, _filter_sidecars(sidecars, exact_week=test_week), refit_opts
            )
            auxiliary_rows.extend(
                {
                    "scope": "test",
                    "model": architecture,
                    "initialization": selected_initialization,
                    "fold_number": fold_number,
                    "test_week": test_week,
                    "metric": metric,
                    "value": value,
                    "direction": "lower",
                }
                for metric, value in auxiliary.items()
            )
            for phase_name, history in (
                ("selection", selection_result.history),
                ("refit", refit_result.history),
            ):
                history = history.copy()
                history.insert(0, "phase", phase_name)
                history.insert(0, "test_week", test_week)
                history.insert(0, "fold_number", fold_number)
                history.insert(0, "model", architecture)
                histories.append(history)
            final_history = refit_result.history.iloc[-1]
            utilization_base = {
                "model": architecture,
                "fold_number": fold_number,
                "test_week": test_week,
                "active_z_base_rows": int(final_history["active_team_rows"]),
            }
            parameter_rows.append(
                {
                    **utilization_base,
                    "module": "<all>",
                    "nominal_parameters": int(final_history["nominal_parameter_count"]),
                    "trainable_parameters": int(final_history["trainable_parameter_count"]),
                    "gradient_receiving_parameters": int(
                        final_history["gradient_receiving_parameter_count"]
                    ),
                }
            )
            nominal_suffix = "_nominal_parameter_count"
            modules = {
                str(column)[len("module_") : -len(nominal_suffix)]
                for column in final_history.index
                if str(column).startswith("module_") and str(column).endswith(nominal_suffix)
            }
            for module in sorted(modules):
                parameter_rows.append(
                    {
                        **utilization_base,
                        "module": module,
                        "nominal_parameters": int(
                            final_history[f"module_{module}_nominal_parameter_count"]
                        ),
                        "trainable_parameters": int(
                            final_history[f"module_{module}_trainable_parameter_count"]
                        ),
                        "gradient_receiving_parameters": int(
                            final_history[f"module_{module}_gradient_receiving_parameter_count"]
                        ),
                    }
                )
            checkpoint = output / "checkpoints" / architecture / f"fold_{test_week}.pt"
            _write_checkpoint(
                refit_result,
                checkpoint,
                architecture=architecture,
                initialization=selected_initialization,
                fold=fold_number,
                test_week=test_week,
                calibrator=calibrator,
            )

    prediction_table = pd.concat(predictions, ignore_index=True)
    evaluation_table = prediction_table
    statbotics_coverage: dict[str, Any] = {"status": "not-provided"}
    if statbotics_path is not None:
        statbotics = validate_statbotics_prediction_table(
            pd.read_parquet(statbotics_path, engine="pyarrow"), 2026
        )
        truth = prediction_table[
            (prediction_table["scope"] == "test") & (prediction_table["model"] == "full-match")
        ]
        candidate_keys = set(truth["match_key"].astype(str))
        statbotics_keys = set(statbotics["match_key"].astype(str))
        statbotics_coverage = {
            "status": "provided",
            "candidate_rows": len(candidate_keys),
            "statbotics_rows": len(statbotics_keys),
            "matched_rows": len(candidate_keys & statbotics_keys),
            "candidate_only_rows": len(candidate_keys - statbotics_keys),
            "statbotics_only_rows": len(statbotics_keys - candidate_keys),
        }
        stat = truth.merge(
            statbotics,
            on="match_key",
            how="inner",
            suffixes=("", "_statbotics"),
            validate="one_to_one",
        )
        if stat.empty:
            raise ValueError(
                "The reference predictions and Statbotics artifact have no matching match keys."
            )
        if not stat.empty:
            stat["model"] = "statbotics"
            stat["architecture"] = "external-control"
            stat["pred_red_total_score"] = stat["statbotics_pred_red_score"]
            stat["pred_blue_total_score"] = stat["statbotics_pred_blue_score"]
            stat["pred_red_win_probability"] = stat["statbotics_pred_red_win_probability"]
            prediction_table = pd.concat(
                [prediction_table, stat[prediction_table.columns]], ignore_index=True
            )
            matched_keys = candidate_keys & statbotics_keys
            evaluation_table = prediction_table[
                prediction_table["scope"].ne("test")
                | prediction_table["match_key"].astype(str).isin(matched_keys)
            ].reset_index(drop=True)

    metrics = reference_metrics(evaluation_table)
    calibration = reference_calibration(evaluation_table)
    bootstrap = paired_event_bootstrap(evaluation_table, resamples=bootstrap_resamples, seed=2026)
    complete = (
        (not smoke)
        and set(tests) == {6, 8, 10}
        and all(
            ((prediction_table["scope"] == "test") & (prediction_table["model"] == name)).any()
            for name in ARCHITECTURES
        )
    )
    coverage = {
        **filtering,
        "complete": complete,
        "prediction_rows": int((prediction_table["scope"] == "test").sum()),
        "test_match_rows": int(
            prediction_table.loc[prediction_table["scope"] == "test", "match_key"].nunique()
        ),
        "evaluation_match_rows": int(
            evaluation_table.loc[evaluation_table["scope"] == "test", "match_key"].nunique()
        ),
        "ties": int(
            (
                prediction_table.loc[prediction_table["scope"] == "test", "actual_red_total_score"]
                == prediction_table.loc[
                    prediction_table["scope"] == "test", "actual_blue_total_score"
                ]
            ).sum()
        ),
        "dq_rows": int(
            prediction_table.loc[prediction_table["scope"] == "test", ["red_has_dq", "blue_has_dq"]]
            .fillna(False)
            .any(axis=1)
            .sum()
        ),
        "surrogate_rows": int(
            prediction_table.loc[
                prediction_table["scope"] == "test", ["red_has_surrogate", "blue_has_surrogate"]
            ]
            .fillna(False)
            .any(axis=1)
            .sum()
        ),
        "statbotics": statbotics_coverage,
        "missing_prediction_fields": {
            column: int(
                prediction_table.loc[prediction_table["scope"] == "test", column].isna().sum()
            )
            for column in (
                "pred_red_total_score",
                "pred_blue_total_score",
                "pred_red_win_probability",
            )
        },
        "missing_target_fields": {
            column: int(
                prediction_table.loc[prediction_table["scope"] == "test", column].isna().sum()
            )
            for column in ("actual_red_total_score", "actual_blue_total_score")
        },
    }
    prediction_table.to_parquet(output / "walk_forward_predictions.parquet", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(auxiliary_rows).to_csv(output / "auxiliary_metrics.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / "training_history.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output / "parameter_utilization.csv", index=False)
    initialization_table.to_csv(output / "initialization_selection.csv", index=False)
    (output / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sources = {"features": features_path, "event_metadata": event_metadata_path}
    if prior_path:
        sources["prior_checkpoint"] = prior_path
    if statbotics_path:
        sources["statbotics"] = Path(statbotics_path)
    sources.update({name: Path(path) for name, path in sidecar_paths.items()})
    manifest_path = write_artifact_manifest(
        output,
        workflow=REFERENCE_WORKFLOW,
        artifact_version="v1",
        resolved_config={
            "study_id": study_id,
            "artifact_schema": REFERENCE_SCHEMA,
            "season": 2026,
            "state_model": "static-z-base",
            "latent_dim": 16,
            "architectures": list(ARCHITECTURES),
            "initialization_candidates": list(INITIALIZATIONS),
            "selected_initialization": selected_initialization,
            "development_week": 4,
            "test_weeks": list(tests),
            "seed": 2026,
            "cpu_budget_minutes": budget_minutes,
            "timing_pilot_seconds": timing_pilot_seconds,
            "common_epoch_cap": common_epoch_cap,
            "known_as_of_policy": (
                "train through T-2; select on T-1; reinitialize and refit through T-1; freeze for T"
            ),
            "target_set": [
                "alliance_score",
                "score_differential",
                *[mapping.target_name for mapping in base_opts.target_map],
                *base_opts.atomic_count_targets,
                *base_opts.foul_targets,
                *base_opts.bonus_binary_targets,
                *base_opts.special_binary_targets,
                "per_team_auto_status",
                "per_team_endgame_status",
                "match_winner",
                *[f"judged_award_{name}" for name in base_opts.award_targets],
            ],
            "auxiliary_tasks": sorted(sidecars),
            "baselines": ["mean", "ridge", "rolling-pridge", "statbotics"],
            "calibration": "week-t-minus-1-platt-logit",
            "resolved_options": base_opts.model_dump(mode="json"),
            "complete": complete,
            "smoke": smoke,
            "promotion_eligible": False,
        },
        source_files=sources,
        promotion_eligible=False,
        promotion_note="One-season, one-seed static reference; evidence is not promotion eligible.",
    )
    return ReferenceStudyResult(
        output,
        manifest_path,
        metrics,
        prediction_table,
        selected_initialization,
        tensorboard_dir,
    )


__all__ = [
    "ARCHITECTURES",
    "REFERENCE_SCHEMA",
    "REFERENCE_WORKFLOW",
    "ReferenceStudyResult",
    "nested_temporal_split",
    "paired_event_bootstrap",
    "reference_calibration",
    "reference_metrics",
    "refit_temporal_split",
    "run_reference_study",
    "validate_reference_rows",
]
