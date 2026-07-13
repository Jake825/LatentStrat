# ruff: noqa: E501
"""2026 physics-consistent static multitask interaction study."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from latentstrat.artifacts import sha256_file, write_artifact_manifest
from latentstrat.config import LatentStratOptions, TargetMapping, default_options
from latentstrat.season.championship_study import (
    _attach_evaluation_context,
    _baseline_predictions,
    _metric_values,
    _per_match_losses,
    compute_metrics,
)
from latentstrat.season.data import Split
from latentstrat.season.evaluate import _predict_batched, sigmoid
from latentstrat.season.features import FeatureTrainingResult, train_feature_table
from latentstrat.season.physics import (
    HUB_PERIODS,
    build_physics_target_table,
    compose_predicted_scores,
    fit_positive_score_logit_scale,
    validate_hard_composer,
)
from latentstrat.season.static_core import (
    CHAMPIONSHIP_EVENTS,
    EINSTEIN_EVENT,
    _hash_strings,
    _write_hparams_same_run,
    build_split_manifest,
    fit_calibrators,
    manifest_keys,
    split_from_keys,
    validate_split_manifest,
)
from latentstrat.season.train import create_tensorboard_writer, match_v5_matrices

WORKFLOW = "season.2026-static-multitask-physics"
ARTIFACT_SCHEMA = "2026-static-multitask-physics-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARMS = ("primary-only", "dense-breakdown", "physics-consistent", "full-structured")
ARCHITECTURES = ("additive", "full-match")
SEEDS = (2026, 2027, 2028)
AWARD_CATEGORIES = ("auto", "design", "control", "excellence")

Arm = Literal["primary-only", "dense-breakdown", "physics-consistent", "full-structured"]
Architecture = Literal["additive", "full-match"]


@dataclass(frozen=True)
class MultitaskStudySpec:
    arm: Arm
    architecture: Architecture
    seed: int
    epochs: int
    train_match_keys: tuple[str, ...]
    evaluation_match_keys: tuple[str, ...]
    score_logit_alpha: float
    score_logit_sigma: float


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    ).hexdigest()


def study_options(spec: MultitaskStudySpec) -> LatentStratOptions:
    dense = spec.arm != "primary-only"
    full = spec.arm == "full-structured"
    return default_options(2026).model_copy(
        update={
            "state_model": "static-z-base",
            "match_architecture": spec.architecture,
            "core_objective": True,
            "study_arm": spec.arm,
            "score_target_mode": "official-total-core",
            "target_map": (
                TargetMapping(target_name="core_red_total", tba_field="redTotalPoints"),
                TargetMapping(target_name="core_blue_total", tba_field="blueTotalPoints"),
            ),
            "continuous_targets": ("core_red_total", "core_blue_total"),
            "binary_targets": ("red_win",),
            "atomic_count_targets": (
                tuple(f"atomic_{period}_count" for period in HUB_PERIODS) if dense else ()
            ),
            "foul_targets": (
                ("committed_minor_foul_count", "committed_major_foul_count") if dense else ()
            ),
            "bonus_binary_targets": (("bonus_energized", "bonus_supercharged") if dense else ()),
            "special_binary_targets": (),
            "auto_class_order": ("None", "Level1"),
            "endgame_class_order": ("None", "Level1", "Level2", "Level3"),
            "award_targets": AWARD_CATEGORIES if full else (),
            "physics_score_logit_alpha": spec.score_logit_alpha,
            "physics_score_logit_sigma": spec.score_logit_sigma,
            "score_loss_weight": 1.0,
            "winner_primary_loss_weight": 0.25,
            "embedding_loss_weight": 1.0,
            "breakdown_loss_weight": 0.5,
            "score_consistency_loss_weight": 0.10,
            "score_ordering_loss_weight": 0.10,
            "winner_consistency_loss_weight": 0.05,
            "competition_probe_loss_weight": 0.10,
            "award_probe_loss_weight": 0.10,
            "l2_embedding": 0.0,
            "epochs": spec.epochs,
            "mini_batch_size": 256,
            "max_grad_norm": 5.0,
            "random_seed": spec.seed,
            "bin_positive_weights": [1.0],
            "use_early_stopping": False,
            "restore_best_validation_model": False,
            "checkpoint_every_epochs": 1,
            "checkpoint_milestone_epochs": tuple(range(1, spec.epochs + 1)),
            "scheduler": "cosine",
            "use_lr_scheduler": True,
            "log_optimizer_steps": False,
        }
    )


def prepare_study_table(feature_table: pd.DataFrame, physics: pd.DataFrame) -> pd.DataFrame:
    """Attach audited labels and apply the study's DQ/tie masks."""

    out = feature_table.merge(
        physics,
        on=["season", "event_key", "match_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_physics"),
    )
    for color in ("red", "blue"):
        out[f"core_{color}_total"] = pd.to_numeric(out[f"{color}_official_score"], errors="coerce")
        for period in HUB_PERIODS:
            out[f"{color}_atomic_{period}_count"] = pd.to_numeric(
                out[f"{color}_hub_{period}_count"], errors="coerce"
            )
        for source in (
            "committed_minor_foul_count",
            "committed_major_foul_count",
            "bonus_energized",
            "bonus_supercharged",
        ):
            physics_column = f"{color}_{source}_physics"
            if physics_column in out:
                out[f"{color}_{source}"] = out[physics_column]
    dq = out["red_has_dq"].astype(bool) | out["blue_has_dq"].astype(bool)
    tie = pd.to_numeric(out["red_total_score"], errors="coerce").eq(
        pd.to_numeric(out["blue_total_score"], errors="coerce")
    )
    loss_columns = [
        "core_red_total",
        "core_blue_total",
        *[f"{color}_atomic_{period}_count" for color in ("red", "blue") for period in HUB_PERIODS],
        *[
            f"{color}_committed_{severity}_foul_count"
            for color in ("red", "blue")
            for severity in ("minor", "major")
        ],
        *[
            f"{color}_bonus_{bonus}"
            for color in ("red", "blue")
            for bonus in ("energized", "supercharged")
        ],
    ]
    out.loc[dq, loss_columns] = np.nan
    out["red_win"] = pd.to_numeric(out["red_win"], errors="coerce").astype(float)
    out.loc[dq | tie, "red_win"] = np.nan
    for column in [name for name in out if "_award_" in name]:
        out[column] = np.nan
    return out


def build_award_candidates(
    features: pd.DataFrame, rankings: pd.DataFrame, eligible_events: set[str]
) -> pd.DataFrame:
    """Build explicit negatives only where a supported robot award was observed."""

    slot_columns = [
        (f"{color}_team_{slot}_key", category, f"{color}_team_{slot}_award_{category}")
        for color in ("red", "blue")
        for slot in (1, 2, 3)
        for category in AWARD_CATEGORIES
    ]
    winners: dict[tuple[str, str], set[str]] = {}
    source = features[features["event_key"].astype(str).isin(eligible_events)]
    for team_column, category, target_column in slot_columns:
        positive = pd.to_numeric(source[target_column], errors="coerce").eq(1)
        for event, team in zip(
            source.loc[positive, "event_key"], source.loc[positive, team_column], strict=True
        ):
            winners.setdefault((str(event), category), set()).add(str(team))
    rows: dict[tuple[str, str], dict[str, object]] = {}
    candidates = rankings[rankings["event_key"].astype(str).isin(eligible_events)]
    for row in candidates.itertuples(index=False):
        key = (str(row.event_key), str(row.team_key))
        record = rows.setdefault(
            key,
            {
                "event_key": key[0],
                "team_key": key[1],
                **{f"award_{c}": np.nan for c in AWARD_CATEGORIES},
            },
        )
        for category in AWARD_CATEGORIES:
            event_winners = winners.get((key[0], category))
            if event_winners:
                record[f"award_{category}"] = float(key[1] in event_winners)
    result = pd.DataFrame(rows.values())
    if result.empty:
        return pd.DataFrame(
            columns=["event_key", "team_key", *[f"award_{c}" for c in AWARD_CATEGORIES]]
        )
    active = result[[f"award_{c}" for c in AWARD_CATEGORIES]].notna().any(axis=1)
    return result[active].sort_values(["event_key", "team_key"]).reset_index(drop=True)


def filter_sidecars(
    rankings: pd.DataFrame,
    playoffs: pd.DataFrame,
    awards: pd.DataFrame,
    table: pd.DataFrame,
    train_keys: tuple[str, ...],
    arm: Arm,
) -> dict[str, pd.DataFrame]:
    if arm != "full-structured":
        return {}
    train_events = set(
        table.loc[table["match_key"].astype(str).isin(train_keys), "event_key"].astype(str)
    )
    return {
        "rankings": rankings[rankings["event_key"].astype(str).isin(train_events)].copy(),
        "playoffs": playoffs[playoffs["event_key"].astype(str).isin(train_events)].copy(),
        "award_candidates": awards[awards["event_key"].astype(str).isin(train_events)].copy(),
    }


def train_multitask_leaf(
    table: pd.DataFrame,
    spec: MultitaskStudySpec,
    *,
    prior_checkpoint: Path,
    sidecars: dict[str, pd.DataFrame],
    output_dir: Path,
    log_dir: Path | None,
    phase: str,
) -> FeatureTrainingResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = study_options(spec)
    if phase == "refit":
        train_mask = table["match_key"].astype(str).isin(spec.train_match_keys).to_numpy(bool)
        split = Split(
            train_mask,
            np.zeros(len(table), dtype=bool),
            ~train_mask,
            "exact-match-key-static-multitask-refit",
        )
    else:
        split = split_from_keys(table, spec.train_match_keys, spec.evaluation_match_keys)
    resume = output_dir / "resume" / "latest.ckpt"
    resume_source = resume if resume.exists() else None
    purge_step = None
    already_complete = False
    if resume_source is not None:
        payload = torch.load(resume_source, map_location="cpu", weights_only=False)
        purge_step = int(payload["completed_epoch"]) + 1
        already_complete = purge_step > spec.epochs
    writer = (
        create_tensorboard_writer(str(log_dir), purge_step=purge_step)
        if log_dir is not None and not already_complete
        else None
    )
    contract = {
        "workflow": WORKFLOW,
        "phase": phase,
        "spec": asdict(spec),
        "options": opts.model_dump(mode="json"),
        "train_match_key_hash": _hash_strings(list(spec.train_match_keys)),
        "evaluation_match_key_hash": _hash_strings(list(spec.evaluation_match_keys)),
        "prior_checkpoint_sha256": sha256_file(prior_checkpoint),
        "sidecars": {name: int(len(frame)) for name, frame in sidecars.items()},
        "known_as_of": "post-event labels restricted to events wholly inside the training role",
        "stop_gradients": False,
    }
    try:
        if writer is not None:
            writer.add_text(
                "Run/Contract", "```json\n" + json.dumps(contract, indent=2) + "\n```", 0
            )
        result = train_feature_table(
            table,
            opts,
            verbose=True,
            prior_checkpoint=None if resume_source else prior_checkpoint,
            sidecar_tables=sidecars,
            split_override=split,
            tensorboard_writer=writer,
            tensorboard_logdir=log_dir,
            resume_checkpoint=resume_source,
            resume_output=resume,
        )
        result.history.to_csv(output_dir / "training_history.csv", index=False)
        checkpoint = {
            "checkpoint_schema_version": 9,
            "study_schema": ARTIFACT_SCHEMA,
            "model_state_dict": result.model.state_dict(),
            "options": opts.model_dump(mode="json"),
            "target_stats": {
                "target_names": list(result.target_stats.target_names),
                "mu": result.target_stats.mu.tolist(),
                "sigma": result.target_stats.sigma.tolist(),
            },
            "v57_target_stats": (
                {
                    "target_names": list(result.v57_target_stats.target_names),
                    "mu": result.v57_target_stats.mu.tolist(),
                    "sigma": result.v57_target_stats.sigma.tolist(),
                }
                if result.v57_target_stats is not None
                else None
            ),
            "team_base_index_map": result.team_index_map,
            "team_event_index_map": result.team_event_index_map,
            "spec": asdict(spec),
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")
        (output_dir / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if writer is not None and not result.history.empty:
            final = result.history.iloc[-1]
            outcomes = {
                "HParams/CompletedEpoch": float(final["epoch"]),
                "HParams/FinalScoreRMSE": float(final.get("validation_score_rmse", math.nan)),
                "HParams/FinalDifferentialRMSE": float(
                    final.get("validation_score_differential_rmse", math.nan)
                ),
                "HParams/FinalBrier": float(final.get("validation_winner_brier", math.nan)),
                "HParams/FinalLogLoss": float(final.get("validation_winner_log_loss", math.nan)),
            }
            _write_hparams_same_run(
                writer,
                {
                    "phase": phase,
                    "arm": spec.arm,
                    "architecture": spec.architecture,
                    "seed": spec.seed,
                    "epochs": spec.epochs,
                },
                {name: value for name, value in outcomes.items() if math.isfinite(value)},
                step=spec.epochs,
            )
        return result
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def load_epoch_checkpoint(result: FeatureTrainingResult, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") or payload.get("model")
    if state is None:
        raise ValueError(f"Milestone checkpoint lacks model state: {checkpoint}")
    result.model.load_state_dict(state)


def prediction_frame(
    result: FeatureTrainingResult,
    spec: MultitaskStudySpec,
    evaluation_keys: tuple[str, ...],
    *,
    phase: str,
) -> pd.DataFrame:
    red, blue, red_event, blue_event, red_missing, blue_missing = match_v5_matrices(result.prepared)
    prediction = _predict_batched(
        result.model,
        red,
        blue,
        result.model.opts,
        red_event=red_event,
        blue_event=blue_event,
        red_missing=red_missing,
        blue_missing=blue_missing,
        return_aux=True,
    )
    direct = prediction["cont"] * result.target_stats.sigma + result.target_stats.mu
    mask = result.prepared["match_key"].astype(str).isin(evaluation_keys).to_numpy()
    source = result.prepared.loc[mask]
    frame = pd.DataFrame(
        {
            "phase": phase,
            "supervision_mode": spec.arm,
            "architecture": spec.architecture,
            "seed": str(spec.seed),
            "epoch": spec.epochs,
            "event_key": source["event_key"].astype(str).to_numpy(),
            "event_week": pd.to_numeric(source["event_week"], errors="coerce").to_numpy(),
            "event_type": (
                pd.to_numeric(source["event_type"], errors="coerce").to_numpy()
                if "event_type" in source
                else np.full(len(source), np.nan)
            ),
            "match_key": source["match_key"].astype(str).to_numpy(),
            "pred_red_total_score": direct[mask, 0],
            "pred_blue_total_score": direct[mask, 1],
            "actual_red_total_score": source["red_total_score"].to_numpy(float),
            "actual_blue_total_score": source["blue_total_score"].to_numpy(float),
            "raw_pred_red_win_probability": sigmoid(prediction["bin"][mask, 0]),
            "score_implied_red_win_probability": sigmoid(
                spec.score_logit_alpha * (direct[mask, 0] - direct[mask, 1])
            ),
            "red_has_dq": source["red_has_dq"].astype(bool).to_numpy(),
            "blue_has_dq": source["blue_has_dq"].astype(bool).to_numpy(),
            "red_has_surrogate": source["red_has_surrogate"].astype(bool).to_numpy(),
            "blue_has_surrogate": source["blue_has_surrogate"].astype(bool).to_numpy(),
        }
    )
    if spec.arm != "primary-only":
        composed = compose_predicted_scores(
            torch.as_tensor(prediction["atomic"], dtype=torch.float32),
            torch.as_tensor(prediction["foul"], dtype=torch.float32),
            torch.as_tensor(prediction["auto"], dtype=torch.float32),
            torch.as_tensor(prediction["endgame"], dtype=torch.float32),
        ).numpy()[mask]
        frame["composed_red_total_score"] = composed[:, 0]
        frame["composed_blue_total_score"] = composed[:, 1]
        frame["direct_composed_score_mae"] = np.mean(np.abs(direct[mask, :2] - composed), axis=1)
    else:
        frame["composed_red_total_score"] = np.nan
        frame["composed_blue_total_score"] = np.nan
        frame["direct_composed_score_mae"] = np.nan
    frame["winner_logit_disagreement"] = np.abs(
        np.log(np.clip(frame["raw_pred_red_win_probability"], 1e-7, 1 - 1e-7))
        - np.log1p(-np.clip(frame["raw_pred_red_win_probability"], 1e-7, 1 - 1e-7))
        - spec.score_logit_alpha * (frame["pred_red_total_score"] - frame["pred_blue_total_score"])
    )
    return frame


def select_common_epochs(
    history: pd.DataFrame, ridge_denominators: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one DCMP epoch per arm/architecture using all three seeds."""

    metric_fields = {
        "score_rmse": "validation_score_rmse",
        "differential_rmse": "validation_score_differential_rmse",
        "brier": "validation_winner_brier",
        "log_loss": "validation_winner_log_loss",
    }
    rows = []
    best_rows = []
    for (arm, architecture), group in history.groupby(["arm", "architecture"]):
        for epoch, epoch_rows in group.groupby("epoch"):
            ratios = {
                name: float(pd.to_numeric(epoch_rows[field], errors="coerce").mean())
                / ridge_denominators[name]
                for name, field in metric_fields.items()
            }
            rows.append(
                {
                    "arm": arm,
                    "architecture": architecture,
                    "epoch": int(epoch),
                    **ratios,
                    "selection_index": (
                        2 * ratios["score_rmse"]
                        + ratios["differential_rmse"]
                        + ratios["brier"]
                        + ratios["log_loss"]
                    )
                    / 5,
                }
            )
        cell = pd.DataFrame(
            [row for row in rows if row["arm"] == arm and row["architecture"] == architecture]
        )
        selected = cell.sort_values(["selection_index", "epoch"]).iloc[0]
        best_rows.append(
            {
                "arm": arm,
                "architecture": architecture,
                "selected_epoch": int(selected["epoch"]),
                "selection_index": float(selected["selection_index"]),
            }
        )
    return pd.DataFrame(best_rows), pd.DataFrame(rows)


def physics_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    neural = predictions[predictions["supervision_mode"].isin(ARMS)]
    for identity, frame in neural.groupby(
        ["supervision_mode", "architecture", "seed", "epoch"], dropna=False
    ):
        clean = frame[~(frame["red_has_dq"].astype(bool) | frame["blue_has_dq"].astype(bool))]
        direct_composed = pd.to_numeric(clean["direct_composed_score_mae"], errors="coerce")
        logit_gap = pd.to_numeric(clean["winner_logit_disagreement"], errors="coerce")
        composed = np.concatenate(
            [
                pd.to_numeric(clean["composed_red_total_score"], errors="coerce").to_numpy(),
                pd.to_numeric(clean["composed_blue_total_score"], errors="coerce").to_numpy(),
            ]
        )
        actual = np.concatenate(
            [
                clean["actual_red_total_score"].to_numpy(float),
                clean["actual_blue_total_score"].to_numpy(float),
            ]
        )
        valid_composed = np.isfinite(composed) & np.isfinite(actual)
        rows.append(
            {
                "arm": identity[0],
                "architecture": identity[1],
                "seed": identity[2],
                "epoch": identity[3],
                "direct_composed_score_mae": float(direct_composed.mean()),
                "winner_logit_disagreement_mae": float(logit_gap.mean()),
                "composed_score_rmse": (
                    float(
                        np.sqrt(np.mean((composed[valid_composed] - actual[valid_composed]) ** 2))
                    )
                    if valid_composed.any()
                    else np.nan
                ),
                "count": int(len(clean)),
            }
        )
    return pd.DataFrame(rows)


def _comparison_definitions() -> list[tuple[str, tuple[str, str], tuple[str, str]]]:
    pairs: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
    for architecture in ARCHITECTURES:
        pairs.extend(
            [
                (
                    f"dense-v-primary:{architecture}",
                    ("primary-only", architecture),
                    ("dense-breakdown", architecture),
                ),
                (
                    f"physics-v-dense:{architecture}",
                    ("dense-breakdown", architecture),
                    ("physics-consistent", architecture),
                ),
                (
                    f"structured-v-physics:{architecture}",
                    ("physics-consistent", architecture),
                    ("full-structured", architecture),
                ),
            ]
        )
    for arm in ARMS:
        pairs.append((f"full-v-additive:{arm}", (arm, "additive"), (arm, "full-match")))
    return pairs


def hierarchical_paired_comparisons(
    predictions: pd.DataFrame, *, resamples: int = 5_000, seed: int = 2026
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = predictions[
        predictions["supervision_mode"].isin(ARMS)
        & ~(predictions["red_has_dq"].astype(bool) | predictions["blue_has_dq"].astype(bool))
    ]
    rng = np.random.default_rng(seed)
    summary: list[dict[str, object]] = []
    samples_out: list[dict[str, object]] = []
    loo_out: list[dict[str, object]] = []
    for name, left_key, right_key in _comparison_definitions():
        joined_seeds = []
        for seed_value in SEEDS:
            left = clean[
                clean["supervision_mode"].eq(left_key[0])
                & clean["architecture"].eq(left_key[1])
                & clean["seed"].astype(str).eq(str(seed_value))
            ]
            right = clean[
                clean["supervision_mode"].eq(right_key[0])
                & clean["architecture"].eq(right_key[1])
                & clean["seed"].astype(str).eq(str(seed_value))
            ]
            joined = _per_match_losses(left).merge(
                _per_match_losses(right),
                on=["event_key", "match_key"],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            joined["seed"] = seed_value
            joined_seeds.append(joined)
        joined = pd.concat(joined_seeds, ignore_index=True)
        events = sorted(joined["event_key"].unique())
        for metric in (
            "score_squared_error",
            "differential_squared_error",
            "winner_brier",
            "winner_log_loss",
        ):
            left_values = joined[f"{metric}_left"]
            right_values = joined[f"{metric}_right"]
            left_mean = float(np.nanmean(left_values))
            right_mean = float(np.nanmean(right_values))
            left_effect = math.sqrt(left_mean) if metric.endswith("squared_error") else left_mean
            right_effect = math.sqrt(right_mean) if metric.endswith("squared_error") else right_mean
            bootstrap = []
            for sample_index in range(resamples):
                selected_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
                selected_events = rng.choice(events, size=len(events), replace=True)
                chunks = [
                    joined[joined["seed"].eq(int(seed_choice)) & joined["event_key"].eq(event)]
                    for seed_choice in selected_seeds
                    for event in selected_events
                ]
                sampled = pd.concat(chunks, ignore_index=True)
                delta = float(np.nanmean(sampled[f"{metric}_right"] - sampled[f"{metric}_left"]))
                bootstrap.append(delta)
                samples_out.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "resample": sample_index,
                        "delta": delta,
                    }
                )
            values = np.asarray(bootstrap)
            seed_directions = []
            for seed_value in SEEDS:
                subset = joined[joined["seed"].eq(seed_value)]
                seed_directions.append(
                    float(np.nanmean(subset[f"{metric}_right"] - subset[f"{metric}_left"]))
                )
            summary.append(
                {
                    "comparison": name,
                    "left": f"{left_key[0]}:{left_key[1]}",
                    "right": f"{right_key[0]}:{right_key[1]}",
                    "metric": metric,
                    "left_value": left_effect,
                    "right_value": right_effect,
                    "relative_delta": (right_effect - left_effect) / left_effect,
                    "delta_right_minus_left": float(np.nanmean(right_values - left_values)),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "probability_improvement": float(np.mean(values < 0)),
                    "seeds_improved": int(sum(value < 0 for value in seed_directions)),
                }
            )
            for omitted in [
                *(f"seed:{value}" for value in SEEDS),
                *(f"event:{value}" for value in events),
            ]:
                kind, value = omitted.split(":", 1)
                subset = (
                    joined[~joined["seed"].eq(int(value))]
                    if kind == "seed"
                    else joined[~joined["event_key"].eq(value)]
                )
                left_loo = float(np.nanmean(subset[f"{metric}_left"]))
                right_loo = float(np.nanmean(subset[f"{metric}_right"]))
                if metric.endswith("squared_error"):
                    left_loo, right_loo = math.sqrt(left_loo), math.sqrt(right_loo)
                loo_out.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "omitted": omitted,
                        "left_value": left_loo,
                        "right_value": right_loo,
                        "relative_delta": (right_loo - left_loo) / left_loo,
                        "delta_right_minus_left": float(
                            np.nanmean(subset[f"{metric}_right"] - subset[f"{metric}_left"])
                        ),
                    }
                )
    rescue_summary, rescue_samples, rescue_loo = _interaction_rescue_comparisons(
        clean, resamples=resamples, seed=seed + 1
    )
    return (
        pd.concat([pd.DataFrame(summary), rescue_summary], ignore_index=True),
        pd.concat([pd.DataFrame(samples_out), rescue_samples], ignore_index=True),
        pd.concat([pd.DataFrame(loo_out), rescue_loo], ignore_index=True),
    )


def _interaction_rescue_comparisons(
    predictions: pd.DataFrame, *, resamples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure whether an objective changes Full-match's effect relative to Additive."""

    rng = np.random.default_rng(seed)
    summaries: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    leave_one_out: list[dict[str, object]] = []
    for target_arm in ("physics-consistent", "full-structured"):
        name = f"interaction-rescue:{target_arm}-v-primary"
        seed_frames = []
        for seed_value in SEEDS:
            parts = {}
            for arm in ("primary-only", target_arm):
                for architecture in ARCHITECTURES:
                    frame = predictions[
                        predictions["supervision_mode"].eq(arm)
                        & predictions["architecture"].eq(architecture)
                        & predictions["seed"].astype(str).eq(str(seed_value))
                    ]
                    parts[(arm, architecture)] = _per_match_losses(frame)
            joined = parts[("primary-only", "additive")]
            joined = joined.rename(
                columns={
                    column: f"{column}_primary_additive"
                    for column in joined.columns
                    if column not in {"event_key", "match_key"}
                }
            )
            for arm, architecture in (
                ("primary-only", "full-match"),
                (target_arm, "additive"),
                (target_arm, "full-match"),
            ):
                suffix = f"{arm.replace('-', '_')}_{architecture.replace('-', '_')}"
                part = parts[(arm, architecture)].rename(
                    columns={
                        column: f"{column}_{suffix}"
                        for column in parts[(arm, architecture)].columns
                        if column not in {"event_key", "match_key"}
                    }
                )
                joined = joined.merge(part, on=["event_key", "match_key"], validate="one_to_one")
            joined["seed"] = seed_value
            seed_frames.append(joined)
        paired = pd.concat(seed_frames, ignore_index=True)
        events = sorted(paired["event_key"].unique())

        def effect(frame: pd.DataFrame, metric: str, arm: str) -> tuple[float, float, float]:
            prefix = arm.replace("-", "_")
            additive = float(np.nanmean(frame[f"{metric}_{prefix}_additive"]))
            full = float(np.nanmean(frame[f"{metric}_{prefix}_full_match"]))
            if metric.endswith("squared_error"):
                additive, full = math.sqrt(additive), math.sqrt(full)
            return additive, full, full - additive

        for metric in (
            "score_squared_error",
            "differential_squared_error",
            "winner_brier",
            "winner_log_loss",
        ):
            primary_additive, primary_full, primary_effect = effect(paired, metric, "primary-only")
            target_additive, target_full, target_effect = effect(paired, metric, target_arm)
            scale = target_additive if metric.endswith("squared_error") else 1.0
            observed_delta = target_effect - primary_effect
            bootstrap_values = []
            for sample_index in range(resamples):
                selected_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
                selected_events = rng.choice(events, size=len(events), replace=True)
                sampled = pd.concat(
                    [
                        paired[
                            paired["seed"].eq(int(seed_choice)) & paired["event_key"].eq(event_key)
                        ]
                        for seed_choice in selected_seeds
                        for event_key in selected_events
                    ],
                    ignore_index=True,
                )
                sampled_primary = effect(sampled, metric, "primary-only")[2]
                sampled_target = effect(sampled, metric, target_arm)[2]
                delta = sampled_target - sampled_primary
                bootstrap_values.append(delta)
                samples.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "resample": sample_index,
                        "delta": delta,
                    }
                )
            values = np.asarray(bootstrap_values)
            seed_directions = []
            for seed_value in SEEDS:
                subset = paired[paired["seed"].eq(seed_value)]
                seed_directions.append(
                    effect(subset, metric, target_arm)[2]
                    - effect(subset, metric, "primary-only")[2]
                )
            summaries.append(
                {
                    "comparison": name,
                    "left": "full-minus-additive:primary-only",
                    "right": f"full-minus-additive:{target_arm}",
                    "metric": metric,
                    "left_value": primary_effect,
                    "right_value": target_effect,
                    "relative_delta": observed_delta / scale,
                    "delta_right_minus_left": observed_delta,
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "probability_improvement": float(np.mean(values < 0)),
                    "seeds_improved": int(sum(value < 0 for value in seed_directions)),
                    "primary_additive_value": primary_additive,
                    "primary_full_value": primary_full,
                    "target_additive_value": target_additive,
                    "target_full_value": target_full,
                }
            )
            for omitted in [
                *(f"seed:{value}" for value in SEEDS),
                *(f"event:{value}" for value in events),
            ]:
                kind, value = omitted.split(":", 1)
                subset = (
                    paired[~paired["seed"].eq(int(value))]
                    if kind == "seed"
                    else paired[~paired["event_key"].eq(value)]
                )
                left_value = effect(subset, metric, "primary-only")[2]
                right_value = effect(subset, metric, target_arm)[2]
                subset_scale = effect(subset, metric, target_arm)[0]
                leave_one_out.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "omitted": omitted,
                        "left_value": left_value,
                        "right_value": right_value,
                        "relative_delta": (right_value - left_value)
                        / (subset_scale if metric.endswith("squared_error") else 1.0),
                        "delta_right_minus_left": right_value - left_value,
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(samples), pd.DataFrame(leave_one_out)


def classify_comparisons(comparisons: pd.DataFrame, leave_one_out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison, frame in comparisons.groupby("comparison"):
        values = frame.set_index("metric")
        score = float(values.loc["score_squared_error", "relative_delta"])
        differential = float(values.loc["differential_squared_error", "relative_delta"])
        brier = float(values.loc["winner_brier", "delta_right_minus_left"])
        log_loss = float(values.loc["winner_log_loss", "delta_right_minus_left"])
        probability = max(
            float(values.loc["differential_squared_error", "probability_improvement"]),
            min(
                float(values.loc["winner_brier", "probability_improvement"]),
                float(values.loc["winner_log_loss", "probability_improvement"]),
            ),
        )
        seeds_improved = max(
            int(values.loc["differential_squared_error", "seeds_improved"]),
            min(
                int(values.loc["winner_brier", "seeds_improved"]),
                int(values.loc["winner_log_loss", "seeds_improved"]),
            ),
        )
        improved = (
            (differential <= -0.02 and brier <= 0.005 and log_loss <= 0.01)
            or (brier <= -0.005 and log_loss <= -0.01 and differential <= 0.02)
        ) and score <= 0.02
        harmed = (
            (differential >= 0.02 and brier >= -0.005 and log_loss >= -0.01)
            or (brier >= 0.005 and log_loss >= 0.01 and differential >= -0.02)
        ) and score >= -0.02
        loo = leave_one_out[leave_one_out["comparison"].eq(comparison)]
        support_reversal = (
            (loo[loo["metric"].eq("score_squared_error")]["relative_delta"] > 0.02).any()
            or (loo[loo["metric"].eq("differential_squared_error")]["relative_delta"] > 0.02).any()
            or (loo[loo["metric"].eq("winner_brier")]["delta_right_minus_left"] > 0.005).any()
            or (loo[loo["metric"].eq("winner_log_loss")]["delta_right_minus_left"] > 0.01).any()
        )
        harm_reversal = (
            (loo[loo["metric"].eq("score_squared_error")]["relative_delta"] < -0.02).any()
            or (loo[loo["metric"].eq("differential_squared_error")]["relative_delta"] < -0.02).any()
            or (loo[loo["metric"].eq("winner_brier")]["delta_right_minus_left"] < -0.005).any()
            or (loo[loo["metric"].eq("winner_log_loss")]["delta_right_minus_left"] < -0.01).any()
        )
        if improved and probability >= 0.80 and seeds_improved >= 2 and not support_reversal:
            classification = "diagnostic-supported"
        elif harmed and probability <= 0.20 and seeds_improved <= 1 and not harm_reversal:
            classification = "harmful"
        else:
            classification = "uncertain"
        rows.append(
            {
                "comparison": comparison,
                "classification": classification,
                "score_rmse_relative_delta": score,
                "differential_rmse_relative_delta": differential,
                "brier_delta": brier,
                "log_loss_delta": log_loss,
                "probability_improvement": probability,
                "seeds_improved": seeds_improved,
                "leave_one_out_stable": not (
                    support_reversal if classification == "diagnostic-supported" else harm_reversal
                ),
            }
        )
    return pd.DataFrame(rows)


def _metric_lookup(metrics: pd.DataFrame, arm: str, architecture: str, metric: str) -> float:
    rows = metrics[
        metrics["supervision_mode"].eq(arm)
        & metrics["architecture"].eq(architecture)
        & metrics["scope"].eq("primary-clean")
        & metrics["probability_variant"].eq("raw_pred_red_win_probability")
        & metrics["metric"].eq(metric)
    ]
    return (
        float(pd.to_numeric(rows["value"], errors="coerce").mean()) if not rows.empty else math.nan
    )


def render_report(
    *,
    metrics: pd.DataFrame,
    selected_epochs: pd.DataFrame,
    classifications: pd.DataFrame,
    diagnostics: pd.DataFrame,
    coverage: dict[str, object],
    history: pd.DataFrame,
) -> str:
    complete = bool(coverage.get("complete"))
    lines = [
        "# 2026 Physics-Consistent Multitask Interaction Study",
        "",
        "> Does dense breakdown supervision, particularly when coupled through an exact "
        "differentiable 2026 score composer, improve the out-of-time Championship "
        "generalization of Full-match relative to Additive without harming official-score "
        "accuracy, relative-strength estimation, or winner probabilities?",
        "",
        "## Direct answer",
        "",
    ]
    if not complete or classifications.empty:
        lines.append(
            "The formal comparison matrix is unavailable, so the declared question cannot be "
            "answered. No architecture-wide conclusion is reported."
        )
    else:
        class_map = dict(
            zip(classifications["comparison"], classifications["classification"], strict=True)
        )
        lines.extend(
            [
                f"- Dense supervision versus primary-only: Additive is **{class_map.get('dense-v-primary:additive', 'uncertain')}**; Full-match is **{class_map.get('dense-v-primary:full-match', 'uncertain')}**.",
                f"- Physics consistency versus independent dense heads: Additive is **{class_map.get('physics-v-dense:additive', 'uncertain')}**; Full-match is **{class_map.get('physics-v-dense:full-match', 'uncertain')}**.",
                f"- Full structured probes versus physics-only multitask: Additive is **{class_map.get('structured-v-physics:additive', 'uncertain')}**; Full-match is **{class_map.get('structured-v-physics:full-match', 'uncertain')}**.",
                f"- Full-match versus Additive under physics consistency: **{class_map.get('full-v-additive:physics-consistent', 'uncertain')}**.",
                f"- Physics-consistent supervision changed Full-match's effect relative to Additive: **{class_map.get('interaction-rescue:physics-consistent-v-primary', 'uncertain')}**.",
                f"- Full structured supervision changed Full-match's effect relative to Additive: **{class_map.get('interaction-rescue:full-structured-v-primary', 'uncertain')}**.",
                "",
                "These are reused-holdout, three-seed diagnostic classifications, not promotion evidence.",
            ]
        )
    lines.extend(["", "## Evidence provenance", ""])
    lines.extend(
        [
            f"- Development trajectories completed: {coverage.get('development_leaves_completed', 0)}/24.",
            f"- Final refits completed: {coverage.get('refit_leaves_completed', 0)}/24.",
            f"- Championship rows: {coverage.get('championship_rows', 0)}; primary-clean: {coverage.get('championship_clean_rows', 0)}.",
            f"- Physics composer audit: {coverage.get('physics_alliance_rows', 0)} alliance rows with maximum error {coverage.get('physics_max_error', 'unknown')}.",
            "- Championship divisions are evaluated once after the final contract is frozen; Einstein is excluded.",
            "- The study uses one static 16-dimensional `Z_base`; it does not test temporal state.",
        ]
    )
    lines.extend(
        [
            "",
            "## Selected development durations",
            "",
            "| Arm | Architecture | Epoch |",
            "|---|---|---:|",
        ]
    )
    for row in selected_epochs.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.architecture} | {int(row.selected_epoch)} |")
    lines.extend(
        [
            "",
            "Each duration was selected from the sealed DCMP block using the predeclared "
            "seed-mean score/differential/Brier/log-loss index. Every development trajectory "
            + (
                "ran through the two-epoch smoke horizon."
                if coverage.get("smoke")
                else "still ran through epoch 100."
            ),
            "",
            "## Championship scoreboard",
            "",
            "| Arm | Architecture | Score RMSE | Differential RMSE | Brier | Log loss | Mean bias |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for architecture in ARCHITECTURES:
            values = [
                _metric_lookup(metrics, arm, architecture, name)
                for name in (
                    "alliance_score_rmse",
                    "score_differential_rmse",
                    "winner_brier",
                    "winner_log_loss",
                    "mean_bias",
                )
            ]
            lines.append(
                f"| {arm} | {architecture} | "
                + " | ".join(f"{value:.4f}" for value in values)
                + " |"
            )
    lines.extend(["", "## Per-run development behavior", ""])
    for (arm, architecture, seed), frame in history[history["phase"].eq("development")].groupby(
        ["arm", "architecture", "seed"]
    ):
        best_score = frame.loc[
            pd.to_numeric(frame["validation_score_rmse"], errors="coerce").idxmin()
        ]
        best_brier = frame.loc[
            pd.to_numeric(frame["validation_winner_brier"], errors="coerce").idxmin()
        ]
        final = frame.sort_values("epoch").iloc[-1]
        lines.extend(
            [
                f"### {arm} / {architecture} / seed {seed}",
                "",
                f"Score RMSE was best at epoch {int(best_score['epoch'])} ({float(best_score['validation_score_rmse']):.4f}) and was {float(final['validation_score_rmse']):.4f} at epoch {int(final['epoch'])}. "
                f"Brier was best at epoch {int(best_brier['epoch'])} ({float(best_brier['validation_winner_brier']):.4f}) and was {float(final['validation_winner_brier']):.4f} at epoch {int(final['epoch'])}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Physics diagnostics",
            "",
            "Composed-score RMSE is diagnostic only. It was never optimized directly against official score; atomic labels and the direct/composed consistency term are the training authorities.",
            "",
        ]
    )
    if not diagnostics.empty:
        for row in diagnostics.itertuples(index=False):
            lines.append(
                f"- {row.arm}/{row.architecture}/seed {row.seed}: composed score RMSE "
                f"{row.composed_score_rmse:.3f}, direct/composed MAE "
                f"{row.direct_composed_score_mae:.3f}, winner-logit disagreement "
                f"{row.winner_logit_disagreement_mae:.3f}."
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Lower training loss or lower head disagreement alone is not evidence that overfitting was rescued.",
            "- Ranking, playoff, and award outcomes are split-local auxiliary labels, never match inputs.",
            "- Arm D combines competition and award probes, so their individual effects are not identifiable here.",
            "- A persistent failure across all arms motivates a temporal/event-state experiment but does not prove temporal state is the cause or solution.",
            "",
            "## Next experiment",
            "",
            "Retain only objective blocks classified as diagnostic-supported or clearly no-harm. If Full-match remains harmful or uncertain after dense and physics-consistent supervision, hold its architecture fixed and compare static `Z_base` against a minimal leakage-safe event-state update using a new out-of-time season boundary.",
            "",
        ]
    )
    return "\n".join(lines)


def run_multitask_study(
    *,
    features_path: str | Path = "data/features/season/features_v58_2026.parquet",
    event_metadata_path: str | Path = "data/features/season/events_2026.parquet",
    corpus_path: str | Path = "data/pretraining/match-breakdown/corpus.sqlite",
    rankings_path: str | Path = "data/sidecars/v58_2026/rankings_2026.parquet",
    playoffs_path: str | Path = "data/sidecars/v58_2026/playoffs_2026.parquet",
    prior_checkpoint: str | Path = "artifacts/baselines/v5.8/prior_checkpoint.pt",
    output_dir: str | Path = "artifacts/reference/2026-static-multitask-physics",
    tensorboard_root: str | Path | None = "runs/2026-static-multitask-physics",
    max_wall_minutes: float = 720.0,
    bootstrap_resamples: int = 5_000,
    smoke: bool = False,
    resume: bool = False,
    prepare_only: bool = False,
) -> Path:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": Path(features_path),
        "events": Path(event_metadata_path),
        "corpus": Path(corpus_path),
        "rankings": Path(rankings_path),
        "playoffs": Path(playoffs_path),
        "prior": Path(prior_checkpoint),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {name} input does not exist: {path}")
    launch_git_clean = _git_clean()
    if not smoke and not launch_git_clean:
        raise RuntimeError("Formal training requires a clean committed Git working tree.")
    finalization_inputs = (
        output / "coverage.json",
        output / "metrics.csv",
        output / "selected_epochs.csv",
        output / "decision_classifications.csv",
        output / "physics_diagnostics.csv",
        output / "development_history.csv",
        output / "frozen_final_contract.json",
    )
    if (
        resume
        and all(path.exists() for path in finalization_inputs)
        and not (output / "report.md").exists()
    ):
        coverage = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(output / "metrics.csv")
        selected_epochs = pd.read_csv(output / "selected_epochs.csv")
        try:
            classifications = pd.read_csv(output / "decision_classifications.csv")
        except pd.errors.EmptyDataError:
            classifications = pd.DataFrame()
        diagnostics = pd.read_csv(output / "physics_diagnostics.csv")
        history = pd.read_csv(output / "development_history.csv")
        frozen = json.loads((output / "frozen_final_contract.json").read_text(encoding="utf-8"))
        report = render_report(
            metrics=metrics,
            selected_epochs=selected_epochs,
            classifications=classifications,
            diagnostics=diagnostics,
            coverage=coverage,
            history=history,
        )
        report_bytes = report.encode("utf-8")
        (output / "report.md").write_bytes(report_bytes)
        docs_report = REPOSITORY_ROOT / "docs/2026-static-multitask-physics-report.md"
        if not bool(coverage.get("smoke")):
            docs_report.write_bytes(report_bytes)
        documentation_hash = sha256_file(docs_report) if docs_report.exists() else None
        (output / "report_hashes.json").write_text(
            json.dumps(
                {
                    "artifact_sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "documentation_sha256": documentation_hash,
                    "matching": documentation_hash == hashlib.sha256(report_bytes).hexdigest(),
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
            source_files={name: path for name, path in paths.items()},
            promotion_eligible=False,
            promotion_note="Three-seed reused-Championship static diagnostic.",
        )
        return output
    existing_resume = list(output.glob("development/**/resume/latest.ckpt")) + list(
        output.glob("refit/**/resume/latest.ckpt")
    )
    if existing_resume and not resume:
        raise RuntimeError("Existing study checkpoints require explicit --resume.")

    feature_table = pd.read_parquet(paths["features"])
    events = pd.read_parquet(paths["events"])
    rankings = pd.read_parquet(paths["rankings"])
    playoffs = pd.read_parquet(paths["playoffs"])
    physics = build_physics_target_table(paths["corpus"])
    physics_audit = validate_hard_composer(physics)
    physics.to_parquet(output / "physics_targets.parquet", index=False)
    split_manifest = build_split_manifest(feature_table, events)
    validate_split_manifest(split_manifest)
    split_manifest.to_parquet(output / "split_manifest.parquet", index=False)
    split_manifest.to_csv(output / "split_manifest.csv", index=False)
    modeled_keys = set(
        split_manifest.loc[
            split_manifest["split_role"].isin(
                ["development_train", "development_holdout", "final_train", "championship_test"]
            ),
            "match_key",
        ].astype(str)
    )
    missing_physics = modeled_keys - set(physics["match_key"].astype(str))
    if missing_physics:
        raise ValueError(
            f"Modeled split rows lack cached physics targets: {sorted(missing_physics)[:5]}"
        )
    table = prepare_study_table(feature_table, physics)
    dev_train_keys = manifest_keys(split_manifest, "development_train")
    dev_eval_keys = manifest_keys(split_manifest, "development_holdout")
    final_train_keys = manifest_keys(split_manifest, "final_train")
    test_keys = manifest_keys(split_manifest, "championship_test")
    dev_table = table[table["match_key"].astype(str).isin((*dev_train_keys, *dev_eval_keys))].copy()
    final_table = table[table["match_key"].astype(str).isin((*final_train_keys, *test_keys))].copy()
    clean_dev = dev_table[
        dev_table["match_key"].astype(str).isin(dev_train_keys)
        & ~(dev_table["red_has_dq"].astype(bool) | dev_table["blue_has_dq"].astype(bool))
    ]
    score_logit = fit_positive_score_logit_scale(clean_dev)
    awards = build_award_candidates(feature_table, rankings, set(table["event_key"].astype(str)))
    awards.to_parquet(output / "award_candidates.parquet", index=False)

    identity = {
        "workflow": WORKFLOW,
        "features_sha256": sha256_file(paths["features"]),
        "physics_targets_sha256": sha256_file(output / "physics_targets.parquet"),
        "split_sha256": sha256_file(output / "split_manifest.parquet"),
        "git_commit": _git_commit(),
        "smoke": smoke,
    }
    study_id_file = output / "tensorboard_study_id.txt"
    if study_id_file.exists():
        study_id = study_id_file.read_text(encoding="utf-8").strip()
    else:
        config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
        study_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + f"-{config_hash}"
        study_id_file.write_text(study_id + "\n", encoding="utf-8")
    tensorboard_dir = Path(tensorboard_root) / study_id if tensorboard_root else None
    run_seeds = (2026,) if smoke else SEEDS
    development_epochs = 2 if smoke else 100

    pretraining_contract = {
        "workflow": WORKFLOW,
        "contract_state": "prepared-before-training",
        "report_question": (
            "Does dense breakdown supervision, particularly when coupled through an exact "
            "differentiable 2026 score composer, improve the out-of-time Championship "
            "generalization of Full-match relative to Additive without harming official-score "
            "accuracy, relative-strength estimation, or winner probabilities?"
        ),
        "arms": list(ARMS),
        "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS),
        "development_epochs": 100,
        "physics_audit": physics_audit,
        "score_logit": score_logit,
        "split_counts": split_manifest.groupby("split_role")["match_key"].count().to_dict(),
        "match_key_hashes": {
            role: _hash_strings(list(manifest_keys(split_manifest, role)))
            for role in (
                "development_train",
                "development_holdout",
                "final_train",
                "championship_test",
                "excluded_einstein",
                "excluded_invalid",
            )
        },
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "physics_targets_sha256": sha256_file(output / "physics_targets.parquet"),
        "award_candidates_sha256": sha256_file(output / "award_candidates.parquet"),
        "split_manifest_sha256": sha256_file(output / "split_manifest.parquet"),
        "git_commit": _git_commit(),
        "git_clean_at_launch": launch_git_clean,
        "tensorboard_study_id": study_id,
        "promotion_eligible": False,
    }
    (output / "pretraining_contract.json").write_text(
        json.dumps(pretraining_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "pretraining_manifest.json").write_text(
        json.dumps(
            {
                "contract_sha256": sha256_file(output / "pretraining_contract.json"),
                "physics_targets_sha256": sha256_file(output / "physics_targets.parquet"),
                "split_manifest_sha256": sha256_file(output / "split_manifest.parquet"),
                "award_candidates_sha256": sha256_file(output / "award_candidates.parquet"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if prepare_only:
        return output

    timing_path = output / "timing_pilot.json"
    if not smoke and not timing_path.exists():
        pilot_rows = []
        for arm in ARMS:
            for architecture in ARCHITECTURES:
                spec = MultitaskStudySpec(
                    arm,
                    architecture,
                    2026,
                    2,
                    dev_train_keys,
                    dev_eval_keys,
                    float(score_logit["alpha"]),
                    float(score_logit["sigma_logit"]),
                )
                leaf = output / "timing-pilot" / arm / architecture
                leaf_started = time.perf_counter()
                result = train_multitask_leaf(
                    dev_table,
                    spec,
                    prior_checkpoint=paths["prior"],
                    sidecars=filter_sidecars(
                        rankings, playoffs, awards, table, dev_train_keys, arm
                    ),
                    output_dir=leaf,
                    log_dir=None,
                    phase="development",
                )
                seconds = pd.to_numeric(result.history["epoch_seconds"], errors="coerce").dropna()
                total_leaf_seconds = time.perf_counter() - leaf_started
                mean_epoch_seconds = float(seconds.mean())
                pilot_rows.append(
                    {
                        "arm": arm,
                        "architecture": architecture,
                        "mean_epoch_seconds": mean_epoch_seconds,
                        "total_leaf_seconds": total_leaf_seconds,
                        "fixed_leaf_seconds": max(
                            total_leaf_seconds - mean_epoch_seconds * len(seconds), 0.0
                        ),
                    }
                )
        row_ratio = len(final_train_keys) / len(dev_train_keys)
        projected = sum(
            len(SEEDS)
            * (
                row["fixed_leaf_seconds"]
                + row["mean_epoch_seconds"] * 100
                + row["fixed_leaf_seconds"]
                + row["mean_epoch_seconds"] * row_ratio * 100
            )
            for row in pilot_rows
        )
        timing = {
            "leaves": pilot_rows,
            "projected_seconds": projected,
            "projected_seconds_with_margin": projected * 1.2,
            "budget_seconds": max_wall_minutes * 60,
            "passes": projected * 1.2 <= max_wall_minutes * 60,
        }
        timing_path.write_text(
            json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not timing["passes"]:
            raise RuntimeError(
                f"Timing pilot projects {projected * 1.2 / 60:.1f} minutes with margin, "
                f"exceeding the {max_wall_minutes:.1f}-minute budget."
            )

    clean_dev_train = table[table["match_key"].astype(str).isin(dev_train_keys)]
    clean_dev_holdout = table[table["match_key"].astype(str).isin(dev_eval_keys)]
    development_baselines, _ = _baseline_predictions(
        clean_dev_train, clean_dev_holdout, clean_dev_train, clean_dev_holdout
    )
    ridge = development_baselines[development_baselines["architecture"].eq("direct-total-ridge")]
    ridge_metrics = _metric_values(ridge, "raw_pred_red_win_probability")
    denominators = {
        "score_rmse": ridge_metrics["alliance_score_rmse"][0],
        "differential_rmse": ridge_metrics["score_differential_rmse"][0],
        "brier": ridge_metrics["winner_brier"][0],
        "log_loss": ridge_metrics["winner_log_loss"][0],
    }

    development_results: dict[tuple[str, str, int], FeatureTrainingResult] = {}
    histories = []
    failed: list[dict[str, object]] = []
    for arm in ARMS:
        for architecture in ARCHITECTURES:
            for seed_value in run_seeds:
                spec = MultitaskStudySpec(
                    arm,
                    architecture,
                    seed_value,
                    development_epochs,
                    dev_train_keys,
                    dev_eval_keys,
                    float(score_logit["alpha"]),
                    float(score_logit["sigma_logit"]),
                )
                leaf = output / "development" / arm / architecture / f"seed_{seed_value}"
                try:
                    result = train_multitask_leaf(
                        dev_table,
                        spec,
                        prior_checkpoint=paths["prior"],
                        sidecars=filter_sidecars(
                            rankings, playoffs, awards, table, dev_train_keys, arm
                        ),
                        output_dir=leaf,
                        log_dir=(
                            tensorboard_dir
                            / "development"
                            / arm
                            / architecture
                            / f"seed_{seed_value}"
                            if tensorboard_dir
                            else None
                        ),
                        phase="development",
                    )
                except (FloatingPointError, RuntimeError) as exc:
                    failed.append(
                        {
                            "phase": "development",
                            "arm": arm,
                            "architecture": architecture,
                            "seed": seed_value,
                            "error": str(exc),
                        }
                    )
                    continue
                development_results[(arm, architecture, seed_value)] = result
                history = result.history.copy()
                history.insert(0, "seed", seed_value)
                history.insert(0, "architecture", architecture)
                history.insert(0, "arm", arm)
                history.insert(0, "phase", "development")
                histories.append(history)
    development_history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    expected_development = len(ARMS) * len(ARCHITECTURES) * len(run_seeds)
    if len(development_results) != expected_development:
        raise RuntimeError("Development matrix is incomplete; final refits are suppressed.")
    selected_epochs, selection_curve = select_common_epochs(development_history, denominators)
    selected_epochs.to_csv(output / "selected_epochs.csv", index=False)
    selection_curve.to_csv(output / "selection_curve.csv", index=False)

    development_predictions = []
    calibrator_rows = []
    for (arm, architecture, seed_value), result in development_results.items():
        selected_epoch = int(
            selected_epochs[
                selected_epochs["arm"].eq(arm) & selected_epochs["architecture"].eq(architecture)
            ]["selected_epoch"].iloc[0]
        )
        milestone = (
            output
            / "development"
            / arm
            / architecture
            / f"seed_{seed_value}"
            / "resume"
            / "milestones"
            / f"epoch_{selected_epoch:03d}.ckpt"
        )
        load_epoch_checkpoint(result, milestone)
        selected_spec = MultitaskStudySpec(
            arm,
            architecture,
            seed_value,
            selected_epoch,
            dev_train_keys,
            dev_eval_keys,
            float(score_logit["alpha"]),
            float(score_logit["sigma_logit"]),
        )
        predictions = prediction_frame(result, selected_spec, dev_eval_keys, phase="development")
        development_predictions.append(predictions)
        checkpoint_hash = sha256_file(milestone)
        calibrator = fit_calibrators(predictions, checkpoint_hash)
        calibrator["seed"] = str(seed_value)
        calibrator_rows.append(calibrator)
    development_prediction_table = pd.concat(development_predictions, ignore_index=True)
    development_prediction_table.to_parquet(output / "development_predictions.parquet", index=False)
    calibrators = pd.concat(calibrator_rows, ignore_index=True)
    calibrators.to_json(output / "calibrators.json", orient="records", indent=2)

    frozen = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "study_id": study_id,
        "season": 2026,
        "state_model": "static-z-base",
        "latent_dim": 16,
        "arms": list(ARMS),
        "architectures": list(ARCHITECTURES),
        "seeds": list(run_seeds),
        "development_max_epochs": development_epochs,
        "selected_epochs": selected_epochs.to_dict(orient="records"),
        "loss_weights": {
            "direct_score": 1.0,
            "breakdown_block": 0.5,
            "winner": 0.25,
            "score_consistency": 0.10,
            "score_ordering": 0.10,
            "winner_consistency": 0.05,
            "competition_probes": 0.10,
            "award_probes": 0.10,
            "embedding": 1.0,
        },
        "score_logit": score_logit,
        "match_key_hashes": {
            role: _hash_strings(list(manifest_keys(split_manifest, role)))
            for role in (
                "development_train",
                "development_holdout",
                "final_train",
                "championship_test",
            )
        },
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "physics_targets_sha256": sha256_file(output / "physics_targets.parquet"),
        "award_candidates_sha256": sha256_file(output / "award_candidates.parquet"),
        "split_manifest_sha256": sha256_file(output / "split_manifest.parquet"),
        "development_predictions_sha256": sha256_file(output / "development_predictions.parquet"),
        "calibrators_sha256": sha256_file(output / "calibrators.json"),
        "git_commit": _git_commit(),
        "git_clean_at_launch": launch_git_clean,
        "known_as_of": "all auxiliary labels are restricted to completed training events",
        "championship_event_keys": list(CHAMPIONSHIP_EVENTS),
        "einstein_event_key": EINSTEIN_EVENT,
        "promotion_eligible": False,
        "evidence_role": "reused-three-seed-diagnostic-holdout",
    }
    frozen_path = output / "frozen_final_contract.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    frozen_hash = sha256_file(frozen_path)

    final_results = []
    final_predictions = []
    final_histories = []
    for arm in ARMS:
        for architecture in ARCHITECTURES:
            selected_epoch = int(
                selected_epochs[
                    selected_epochs["arm"].eq(arm)
                    & selected_epochs["architecture"].eq(architecture)
                ]["selected_epoch"].iloc[0]
            )
            for seed_value in run_seeds:
                spec = MultitaskStudySpec(
                    arm,
                    architecture,
                    seed_value,
                    selected_epoch,
                    final_train_keys,
                    test_keys,
                    float(score_logit["alpha"]),
                    float(score_logit["sigma_logit"]),
                )
                leaf = output / "refit" / arm / architecture / f"seed_{seed_value}"
                try:
                    result = train_multitask_leaf(
                        final_table,
                        spec,
                        prior_checkpoint=paths["prior"],
                        sidecars=filter_sidecars(
                            rankings, playoffs, awards, table, final_train_keys, arm
                        ),
                        output_dir=leaf,
                        log_dir=(
                            tensorboard_dir / "refit" / arm / architecture / f"seed_{seed_value}"
                            if tensorboard_dir
                            else None
                        ),
                        phase="refit",
                    )
                except (FloatingPointError, RuntimeError) as exc:
                    failed.append(
                        {
                            "phase": "refit",
                            "arm": arm,
                            "architecture": architecture,
                            "seed": seed_value,
                            "error": str(exc),
                        }
                    )
                    continue
                final_results.append((arm, architecture, seed_value))
                history = result.history.copy()
                history.insert(0, "seed", seed_value)
                history.insert(0, "architecture", architecture)
                history.insert(0, "arm", arm)
                history.insert(0, "phase", "refit")
                final_histories.append(history)
                predictions = prediction_frame(result, spec, test_keys, phase="test")
                leaf_calibrators = calibrators[
                    calibrators["supervision_mode"].eq(arm)
                    & calibrators["architecture"].eq(architecture)
                    & calibrators["seed"].astype(str).eq(str(seed_value))
                ]
                from latentstrat.season.static_core import apply_calibrators

                final_predictions.append(apply_calibrators(predictions, leaf_calibrators))
    if sha256_file(frozen_path) != frozen_hash:
        raise RuntimeError("Frozen final contract changed after refits began.")
    histories.extend(final_histories)
    history_table = pd.concat(histories, ignore_index=True)
    history_table.to_csv(output / "development_history.csv", index=False)
    complete = not failed and len(final_results) == expected_development
    neural_predictions = pd.concat(final_predictions, ignore_index=True)
    clean_final_train = table[table["match_key"].astype(str).isin(final_train_keys)]
    championship = table[table["match_key"].astype(str).isin(test_keys)].reset_index(drop=True)
    if "event_type" not in championship:
        metadata = events.rename(columns={"key": "event_key"})
        championship = championship.merge(
            metadata[["event_key", "event_type"]],
            on="event_key",
            how="left",
            validate="many_to_one",
        )
    baseline_predictions, baseline_contract = _baseline_predictions(
        clean_final_train, championship, clean_dev_train, clean_dev_holdout
    )
    baseline_contract.to_csv(output / "baseline_contract.csv", index=False)
    prediction_table = pd.concat([neural_predictions, baseline_predictions], ignore_index=True)
    prediction_table = _attach_evaluation_context(prediction_table, clean_final_train, championship)
    prediction_table.to_parquet(output / "walk_forward_predictions.parquet", index=False)
    metrics, calibration = compute_metrics(prediction_table)
    metrics.to_csv(output / "metrics.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    diagnostics = physics_diagnostics(neural_predictions)
    diagnostics.to_csv(output / "physics_diagnostics.csv", index=False)
    if complete and not smoke:
        comparisons, bootstrap, leave_one_out = hierarchical_paired_comparisons(
            neural_predictions, resamples=bootstrap_resamples
        )
        classifications = classify_comparisons(comparisons, leave_one_out)
    else:
        comparisons = bootstrap = leave_one_out = classifications = pd.DataFrame()
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    leave_one_out.to_csv(output / "leave_one_out.csv", index=False)
    classifications.to_csv(output / "decision_classifications.csv", index=False)
    coverage = {
        "complete": complete,
        "smoke": smoke,
        "failed_leaves": failed,
        "development_leaves_completed": len(development_results),
        "refit_leaves_completed": len(final_results),
        "championship_rows": len(test_keys),
        "championship_clean_rows": int(
            split_manifest.loc[
                split_manifest["split_role"].eq("championship_test"), "included_for_score"
            ].sum()
        ),
        "physics_alliance_rows": physics_audit["alliance_rows"],
        "physics_max_error": physics_audit["max_absolute_error"],
        "frozen_final_contract_sha256": frozen_hash,
        "elapsed_seconds": time.perf_counter() - started,
        "promotion_eligible": False,
    }
    (output / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_data = {
        "question": (
            "Does dense breakdown supervision, particularly when coupled through an exact "
            "differentiable 2026 score composer, improve the out-of-time Championship "
            "generalization of Full-match relative to Additive without harming official-score "
            "accuracy, relative-strength estimation, or winner probabilities?"
        ),
        "coverage": coverage,
        "selected_epochs": selected_epochs.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "physics_diagnostics": diagnostics.to_dict(orient="records"),
        "classifications": classifications.to_dict(orient="records"),
    }
    (output / "report_data.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = render_report(
        metrics=metrics,
        selected_epochs=selected_epochs,
        classifications=classifications,
        diagnostics=diagnostics,
        coverage=coverage,
        history=history_table,
    )
    report_bytes = report.encode("utf-8")
    artifact_report = output / "report.md"
    docs_report = REPOSITORY_ROOT / "docs/2026-static-multitask-physics-report.md"
    artifact_report.write_bytes(report_bytes)
    if not smoke:
        docs_report.write_bytes(report_bytes)
    (output / "report_hashes.json").write_text(
        json.dumps(
            {
                "artifact_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "documentation_sha256": sha256_file(docs_report) if docs_report.exists() else None,
                "matching": docs_report.exists()
                and sha256_file(docs_report) == hashlib.sha256(report_bytes).hexdigest(),
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
        source_files={name: path for name, path in paths.items()},
        promotion_eligible=False,
        promotion_note="Three-seed reused-Championship static diagnostic.",
    )
    return output
