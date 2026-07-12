"""Reusable fixed-objective static-study components."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from latentstrat.artifacts import sha256_file
from latentstrat.config import LatentStratOptions, TargetMapping, default_options
from latentstrat.season.data import Split
from latentstrat.season.evaluate import _predict_batched, sigmoid
from latentstrat.season.features import FeatureTrainingResult, train_feature_table
from latentstrat.season.reference_study import OFFICIAL_EVENT_TYPES
from latentstrat.season.train import create_tensorboard_writer, match_v5_matrices

ScoreMode = Literal["phase-core", "official-total-core"]
Architecture = Literal["additive", "teammate-set", "full-match"]

DEVELOPMENT_HOLDOUT_EVENTS = (
    "2026flta",
    "2026incmp",
    "2026micmp1",
    "2026micmp2",
    "2026micmp3",
    "2026micmp4",
    "2026mrcmp",
    "2026necmp1",
    "2026necmp2",
    "2026nytr",
    "2026oncmp1",
    "2026oncmp2",
    "2026txcmp1",
    "2026txcmp2",
    "2026utwv",
    "2026wicmp",
)
CHAMPIONSHIP_EVENTS = (
    "2026arc",
    "2026cur",
    "2026dal",
    "2026gal",
    "2026hop",
    "2026joh",
    "2026mil",
    "2026new",
)
EINSTEIN_EVENT = "2026cmptx"


@dataclass(frozen=True)
class StaticStudySpec:
    score_target_mode: ScoreMode
    architecture: Architecture
    epochs: int
    seed: int
    active_objectives: tuple[str, ...]
    train_match_keys: tuple[str, ...]
    evaluation_match_keys: tuple[str, ...]
    score_loss_weight: float
    win_loss_weight: float
    embedding_loss_weight: float


def _hash_strings(values: tuple[str, ...] | list[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode()
    return hashlib.sha256(payload).hexdigest()


def build_split_manifest(
    table: pd.DataFrame, event_metadata: pd.DataFrame | str | Path
) -> pd.DataFrame:
    """Return the frozen one-row-per-match-role membership table."""

    if isinstance(event_metadata, (str, Path)):
        metadata = pd.read_parquet(event_metadata)
    else:
        metadata = event_metadata.copy()
    metadata = metadata.rename(columns={"key": "event_key"})
    required_metadata = {"event_key", "start_date", "event_type"}
    if not required_metadata.issubset(metadata):
        raise ValueError("Event metadata lacks event_key, start_date, or event_type.")
    rows = table.merge(
        metadata[["event_key", "start_date", "event_type"]],
        on="event_key",
        how="left",
        validate="many_to_one",
        suffixes=("", "_metadata"),
    )
    event_type = pd.to_numeric(
        rows.get("event_type_metadata", rows.get("event_type")), errors="coerce"
    )
    score = rows[["red_score_raw", "blue_score_raw"]].apply(pd.to_numeric, errors="coerce")
    played = score.notna().all(axis=1) & score.ge(0).all(axis=1)
    standard = (
        rows.get("comp_level", pd.Series("qm", index=rows.index))
        .astype(str)
        .isin({"qm", "ef", "qf", "sf", "f"})
    )
    official = event_type.isin(OFFICIAL_EVENT_TYPES)
    dq = rows["red_has_dq"].astype(bool) | rows["blue_has_dq"].astype(bool)
    tie = pd.to_numeric(rows["red_total_score"], errors="coerce").eq(
        pd.to_numeric(rows["blue_total_score"], errors="coerce")
    )
    base = pd.DataFrame(
        {
            "match_key": rows["match_key"].astype(str),
            "event_key": rows["event_key"].astype(str),
            "event_start_date": rows["start_date"].astype(str),
            "event_type": event_type,
            "canonical_week": pd.to_numeric(rows["event_week"], errors="coerce"),
            "dq": dq,
            "tie": tie,
            "valid": official & played & standard,
        }
    )

    memberships: list[pd.DataFrame] = []

    def add(mask: pd.Series, role: str, *, reason: str = "") -> None:
        frame = base.loc[
            mask,
            [
                "match_key",
                "event_key",
                "event_start_date",
                "event_type",
                "canonical_week",
                "dq",
                "tie",
            ],
        ].copy()
        frame["split_role"] = role
        eligible_role = role not in {
            "excluded_einstein",
            "excluded_invalid",
            "excluded_dq",
        }
        frame["included_for_score"] = (~frame["dq"]) & eligible_role
        frame["included_for_win"] = frame["included_for_score"] & ~frame["tie"]
        frame["exclusion_reason"] = reason
        memberships.append(frame)

    valid = base["valid"]
    dev_holdout = valid & base["event_key"].isin(DEVELOPMENT_HOLDOUT_EVENTS)
    championship = valid & base["event_key"].isin(CHAMPIONSHIP_EVENTS)
    einstein = base["event_key"].eq(EINSTEIN_EVENT)
    dev_train = valid & base["canonical_week"].between(2, 7) & ~championship
    final_train = valid & ~base["event_key"].isin((*CHAMPIONSHIP_EVENTS, EINSTEIN_EVENT))
    add(dev_train, "development_train")
    add(dev_holdout, "development_holdout")
    add(final_train, "final_train")
    add(championship, "championship_test")
    add(einstein, "excluded_einstein", reason="Einstein is outside this diagnostic")
    add(~valid & ~einstein, "excluded_invalid", reason="nonofficial, unplayed, or nonstandard")
    add(valid & dq, "excluded_dq", reason="DQ excluded from primary loss and metrics")
    result = pd.concat(memberships, ignore_index=True)
    result = result.drop(columns=["dq", "tie"]).sort_values(
        ["split_role", "event_key", "match_key"]
    )
    if result.duplicated(["match_key", "split_role"]).any():
        raise ValueError("Split manifest contains duplicate match-role memberships.")
    validate_split_manifest(result)
    return result.reset_index(drop=True)


def validate_split_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "match_key",
        "event_key",
        "event_start_date",
        "event_type",
        "canonical_week",
        "split_role",
        "included_for_score",
        "included_for_win",
        "exclusion_reason",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError("Split manifest is missing: " + ", ".join(sorted(missing)))
    keys = {
        role: set(manifest.loc[manifest["split_role"].eq(role), "match_key"].astype(str))
        for role in manifest["split_role"].unique()
    }
    expected = {
        "development_train": 15_373,
        "development_holdout": 1_642,
        "final_train": 17_029,
        "championship_test": 1_119,
        "excluded_einstein": 16,
    }
    for role, count in expected.items():
        if len(keys.get(role, set())) != count:
            raise ValueError(
                f"Split role {role} has {len(keys.get(role, set()))}, expected {count}."
            )
    if keys["development_train"] & keys["development_holdout"]:
        raise ValueError("Development train and holdout overlap.")
    if keys["final_train"] & keys["championship_test"]:
        raise ValueError("Final train and Championship test overlap.")
    training_events = set(
        manifest.loc[
            manifest["split_role"].isin(["development_train", "final_train"]), "event_key"
        ].astype(str)
    )
    if training_events & set(CHAMPIONSHIP_EVENTS):
        raise ValueError("Championship division events entered training.")
    modeled = manifest[
        manifest["split_role"].isin(
            ["development_train", "development_holdout", "final_train", "championship_test"]
        )
    ]
    if EINSTEIN_EVENT in set(modeled["event_key"].astype(str)):
        raise ValueError("Einstein entered a modeled or evaluated split.")


def manifest_keys(
    manifest: pd.DataFrame, role: str, *, clean_score: bool = False
) -> tuple[str, ...]:
    rows = manifest[manifest["split_role"].eq(role)]
    if clean_score:
        rows = rows[rows["included_for_score"].astype(bool)]
    return tuple(sorted(rows["match_key"].astype(str)))


def core_options(spec: StaticStudySpec) -> LatentStratOptions:
    if spec.score_target_mode == "phase-core":
        targets = (
            TargetMapping(target_name="core_red_auto", tba_field="redAutoPoints"),
            TargetMapping(target_name="core_red_teleop", tba_field="redTeleopPoints"),
            TargetMapping(target_name="core_blue_auto", tba_field="blueAutoPoints"),
            TargetMapping(target_name="core_blue_teleop", tba_field="blueTeleopPoints"),
        )
    else:
        targets = (
            TargetMapping(target_name="core_red_total", tba_field="redTotalPoints"),
            TargetMapping(target_name="core_blue_total", tba_field="blueTotalPoints"),
        )
    return default_options(2026).model_copy(
        update={
            "state_model": "static-z-base",
            "match_architecture": spec.architecture,
            "core_objective": True,
            "score_target_mode": spec.score_target_mode,
            "score_loss_weight": spec.score_loss_weight,
            "win_loss_weight": spec.win_loss_weight,
            "embedding_loss_weight": spec.embedding_loss_weight,
            "l2_embedding": 0.0,
            "target_map": targets,
            "continuous_targets": tuple(item.target_name for item in targets),
            "binary_targets": ("red_win",),
            "atomic_count_targets": (),
            "foul_targets": (),
            "bonus_binary_targets": (),
            "special_binary_targets": (),
            "award_targets": (),
            "epochs": spec.epochs,
            "mini_batch_size": 256,
            "max_grad_norm": 5.0,
            "random_seed": spec.seed,
            "use_early_stopping": False,
            "restore_best_validation_model": False,
            "checkpoint_every_epochs": 1,
            "checkpoint_milestone_epochs": tuple(
                epoch for epoch in (25, 50, 75, 100) if epoch <= spec.epochs
            ),
            "scheduler": "cosine",
            "use_lr_scheduler": True,
            "log_optimizer_steps": False,
        }
    )


def prepare_core_table(table: pd.DataFrame, score_mode: ScoreMode) -> pd.DataFrame:
    out = table.copy()
    dq = out["red_has_dq"].astype(bool) | out["blue_has_dq"].astype(bool)
    tie = pd.to_numeric(out["red_total_score"], errors="coerce").eq(
        pd.to_numeric(out["blue_total_score"], errors="coerce")
    )
    out.loc[dq | tie, "red_win"] = np.nan
    if score_mode == "phase-core":
        mapping = {
            "core_red_auto": "red_auto_pts",
            "core_red_teleop": "red_teleop_pts",
            "core_blue_auto": "blue_auto_pts",
            "core_blue_teleop": "blue_teleop_pts",
        }
    else:
        mapping = {
            "core_red_total": "red_total_score",
            "core_blue_total": "blue_total_score",
        }
    for target, source in mapping.items():
        out[target] = pd.to_numeric(out[source], errors="coerce")
        out.loc[dq, target] = np.nan
    return out


def split_from_keys(
    table: pd.DataFrame, train_keys: tuple[str, ...], eval_keys: tuple[str, ...]
) -> Split:
    keys = table["match_key"].astype(str)
    train = keys.isin(train_keys).to_numpy(bool)
    validation = keys.isin(eval_keys).to_numpy(bool)
    if not train.any() or not validation.any() or np.any(train & validation):
        raise ValueError("Invalid exact-key split.")
    return Split(train, validation, ~(train | validation), "exact-match-key-static-core")


def pooled_phase_residual(table: pd.DataFrame, train_keys: tuple[str, ...]) -> dict[str, object]:
    rows = table[table["match_key"].astype(str).isin(train_keys)].copy()
    rows = rows[~(rows["red_has_dq"].astype(bool) | rows["blue_has_dq"].astype(bool))]
    red = rows["red_total_score"].to_numpy(float) - (
        rows["red_auto_pts"].to_numpy(float) + rows["red_teleop_pts"].to_numpy(float)
    )
    blue = rows["blue_total_score"].to_numpy(float) - (
        rows["blue_auto_pts"].to_numpy(float) + rows["blue_teleop_pts"].to_numpy(float)
    )
    values = np.concatenate([red, blue])
    finite = values[np.isfinite(values)]

    def summarize(group: pd.DataFrame) -> dict[str, float | int]:
        group_red = group["red_total_score"].to_numpy(float) - (
            group["red_auto_pts"].to_numpy(float) + group["red_teleop_pts"].to_numpy(float)
        )
        group_blue = group["blue_total_score"].to_numpy(float) - (
            group["blue_auto_pts"].to_numpy(float) + group["blue_teleop_pts"].to_numpy(float)
        )
        group_values = np.concatenate([group_red, group_blue])
        group_values = group_values[np.isfinite(group_values)]
        return {
            "mean": float(group_values.mean()),
            "std": float(group_values.std()),
            "count": int(len(group_values)),
        }

    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "count": int(len(finite)),
        "by_week": {
            str(key): summarize(group) for key, group in rows.groupby("event_week", dropna=False)
        },
        "by_event_type": {
            str(key): summarize(group) for key, group in rows.groupby("event_type", dropna=False)
        },
    }


def _tensorboard_contract(writer: object, contract: dict[str, object]) -> None:
    writer.add_text("Run/Contract", "```json\n" + json.dumps(contract, indent=2) + "\n```", 0)


def _write_hparams_same_run(
    writer: object,
    hparams_values: dict[str, str | int | float],
    outcomes: dict[str, float],
    *,
    step: int,
) -> None:
    """Write one HParams session into the existing leaf event stream."""

    from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams

    experiment, session_start, session_end = hparams(hparams_values, outcomes)
    plugin_data = HParamsPluginData.FromString(experiment.value[0].metadata.plugin_data.content)
    for metric in plugin_data.experiment.metric_infos:
        metric.name.group = "hparams_metrics"
    experiment.value[0].metadata.plugin_data.content = plugin_data.SerializeToString()
    writer.file_writer.add_summary(experiment, step)
    writer.file_writer.add_summary(session_start, step)
    writer.file_writer.add_summary(session_end, step)
    metric_logdir = Path(writer.log_dir) / "hparams_metrics"
    with SummaryWriter(log_dir=metric_logdir) as metric_writer:
        for tag, value in outcomes.items():
            metric_writer.add_scalar(tag, value, step)


def train_core_leaf(
    table: pd.DataFrame,
    spec: StaticStudySpec,
    *,
    prior_checkpoint: Path,
    output_dir: Path,
    log_dir: Path | None,
    phase: str,
) -> FeatureTrainingResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    options = core_options(spec)
    prepared = prepare_core_table(table, spec.score_target_mode)
    if phase == "refit":
        train_mask = prepared["match_key"].astype(str).isin(spec.train_match_keys).to_numpy(bool)
        split = Split(
            train_mask,
            np.zeros(len(prepared), dtype=bool),
            ~train_mask,
            "exact-match-key-static-core-refit",
        )
    else:
        split = split_from_keys(prepared, spec.train_match_keys, spec.evaluation_match_keys)
    resume = output_dir / "resume" / "latest.ckpt"
    purge_step = None
    resume_source: Path | None = None
    already_complete = False
    if resume.exists():
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        purge_step = int(payload["completed_epoch"]) + 1
        resume_source = resume
        already_complete = purge_step > spec.epochs
    writer = (
        create_tensorboard_writer(str(log_dir), purge_step=purge_step)
        if log_dir is not None and not already_complete
        else None
    )
    contract = {
        "spec": asdict(spec),
        "phase": phase,
        "options": options.model_dump(mode="json"),
        "train_match_key_hash": _hash_strings(list(spec.train_match_keys)),
        "evaluation_match_key_hash": _hash_strings(list(spec.evaluation_match_keys)),
        "prior_checkpoint": str(prior_checkpoint),
        "prior_checkpoint_sha256": sha256_file(prior_checkpoint),
        "pos_weight": 1.0,
    }
    try:
        if writer is not None:
            _tensorboard_contract(writer, contract)
        result = train_feature_table(
            prepared,
            options,
            verbose=True,
            prior_checkpoint=(None if resume_source else prior_checkpoint),
            sidecar_tables={},
            split_override=split,
            tensorboard_writer=writer,
            tensorboard_logdir=log_dir,
            resume_checkpoint=resume_source,
            resume_output=resume,
        )
        result.history.to_csv(output_dir / "training_history.csv", index=False)
        residual = (
            pooled_phase_residual(table, spec.train_match_keys)
            if spec.score_target_mode == "phase-core"
            else {"mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "count": 0}
        )
        checkpoint = {
            "checkpoint_schema_version": 8,
            "study_schema": "2026-static-core-v1",
            "model_state_dict": result.model.state_dict(),
            "options": options.model_dump(mode="json"),
            "target_stats": {
                "target_names": list(result.target_stats.target_names),
                "mu": result.target_stats.mu.tolist(),
                "sigma": result.target_stats.sigma.tolist(),
            },
            "team_base_index_map": result.team_index_map,
            "team_event_index_map": result.team_event_index_map,
            "spec": asdict(spec),
            "phase_residual": residual,
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")
        (output_dir / "contract.json").write_text(
            json.dumps({**contract, "phase_residual": residual}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if writer is not None:
            final = result.history.iloc[-1]
            outcomes = {
                "HParams/CompletedEpoch": float(final["epoch"]),
            }
            for tag, field in (
                ("HParams/FinalScoreRMSE", "validation_score_rmse"),
                ("HParams/FinalDifferentialRMSE", "validation_score_differential_rmse"),
                ("HParams/FinalBrier", "validation_winner_brier"),
                ("HParams/FinalLogLoss", "validation_winner_log_loss"),
            ):
                value = float(final.get(field, np.nan))
                if np.isfinite(value):
                    outcomes[tag] = value
            _write_hparams_same_run(
                writer,
                {
                    "phase": phase,
                    "supervision": spec.score_target_mode,
                    "architecture": spec.architecture,
                    "epochs": spec.epochs,
                    "seed": spec.seed,
                    "score_weight": spec.score_loss_weight,
                    "win_weight": spec.win_loss_weight,
                    "embedding_weight": spec.embedding_loss_weight,
                },
                outcomes,
                step=spec.epochs,
            )
        return result
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


def prediction_frame(
    result: FeatureTrainingResult,
    spec: StaticStudySpec,
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
    cont = prediction["cont"] * result.target_stats.sigma + result.target_stats.mu
    mask = result.prepared["match_key"].astype(str).isin(evaluation_keys).to_numpy()
    source = result.prepared.loc[mask]
    if spec.score_target_mode == "official-total-core":
        totals = cont[mask, :2]
    else:
        residual = pooled_phase_residual(result.table, spec.train_match_keys)["mean"]
        totals = np.column_stack(
            [cont[mask, :2].sum(axis=1) + residual, cont[mask, 2:4].sum(axis=1) + residual]
        )
    frame = pd.DataFrame(
        {
            "phase": phase,
            "supervision_mode": spec.score_target_mode,
            "architecture": spec.architecture,
            "seed": str(spec.seed),
            "epoch": spec.epochs,
            "event_key": source["event_key"].astype(str).to_numpy(),
            "event_week": pd.to_numeric(source["event_week"], errors="coerce").to_numpy(),
            "event_type": pd.to_numeric(source["event_type"], errors="coerce").to_numpy(),
            "match_key": source["match_key"].astype(str).to_numpy(),
            "pred_red_total_score": totals[:, 0],
            "pred_blue_total_score": totals[:, 1],
            "actual_red_total_score": source["red_total_score"].to_numpy(float),
            "actual_blue_total_score": source["blue_total_score"].to_numpy(float),
            "raw_pred_red_win_probability": sigmoid(prediction["bin"][mask, 0]),
            "red_has_dq": source["red_has_dq"].astype(bool).to_numpy(),
            "blue_has_dq": source["blue_has_dq"].astype(bool).to_numpy(),
            "red_has_surrogate": source["red_has_surrogate"].astype(bool).to_numpy(),
            "blue_has_surrogate": source["blue_has_surrogate"].astype(bool).to_numpy(),
        }
    )
    team_columns = [f"{color}_team_{slot}_key" for color in ("red", "blue") for slot in range(1, 4)]
    for column in team_columns:
        frame[column] = source[column].astype(str).to_numpy()
    training = result.prepared[result.prepared["match_key"].astype(str).isin(spec.train_match_keys)]
    appearances = pd.Series(
        training[team_columns].astype(str).to_numpy().reshape(-1)
    ).value_counts()
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
    return frame


def fit_calibrators(predictions: pd.DataFrame, model_hash: str) -> pd.DataFrame:
    actual_red = predictions["actual_red_total_score"].to_numpy(float)
    actual_blue = predictions["actual_blue_total_score"].to_numpy(float)
    clean = ~(
        predictions["red_has_dq"].astype(bool).to_numpy()
        | predictions["blue_has_dq"].astype(bool).to_numpy()
        | np.isclose(actual_red, actual_blue)
    )
    outcome = (actual_red > actual_blue).astype(int)
    raw = np.clip(predictions["raw_pred_red_win_probability"].to_numpy(float), 1e-7, 1 - 1e-7)
    logit = np.log(raw / (1 - raw))
    diff = predictions["pred_red_total_score"].to_numpy(float) - predictions[
        "pred_blue_total_score"
    ].to_numpy(float)
    rows = []
    for kind, values in (("independent-platt", logit), ("score-differential-logistic", diff)):
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=2026)
        model.fit(values[clean, None], outcome[clean])
        rows.append(
            {
                "calibrator": kind,
                "supervision_mode": predictions["supervision_mode"].iloc[0],
                "architecture": predictions["architecture"].iloc[0],
                "source_match_key_hash": _hash_strings(
                    predictions["match_key"].astype(str).tolist()
                ),
                "source_row_count": int(len(predictions)),
                "source_non_tie_count": int(clean.sum()),
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
                "development_model_hash": model_hash,
                "development_prediction_hash": hashlib.sha256(
                    pd.util.hash_pandas_object(predictions, index=False).values.tobytes()
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def apply_calibrators(predictions: pd.DataFrame, calibrators: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    raw = np.clip(out["raw_pred_red_win_probability"].to_numpy(float), 1e-7, 1 - 1e-7)
    logit = np.log(raw / (1 - raw))
    diff = out["pred_red_total_score"].to_numpy(float) - out["pred_blue_total_score"].to_numpy(
        float
    )
    for kind, values, column in (
        ("independent-platt", logit, "transferred_platt_red_win_probability"),
        (
            "score-differential-logistic",
            diff,
            "transferred_score_red_win_probability",
        ),
    ):
        row = calibrators[calibrators["calibrator"].eq(kind)].iloc[0]
        out[column] = 1 / (
            1 + np.exp(-(float(row["coefficient"]) * values + float(row["intercept"])))
        )
    return out
