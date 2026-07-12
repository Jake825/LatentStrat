"""Experimental offline frozen target spaces."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from latentstrat.config import PriorOpts
from latentstrat.pretraining.prior.features import EmbeddingStats, embed_narratives
from latentstrat.season.metrics import (
    paired_bootstrap_noninferiority as _supported_paired_bootstrap_noninferiority,
)
from latentstrat.training_runtime import (
    OptimizationController,
    TrainingRuntimeConfig,
    build_lr_scheduler,
    epoch_runtime_metrics,
    load_resume_checkpoint,
    load_training_state,
    resume_payload,
    save_resume_checkpoint,
    seed_everything,
    validate_resume_checkpoint,
)

PARQUET_ENGINE = "pyarrow"
OFFICIAL_EVENT_TYPES = frozenset(range(8))
SCORE_INTEGRATION_PENDING = (
    "score_embedding is offline-only in V6.1. Build and validate the historical "
    "match-breakdown encoder first; per-alliance Set Transformer integration is deferred."
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TargetSpaceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    width: int = Field(gt=0)


class WorldModelOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    baseline_ref: str = "v5.8-baseline"
    score_embedding: TargetSpaceOptions = TargetSpaceOptions(enabled=False, width=16)
    award_embedding: TargetSpaceOptions = TargetSpaceOptions(enabled=False, width=256)
    rank_embedding: TargetSpaceOptions = TargetSpaceOptions(enabled=False, width=16)
    pick_embedding: TargetSpaceOptions = TargetSpaceOptions(enabled=False, width=16)
    target_epochs: int = Field(default=50, gt=0)
    target_learning_rate: float = Field(default=1e-3, gt=0)
    random_seed: int = 2026
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    lr_eta_min: float = 1e-5
    one_cycle_pct_start: float = 0.3
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: float = 10_000.0
    deterministic_algorithms: bool = False
    checkpoint_every_epochs: int = 1
    optimizer_log_interval: int = 10

    def active_spaces(self) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        if self.score_embedding.enabled:
            raise ValueError(SCORE_INTEGRATION_PENDING)
        return tuple(
            name
            for name, options in (
                ("award", self.award_embedding),
                ("rank", self.rank_embedding),
                ("pick", self.pick_embedding),
            )
            if options.enabled
        )

    def space(self, name: str) -> TargetSpaceOptions:
        return {
            "score": self.score_embedding,
            "award": self.award_embedding,
            "rank": self.rank_embedding,
            "pick": self.pick_embedding,
        }[name]


@dataclass
class FrozenTargetSpace:
    name: str
    root: Path
    embeddings: pd.DataFrame
    bundle: dict[str, Any]
    schema: dict[str, Any]


@dataclass
class LoadedWorldModelBundle:
    root: Path
    options: WorldModelOptions
    spaces: dict[str, FrozenTargetSpace]


def load_world_model_options(path: str | Path) -> WorldModelOptions:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    frozen_targets = payload.get("frozen_targets", payload.get("world_model", payload))
    return WorldModelOptions.model_validate(frozen_targets)


def _git_head_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _embedding_columns(width: int) -> list[str]:
    return [f"z_{idx:03d}" for idx in range(width)]


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_record(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    source = Path(path)
    return {"path": str(source), "sha256": sha256_file(source)}


def _write_target_space(
    output_dir: str | Path,
    *,
    name: str,
    embeddings: pd.DataFrame,
    model_payload: dict[str, Any],
    schema: dict[str, Any],
    bundle: dict[str, Any],
    report: dict[str, Any],
) -> FrozenTargetSpace:
    root = Path(output_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model_payload, root / "model.pt")
    embeddings.to_parquet(root / "embeddings.parquet", engine=PARQUET_ENGINE, index=False)
    _json_write(root / "schema.json", schema)
    _json_write(root / "bundle.json", bundle)
    _json_write(root / "validation_report.json", report)
    return FrozenTargetSpace(
        name=name, root=root, embeddings=embeddings, bundle=bundle, schema=schema
    )


def load_target_space(path: str | Path) -> FrozenTargetSpace:
    root = Path(path)
    return FrozenTargetSpace(
        name=root.name,
        root=root,
        embeddings=pd.read_parquet(root / "embeddings.parquet", engine=PARQUET_ENGINE),
        bundle=json.loads((root / "bundle.json").read_text(encoding="utf-8")),
        schema=json.loads((root / "schema.json").read_text(encoding="utf-8")),
    )


def load_world_model_bundle(path: str | Path) -> LoadedWorldModelBundle:
    root = Path(path)
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    options_payload = payload.get("frozen_targets")
    if options_payload is None:
        options_payload = payload["world_model"]
    options = WorldModelOptions.model_validate(options_payload)
    spaces = {
        name: load_target_space(root / relative_path)
        for name, relative_path in payload.get("spaces", {}).items()
    }
    missing = sorted(set(options.active_spaces()) - set(spaces))
    if missing:
        raise ValueError("World-model bundle is missing enabled spaces: " + ", ".join(missing))
    return LoadedWorldModelBundle(root=root, options=options, spaces=spaces)


def _official_rows(
    table: pd.DataFrame,
    event_metadata: pd.DataFrame | None,
    *,
    fit_max_week: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = table.copy()
    if event_metadata is not None:
        metadata = event_metadata.rename(columns={"key": "event_key"})
        if "event_key" not in metadata.columns:
            raise ValueError("Official-event filtering requires event_key and event_type metadata.")
        metadata_columns = [
            column
            for column in ("event_type", "event_completed", "end_date")
            if column not in rows.columns and column in metadata.columns
        ]
        if metadata_columns:
            rows = rows.merge(
                metadata[["event_key", *metadata_columns]].drop_duplicates("event_key"),
                on="event_key",
                how="left",
                validate="many_to_one",
            )
    if "event_type" not in rows.columns:
        raise ValueError("Official-event filtering requires event_key and event_type metadata.")
    event_type = pd.to_numeric(rows["event_type"], errors="coerce")
    official = event_type.isin(OFFICIAL_EVENT_TYPES)
    week_ok = pd.Series(True, index=rows.index)
    if fit_max_week is not None:
        if "event_week" not in rows.columns:
            raise ValueError("Fold-local target fitting requires event_week.")
        week_ok = pd.to_numeric(rows["event_week"], errors="coerce") <= int(fit_max_week)
    report = {
        "input_rows": int(len(rows)),
        "official_rows": int(official.sum()),
        "excluded_non_official_rows": int((~official).sum()),
        "excluded_future_rows": int((official & ~week_ok.fillna(False)).sum()),
    }
    return rows.loc[official & week_ok.fillna(False)].reset_index(drop=True), report


def _finite_matrix(table: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = table.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return values, np.isfinite(values)


class NumericTargetAutoencoder(nn.Module):
    def __init__(self, input_width: int, latent_width: int) -> None:
        super().__init__()
        hidden = max(2 * latent_width, input_width, 16)
        self.encoder = nn.Sequential(
            nn.Linear(input_width, hidden), nn.GELU(), nn.Linear(hidden, latent_width)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_width, hidden), nn.GELU(), nn.Linear(hidden, input_width)
        )

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encoder(values)
        return self.decoder(latent), latent


def _grouped_masked_mse(
    prediction: Tensor,
    target: Tensor,
    observed: Tensor,
    groups: dict[str, list[int]],
) -> Tensor:
    losses = []
    for indices in groups.values():
        group_mask = observed[:, indices]
        if torch.any(group_mask):
            losses.append(
                F.mse_loss(prediction[:, indices][group_mask], target[:, indices][group_mask])
            )
    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def _effective_rank(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    singular = np.linalg.svd(embeddings - embeddings.mean(axis=0), compute_uv=False)
    if not np.any(singular > 0):
        return 0.0
    probability = singular / singular.sum()
    return float(np.exp(-np.sum(probability * np.log(probability + np.finfo(float).eps))))


def _nearest_neighbor_report(table: pd.DataFrame, embeddings: np.ndarray) -> list[dict[str, Any]]:
    if len(embeddings) < 2:
        return []
    label_column = next(
        (column for column in ("match_key", "team_key", "award_id") if column in table.columns),
        None,
    )
    labels = (
        table[label_column].astype(str).tolist()
        if label_column is not None
        else [str(idx) for idx in range(len(table))]
    )
    distances = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    rows = []
    for idx in range(min(len(embeddings), 10)):
        neighbor = int(np.argmin(distances[idx]))
        rows.append(
            {
                "row": labels[idx],
                "nearest_neighbor": labels[neighbor],
                "distance": float(distances[idx, neighbor]),
            }
        )
    return rows


def _linear_probe_report(
    embeddings: np.ndarray,
    normalized: np.ndarray,
    observed: np.ndarray,
    columns: list[str],
) -> dict[str, float | None]:
    result = {}
    design = np.column_stack([np.ones(len(embeddings)), embeddings])
    for idx, column in enumerate(columns):
        valid = observed[:, idx]
        target = normalized[valid, idx]
        if valid.sum() < 2 or np.allclose(target, target[0]):
            result[column] = None
            continue
        coefficients, *_ = np.linalg.lstsq(design[valid], target, rcond=None)
        prediction = design[valid] @ coefficients
        residual = float(np.sum((target - prediction) ** 2))
        total = float(np.sum((target - target.mean()) ** 2))
        result[column] = 1.0 - residual / total
    return result


def _runtime_config(options: WorldModelOptions) -> TrainingRuntimeConfig:
    return TrainingRuntimeConfig(
        gradient_accumulation_steps=options.gradient_accumulation_steps,
        max_grad_norm=options.max_grad_norm,
        scheduler=options.scheduler,
        lr_eta_min=options.lr_eta_min,
        one_cycle_pct_start=options.one_cycle_pct_start,
        one_cycle_div_factor=options.one_cycle_div_factor,
        one_cycle_final_div_factor=options.one_cycle_final_div_factor,
        deterministic_algorithms=options.deterministic_algorithms,
        checkpoint_every_epochs=options.checkpoint_every_epochs,
        optimizer_log_interval=options.optimizer_log_interval,
    )


def _write_runtime_epoch(
    writer: Any | None, phase: str, row: dict[str, Any], epoch: int
) -> None:
    if writer is None:
        return
    for name, value in row.items():
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            writer.add_scalar(f"FrozenTargets/{phase}/{name}", float(value), epoch)


def _save_frozen_resume(
    path: str | Path | None,
    runtime_config: TrainingRuntimeConfig,
    *,
    epoch: int,
    phase: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    controller: OptimizationController,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    source_fingerprint: str,
) -> None:
    if (
        path is None
        or runtime_config.checkpoint_every_epochs <= 0
        or epoch % runtime_config.checkpoint_every_epochs
    ):
        return
    save_resume_checkpoint(
        path,
        resume_payload(
            trainer="frozen-targets",
            phase=phase,
            completed_epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_step=controller.optimizer_step,
            history=history,
            config=config,
            source_fingerprint=source_fingerprint,
        ),
    )


def _train_numeric_space(
    table: pd.DataFrame,
    *,
    columns: list[str],
    groups: dict[str, list[int]],
    width: int,
    options: WorldModelOptions,
    phase: str,
    resume_checkpoint: str | Path | None = None,
    resume_output: str | Path | None = None,
    tensorboard_writer: Any | None = None,
) -> tuple[NumericTargetAutoencoder, np.ndarray, dict[str, Any], dict[str, Any]]:
    values, observed = _finite_matrix(table, columns)
    if not np.any(observed):
        raise ValueError("Numeric target space has no observed values.")
    mu = np.nanmean(values, axis=0)
    sigma = np.nanstd(values, axis=0)
    sigma[~np.isfinite(sigma) | (sigma < 1e-8)] = 1.0
    normalized = (values - mu) / sigma
    normalized[~observed] = 0.0
    runtime_config = _runtime_config(options)
    seed_everything(
        options.random_seed, deterministic_algorithms=options.deterministic_algorithms
    )
    model = NumericTargetAutoencoder(len(columns), width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.target_learning_rate)
    scheduler = build_lr_scheduler(
        optimizer,
        runtime_config,
        epochs=options.target_epochs,
        microbatches_per_epoch=1,
    )
    controller = OptimizationController(
        model,
        optimizer,
        runtime_config,
        scheduler=scheduler,
        tensorboard_writer=tensorboard_writer,
        tensorboard_prefix=f"Optimization/FrozenTargets/{phase}",
    )
    x = torch.as_tensor(normalized, dtype=torch.float32)
    mask = torch.as_tensor(observed, dtype=torch.bool)
    history = []
    resolved_config = options.model_dump(mode="json")
    digest = hashlib.sha256()
    digest.update(x.numpy().tobytes())
    digest.update(mask.numpy().tobytes())
    digest.update(json.dumps(groups, sort_keys=True).encode("utf-8"))
    source_fingerprint = digest.hexdigest()
    start_epoch = 1
    if resume_checkpoint is not None:
        payload = load_resume_checkpoint(resume_checkpoint)
        validate_resume_checkpoint(
            payload,
            trainer="frozen-targets",
            phase=phase,
            config=resolved_config,
            source_fingerprint=source_fingerprint,
        )
        load_training_state(
            payload, model=model, optimizer=optimizer, scheduler=scheduler
        )
        history = list(payload.get("history", []))
        controller.optimizer_step = int(payload.get("optimizer_step", 0))
        start_epoch = int(payload["completed_epoch"]) + 1
    for epoch in range(start_epoch, options.target_epochs + 1):
        epoch_started = time.perf_counter()
        epoch_step_start = len(controller.step_results)
        reconstruction, _ = model(x)
        loss = _grouped_masked_mse(reconstruction, x, mask, groups)
        controller.backward(loss, microbatch_index=0, microbatch_count=1)
        row = {"epoch": epoch, "grouped_reconstruction_loss": float(loss.detach())}
        row.update(
            epoch_runtime_metrics(
                started_at=epoch_started,
                sample_count=len(x),
                optimizer_summary=controller.epoch_summary(start_index=epoch_step_start),
            )
        )
        history.append(row)
        _write_runtime_epoch(tensorboard_writer, phase, row, epoch)
        _save_frozen_resume(
            resume_output,
            runtime_config,
            epoch=epoch,
            phase=phase,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            controller=controller,
            history=history,
            config=resolved_config,
            source_fingerprint=source_fingerprint,
        )
    model.eval()
    with torch.inference_mode():
        reconstruction, latent = model(x)
    reconstruction_np = reconstruction.numpy()
    group_mse = {}
    for name, indices in groups.items():
        valid = observed[:, indices]
        group_mse[name] = (
            float(
                np.mean(
                    (reconstruction_np[:, indices][valid] - normalized[:, indices][valid])
                    ** 2
                )
            )
            if np.any(valid)
            else math.nan
        )
    normalizers = {
        column: {"mu": float(mu[idx]), "sigma": float(sigma[idx])}
        for idx, column in enumerate(columns)
    }
    report = {
        "grouped_reconstruction_mse": group_mse,
        "effective_rank": _effective_rank(latent.numpy()),
        "nearest_neighbors": _nearest_neighbor_report(table, latent.numpy()),
        "linear_probes_r2": _linear_probe_report(
            latent.numpy(), normalized, observed, columns
        ),
        "history": history,
    }
    return model, latent.numpy(), normalizers, report


def _base_bundle(
    *,
    name: str,
    width: int,
    season: int,
    fit_max_week: int | None,
    options: WorldModelOptions,
    source_path: str | Path | None,
    audit: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "space": name,
        "season": season,
        "width": width,
        "fit_max_week": fit_max_week,
        "official_event_types": sorted(OFFICIAL_EVENT_TYPES),
        "baseline_ref": options.baseline_ref,
        "git_commit": _git_head_commit(),
        "source": _source_record(source_path),
        "audit": audit,
    }


def _semantic_award_rows(catalog_path: str | Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(catalog_path).read_text(encoding="utf-8")) or {}
    rows = list(payload.get("awards", []))
    if not rows:
        raise ValueError("Award prototype catalog has no awards.")
    for row in rows:
        if not row.get("id") or not row.get("description"):
            raise ValueError("Each award prototype requires id and description.")
    return rows


def build_award_target_space(
    catalog_path: str | Path,
    output_dir: str | Path,
    options: WorldModelOptions,
    *,
    cache_path: str | Path,
    client: Any | None = None,
) -> FrozenTargetSpace:
    rows = _semantic_award_rows(catalog_path)
    narratives = {}
    keys_by_award: dict[str, list[str]] = {}
    for award in rows:
        award_id = str(award["id"])
        variants = [str(award["description"]), *[str(value) for value in award.get("variants", [])]]
        keys_by_award[award_id] = []
        for idx, text in enumerate(variants):
            key = f"{award_id}:{idx}"
            keys_by_award[award_id].append(key)
            narratives[key] = text
    prior_options = PriorOpts(
        llm_dim=options.award_embedding.width,
        cache_path=str(cache_path),
    )
    stats = EmbeddingStats()
    raw = embed_narratives(narratives, prior_options, client=client, stats=stats)
    output_rows = []
    for award in rows:
        award_id = str(award["id"])
        variants = np.stack([np.asarray(raw[key], dtype=float) for key in keys_by_award[award_id]])
        norms = np.linalg.norm(variants, axis=1, keepdims=True).clip(min=np.finfo(float).eps)
        prototype = np.mean(variants / norms, axis=0)
        prototype = prototype / max(np.linalg.norm(prototype), np.finfo(float).eps)
        output_rows.append(
            {
                "award_id": award_id,
                "family": str(award.get("family", "")),
                **dict(zip(_embedding_columns(len(prototype)), prototype, strict=True)),
            }
        )
    embeddings = pd.DataFrame(output_rows)
    width = options.award_embedding.width
    bundle = _base_bundle(
        name="award",
        width=width,
        season=0,
        fit_max_week=None,
        options=options,
        source_path=catalog_path,
        audit={"prototype_count": len(embeddings)},
    )
    bundle["row_keys"] = ["award_id"]
    bundle["embedding_model"] = prior_options.embedding_model
    report = {
        "prototype_count": len(embeddings),
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "api_batches": stats.api_batches,
        "nearest_neighbors": _nearest_neighbor_report(
            embeddings, embeddings[_embedding_columns(width)].to_numpy()
        ),
    }
    return _write_target_space(
        output_dir,
        name="award",
        embeddings=embeddings,
        model_payload={"kind": "normalized-openai-prototypes", "width": width},
        schema={"row_keys": ["award_id"], "embedding_columns": _embedding_columns(width)},
        bundle=bundle,
        report=report,
    )


def _playoff_team_outcomes(playoffs: pd.DataFrame | None) -> pd.DataFrame:
    if playoffs is None or playoffs.empty:
        return pd.DataFrame(columns=["event_key", "team_key", "playoff_finish_order"])
    rows = []
    for row in playoffs.itertuples():
        for slot in (1, 2, 3):
            team_key = str(getattr(row, f"team_{slot}_key", "") or "")
            if team_key:
                rows.append(
                    {
                        "event_key": str(row.event_key),
                        "team_key": team_key,
                        "playoff_finish_order": getattr(row, "playoff_finish_order", np.nan),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["event_key", "team_key"])


def build_rank_target_space(
    rankings: pd.DataFrame,
    output_dir: str | Path,
    options: WorldModelOptions,
    *,
    playoffs: pd.DataFrame | None = None,
    event_metadata: pd.DataFrame | None = None,
    source_path: str | Path | None = None,
    fit_max_week: int | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_output: str | Path | None = None,
    tensorboard_writer: Any | None = None,
) -> FrozenTargetSpace:
    rows, audit = _official_rows(rankings, event_metadata, fit_max_week=fit_max_week)
    if "event_completed" in rows.columns:
        completed = rows["event_completed"].fillna(False).astype(bool)
    elif "end_date" in rows.columns:
        completed = pd.to_datetime(rows["end_date"], errors="coerce").dt.date < date.today()
    else:
        completed = None
    if completed is not None:
        audit["excluded_incomplete_event_rows"] = int((~completed).sum())
        rows = rows.loc[completed].reset_index(drop=True)
    else:
        audit["excluded_incomplete_event_rows"] = 0
    rows = rows.merge(_playoff_team_outcomes(playoffs), on=["event_key", "team_key"], how="left")
    columns = [
        column
        for column in rows.columns
        if column in {"qual_rank", "matches_played", "playoff_finish_order"}
        or column.startswith(("record_", "ranking_stat_"))
    ]
    columns = [
        column
        for column in columns
        if pd.to_numeric(rows[column], errors="coerce").notna().any()
    ]
    groups = {column: [idx] for idx, column in enumerate(columns)}
    width = options.rank_embedding.width
    model, latent, normalizers, report = _train_numeric_space(
        rows,
        columns=columns,
        groups=groups,
        width=width,
        options=options,
        phase="rank",
        resume_checkpoint=resume_checkpoint,
        resume_output=resume_output,
        tensorboard_writer=tensorboard_writer,
    )
    embeddings = rows[["event_key", "team_key", "event_week"]].copy()
    embeddings[_embedding_columns(width)] = latent
    bundle = _base_bundle(
        name="rank",
        width=width,
        season=int(rows["season"].iloc[0]),
        fit_max_week=fit_max_week,
        options=options,
        source_path=source_path,
        audit=audit,
    )
    bundle["row_keys"] = ["event_key", "team_key"]
    bundle["normalizers"] = normalizers
    return _write_target_space(
        output_dir,
        name="rank",
        embeddings=embeddings,
        model_payload={"state_dict": model.state_dict(), "columns": columns, "groups": groups},
        schema={
            "row_keys": ["event_key", "team_key"],
            "embedding_columns": _embedding_columns(width),
        },
        bundle=bundle,
        report=report,
    )


def ranked_unselected_opportunities(
    selections: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    ranking_groups = {
        str(event_key): event_rows.dropna(subset=["qual_rank"]).sort_values("qual_rank")
        for event_key, event_rows in rankings.groupby("event_key")
    }
    for event_key, event_rows in selections.groupby("event_key", sort=True):
        ranked = ranking_groups.get(str(event_key))
        if ranked is None or ranked.empty:
            continue
        selected = set(event_rows["captain_team_key"].dropna().astype(str))
        ordered = event_rows.sort_values(["alliance_number", "pick_order"])
        for snapshot, selection in enumerate(ordered.itertuples(), start=1):
            captain = str(selection.captain_team_key)
            picked = str(selection.pick_team_key)
            for rank in ranked.itertuples():
                candidate = str(rank.team_key)
                if candidate in selected:
                    continue
                rows.append(
                    {
                        "season": int(selection.season),
                        "event_key": str(event_key),
                        "event_week": getattr(selection, "event_week", np.nan),
                        "snapshot_number": snapshot,
                        "captain_team_key": captain,
                        "candidate_team_key": candidate,
                        "candidate_qual_rank": int(rank.qual_rank),
                        "picked": candidate == picked,
                    }
                )
            selected.add(picked)
    return pd.DataFrame(rows)


class PickTargetEncoder(nn.Module):
    def __init__(self, num_teams: int, num_events: int, width: int) -> None:
        super().__init__()
        self.team = nn.Embedding(num_teams, width)
        self.event = nn.Embedding(num_events, width)
        self.project = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.score = nn.Linear(width, 1)

    def forward(
        self,
        captain: Tensor,
        candidate: Tensor,
        event: Tensor,
        pool_membership: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pool = pool_membership.to(dtype=self.team.weight.dtype) @ self.team.weight
        pool = pool / pool_membership.sum(dim=1, keepdim=True).clamp_min(1)
        latent = self.project(
            torch.cat(
                [self.team(captain), self.team(candidate), self.event(event), pool],
                dim=1,
            )
        )
        return latent, self.score(latent).squeeze(1)


def build_pick_target_space(
    selections: pd.DataFrame,
    rankings: pd.DataFrame,
    output_dir: str | Path,
    options: WorldModelOptions,
    *,
    event_metadata: pd.DataFrame | None = None,
    source_path: str | Path | None = None,
    fit_max_week: int | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_output: str | Path | None = None,
    tensorboard_writer: Any | None = None,
) -> FrozenTargetSpace:
    rows, audit = _official_rows(selections, event_metadata, fit_max_week=fit_max_week)
    rank_rows, _ = _official_rows(rankings, event_metadata, fit_max_week=fit_max_week)
    opportunities = ranked_unselected_opportunities(rows, rank_rows)
    if opportunities.empty or not opportunities["picked"].any():
        raise ValueError("Pick target space has no ranked-unselected opportunity snapshots.")
    team_keys = sorted(
        set(opportunities["captain_team_key"]) | set(opportunities["candidate_team_key"])
    )
    event_keys = sorted(set(opportunities["event_key"]))
    team_index = {key: idx for idx, key in enumerate(team_keys)}
    event_index = {key: idx for idx, key in enumerate(event_keys)}
    captain = torch.as_tensor([team_index[key] for key in opportunities["captain_team_key"]])
    candidate = torch.as_tensor([team_index[key] for key in opportunities["candidate_team_key"]])
    event = torch.as_tensor([event_index[key] for key in opportunities["event_key"]])
    picked = torch.as_tensor(opportunities["picked"].to_numpy(bool, copy=True))
    pool_membership = torch.zeros((len(opportunities), len(team_keys)), dtype=torch.float32)
    for _, indices in opportunities.groupby(["event_key", "snapshot_number"]).groups.items():
        index_list = list(indices)
        pool_indices = [
            team_index[key] for key in opportunities.loc[index_list, "candidate_team_key"]
        ]
        for row_index in index_list:
            pool_membership[row_index, pool_indices] = 1.0
    width = options.pick_embedding.width
    runtime_config = _runtime_config(options)
    seed_everything(
        options.random_seed, deterministic_algorithms=options.deterministic_algorithms
    )
    model = PickTargetEncoder(len(team_keys), len(event_keys), width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.target_learning_rate)
    scheduler = build_lr_scheduler(
        optimizer,
        runtime_config,
        epochs=options.target_epochs,
        microbatches_per_epoch=1,
    )
    controller = OptimizationController(
        model,
        optimizer,
        runtime_config,
        scheduler=scheduler,
        tensorboard_writer=tensorboard_writer,
        tensorboard_prefix="Optimization/FrozenTargets/pick",
    )
    history = []
    resolved_config = options.model_dump(mode="json")
    digest = hashlib.sha256()
    for tensor in (captain, candidate, event, picked, pool_membership):
        digest.update(tensor.contiguous().numpy().tobytes())
    source_fingerprint = digest.hexdigest()
    start_epoch = 1
    if resume_checkpoint is not None:
        payload = load_resume_checkpoint(resume_checkpoint)
        validate_resume_checkpoint(
            payload,
            trainer="frozen-targets",
            phase="pick",
            config=resolved_config,
            source_fingerprint=source_fingerprint,
        )
        load_training_state(
            payload, model=model, optimizer=optimizer, scheduler=scheduler
        )
        history = list(payload.get("history", []))
        controller.optimizer_step = int(payload.get("optimizer_step", 0))
        start_epoch = int(payload["completed_epoch"]) + 1
    for epoch in range(start_epoch, options.target_epochs + 1):
        epoch_started = time.perf_counter()
        epoch_step_start = len(controller.step_results)
        _, scores = model(captain, candidate, event, pool_membership)
        losses = []
        for _, indices in opportunities.groupby(["event_key", "snapshot_number"]).groups.items():
            idx = torch.as_tensor(list(indices), dtype=torch.long)
            positive = scores[idx][picked[idx]]
            negative = scores[idx][~picked[idx]]
            if positive.numel() and negative.numel():
                losses.append(torch.relu(0.5 - positive.mean() + negative).mean())
        if not losses:
            raise ValueError(
                "Pick target snapshots require at least one ranked-unselected negative."
            )
        loss = torch.stack(losses).mean()
        controller.backward(loss, microbatch_index=0, microbatch_count=1)
        row = {"epoch": epoch, "contrastive_margin_loss": float(loss.detach())}
        row.update(
            epoch_runtime_metrics(
                started_at=epoch_started,
                sample_count=len(opportunities),
                optimizer_summary=controller.epoch_summary(start_index=epoch_step_start),
            )
        )
        history.append(row)
        _write_runtime_epoch(tensorboard_writer, "pick", row, epoch)
        _save_frozen_resume(
            resume_output,
            runtime_config,
            epoch=epoch,
            phase="pick",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            controller=controller,
            history=history,
            config=resolved_config,
            source_fingerprint=source_fingerprint,
        )
    model.eval()
    with torch.inference_mode():
        latent, scores = model(captain, candidate, event, pool_membership)
    embeddings = opportunities.copy()
    embeddings[_embedding_columns(width)] = latent.numpy()
    bundle = _base_bundle(
        name="pick",
        width=width,
        season=int(opportunities["season"].iloc[0]),
        fit_max_week=fit_max_week,
        options=options,
        source_path=source_path,
        audit=audit,
    )
    bundle["row_keys"] = [
        "event_key",
        "snapshot_number",
        "captain_team_key",
        "candidate_team_key",
    ]
    scores_np = scores.numpy()
    retrieval_hits = []
    score_margins = []
    for _, indices in opportunities.groupby(["event_key", "snapshot_number"]).groups.items():
        index_list = list(indices)
        snapshot_scores = scores_np[index_list]
        snapshot_picked = opportunities.loc[index_list, "picked"].to_numpy(bool)
        if np.any(snapshot_picked):
            positive_score = float(np.max(snapshot_scores[snapshot_picked]))
            retrieval_hits.append(positive_score >= float(np.max(snapshot_scores)))
            if np.any(~snapshot_picked):
                score_margins.append(
                    positive_score - float(np.max(snapshot_scores[~snapshot_picked]))
                )
    report = {
        "opportunity_rows": len(embeddings),
        "snapshot_count": int(
            embeddings[["event_key", "snapshot_number"]].drop_duplicates().shape[0]
        ),
        "retrieval_top1_accuracy": float(np.mean(retrieval_hits)),
        "picked_score_margin_mean": float(np.mean(score_margins)),
        "history": history,
    }
    return _write_target_space(
        output_dir,
        name="pick",
        embeddings=embeddings,
        model_payload={
            "state_dict": model.state_dict(),
            "team_index": team_index,
            "event_index": event_index,
            "offline_context": "ranked-unselected-pool-mean",
        },
        schema={
            "row_keys": ["event_key", "snapshot_number", "captain_team_key", "candidate_team_key"],
            "embedding_columns": _embedding_columns(width),
        },
        bundle=bundle,
        report=report,
    )


def build_world_model_bundle(
    features_path: str | Path,
    output_dir: str | Path,
    options: WorldModelOptions,
    *,
    events_path: str | Path,
    rankings_path: str | Path | None = None,
    selections_path: str | Path | None = None,
    playoffs_path: str | Path | None = None,
    award_catalog_path: str | Path | None = None,
    openai_cache_path: str | Path | None = None,
    fit_max_week: int | None = None,
    client: Any | None = None,
    resume_checkpoint: str | Path | None = None,
    tensorboard_logdir: str | Path | None = None,
    tensorboard_run_name: str | None = None,
) -> LoadedWorldModelBundle:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    if tensorboard_logdir is not None:
        from torch.utils.tensorboard import SummaryWriter

        run_name = tensorboard_run_name or f"frozen_targets_{int(time.time())}"
        writer = SummaryWriter(str(Path(tensorboard_logdir) / run_name))
    resume_phase = None
    if resume_checkpoint is not None:
        resume_phase = str(load_resume_checkpoint(resume_checkpoint).get("phase"))
    events = pd.read_parquet(events_path, engine=PARQUET_ENGINE)
    rankings = pd.read_parquet(rankings_path, engine=PARQUET_ENGINE) if rankings_path else None
    selections = (
        pd.read_parquet(selections_path, engine=PARQUET_ENGINE) if selections_path else None
    )
    playoffs = pd.read_parquet(playoffs_path, engine=PARQUET_ENGINE) if playoffs_path else None
    spaces = {}
    if "award" in options.active_spaces():
        if award_catalog_path is None or openai_cache_path is None:
            raise ValueError("Award prototypes require award_catalog_path and openai_cache_path.")
        spaces["award"] = build_award_target_space(
            award_catalog_path, out, options, cache_path=openai_cache_path, client=client
        )
    if "rank" in options.active_spaces():
        if rankings is None:
            raise ValueError("Rank target space requires rankings_path.")
        spaces["rank"] = build_rank_target_space(
            rankings,
            out,
            options,
            playoffs=playoffs,
            event_metadata=events,
            source_path=rankings_path,
            fit_max_week=fit_max_week,
            resume_checkpoint=resume_checkpoint if resume_phase == "rank" else None,
            resume_output=out / "resume" / "rank" / "latest.ckpt",
            tensorboard_writer=writer,
        )
    if "pick" in options.active_spaces():
        if selections is None or rankings is None:
            raise ValueError("Pick target space requires selections_path and rankings_path.")
        spaces["pick"] = build_pick_target_space(
            selections,
            rankings,
            out,
            options,
            event_metadata=events,
            source_path=selections_path,
            fit_max_week=fit_max_week,
            resume_checkpoint=resume_checkpoint if resume_phase == "pick" else None,
            resume_output=out / "resume" / "pick" / "latest.ckpt",
            tensorboard_writer=writer,
        )
    payload = {
        "schema_version": 1,
        "frozen_targets": options.model_dump(mode="json"),
        # Kept for historical schema-6 readers through the V6.2 migration window.
        "world_model": options.model_dump(mode="json"),
        "spaces": {name: space.root.name for name, space in spaces.items()},
        "sources": {
            "features": _source_record(features_path),
            "events": _source_record(events_path),
            "rankings": _source_record(rankings_path),
            "selections": _source_record(selections_path),
            "playoffs": _source_record(playoffs_path),
            "award_catalog": _source_record(award_catalog_path),
        },
        "git_commit": _git_head_commit(),
        "fit_max_week": fit_max_week,
    }
    _json_write(out / "bundle.json", payload)
    if writer is not None:
        writer.flush()
        writer.close()
    return LoadedWorldModelBundle(root=out, options=options, spaces=spaces)


def attach_world_model_targets(
    table: pd.DataFrame,
    sidecars: dict[str, pd.DataFrame] | None,
    bundle: LoadedWorldModelBundle,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    out = table.copy()
    attached = {name: frame.copy() for name, frame in (sidecars or {}).items()}
    if "award" in bundle.spaces:
        prototype = bundle.spaces["award"].embeddings
        columns = _embedding_columns(bundle.options.award_embedding.width)
        prototype_map = {
            str(row.award_id): np.asarray(
                [getattr(row, column) for column in columns], dtype=float
            )
            for row in prototype.itertuples()
        }
        for color in ("red", "blue"):
            for slot in (1, 2, 3):
                prefix = f"{color}_team_{slot}"
                vectors = np.full((len(out), len(columns)), np.nan, dtype=float)
                vector_sum = np.zeros((len(out), len(columns)), dtype=float)
                vector_count = np.zeros(len(out), dtype=int)
                for axis in out.columns:
                    marker = f"{prefix}_award_"
                    if not axis.startswith(marker):
                        continue
                    award_id = axis.removeprefix(marker)
                    if award_id not in prototype_map:
                        continue
                    mask = pd.to_numeric(out[axis], errors="coerce").fillna(0).to_numpy() > 0
                    vector_sum[mask] += prototype_map[award_id]
                    vector_count[mask] += 1
                positive = vector_count > 0
                vectors[positive] = vector_sum[positive] / vector_count[positive, None]
                norm = np.linalg.norm(vectors[positive], axis=1, keepdims=True)
                vectors[positive] /= np.maximum(norm, np.finfo(float).eps)
                for idx, column in enumerate(columns):
                    out[f"{prefix}_wm_award_{column}"] = vectors[:, idx]
    if "rank" in bundle.spaces:
        attached["world_rank"] = bundle.spaces["rank"].embeddings.copy()
    if "pick" in bundle.spaces:
        attached["world_pick"] = bundle.spaces["pick"].embeddings.copy()
    return out, attached


def paired_bootstrap_noninferiority(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    resamples: int = 2_000,
    seed: int = 2026,
) -> pd.DataFrame:
    return _supported_paired_bootstrap_noninferiority(
        baseline, candidate, resamples=resamples, seed=seed
    )
