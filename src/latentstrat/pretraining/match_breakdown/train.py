"""Train and export the offline multi-season match-breakdown encoder."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import rankdata
from torch import Tensor

from latentstrat.artifacts import write_artifact_manifest
from latentstrat.pretraining.match_breakdown.corpus import (
    DEFAULT_EVENT_TYPES,
    FOC_EVENT_TYPE,
    REMOTE_EVENT_TYPE,
    MatchBreakdownCorpus,
)
from latentstrat.pretraining.match_breakdown.loss import (
    grouped_type_aware_loss,
    grouped_type_aware_report,
)
from latentstrat.pretraining.match_breakdown.model import MatchBreakdownAutoencoder
from latentstrat.pretraining.match_breakdown.parse import (
    AllianceBreakdownRow,
    extract_alliance_rows,
)
from latentstrat.pretraining.match_breakdown.rules import (
    LinearRule,
    audit_rules,
    load_rules_by_season,
    numeric_field_to_index,
    score_consistency_loss,
    score_consistency_report,
)
from latentstrat.pretraining.match_breakdown.vectorize import (
    SeasonSchema,
    discover_season_schema,
    union_schema_audit,
    vectorize_flat_row,
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
LATENT_DIM = 16
DEFAULT_RULE_DIR = str(Path(__file__).with_name("rules"))


@dataclass(frozen=True)
class MatchBreakdownTrainingOptions:
    artifact_version: str = "v1"
    start_season: int = 2015
    end_season: int = 2026
    epochs: int = 50
    seasons_per_step: int = 4
    rows_per_season: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 2026
    validation_fraction: float = 0.10
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    scheduler: str = "cosine"
    lr_eta_min: float = 1e-5
    one_cycle_pct_start: float = 0.3
    one_cycle_div_factor: float = 25.0
    one_cycle_final_div_factor: float = 10_000.0
    deterministic_algorithms: bool = False
    checkpoint_every_epochs: int = 1
    optimizer_log_interval: int = 10
    device: str = "cpu"
    include_foc: bool = False
    include_remote: bool = False
    reconstruction_weight: float = 1.0
    denoising_consistency_weight: float = 0.0
    winner_margin_rank_weight: float = 0.0
    score_consistency_weight: float = 0.0
    denoising_enabled: bool = False
    denoising_start_epoch: int = 5
    observed_source_dropout: float = 0.10
    reconstruct_from_corrupt: bool = False
    winner_ranking_enabled: bool = False
    winner_ranking_start_epoch: int = 10
    winner_ranking_margin: float = 0.05
    pairs_per_season: int = 32
    skip_ties: bool = True
    score_consistency_enabled: bool = False
    score_consistency_start_epoch: int = 15
    score_rule_dir: str = DEFAULT_RULE_DIR
    default_rule_tolerance: float = 1e-6
    min_rule_pass_rate: float = 0.995

    @property
    def derived_batch_size(self) -> int:
        return self.seasons_per_step * self.rows_per_season

    @property
    def event_types(self) -> tuple[int, ...]:
        event_types = set(DEFAULT_EVENT_TYPES)
        if self.include_foc:
            event_types.add(FOC_EVENT_TYPE)
        if self.include_remote:
            event_types.add(REMOTE_EVENT_TYPE)
        return tuple(sorted(event_types))

    @property
    def checkpoint_schema_version(self) -> int:
        return 2 if self.artifact_version == "v2" else 1

    @property
    def structured_objective_enabled(self) -> bool:
        return self.checkpoint_schema_version == 2


@dataclass(frozen=True)
class SeasonTensors:
    season: int
    rows: tuple[AllianceBreakdownRow, ...]
    values: Tensor
    masks: Tensor
    schema: SeasonSchema


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    alliance_features_path: Path
    embeddings_path: Path
    bundle: dict[str, Any]
    validation_report: dict[str, Any]


class _RowQueue:
    def __init__(self, indices: np.ndarray, rng: np.random.Generator) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = rng
        self.order = self.rng.permutation(self.indices)
        self.offset = 0

    def take(self, count: int) -> np.ndarray:
        if not len(self.indices):
            raise ValueError("Cannot sample an empty season row queue.")
        parts = []
        remaining = count
        while remaining > 0:
            if self.offset >= len(self.order):
                self.order = self.rng.permutation(self.indices)
                self.offset = 0
            size = min(remaining, len(self.order) - self.offset)
            parts.append(self.order[self.offset : self.offset + size])
            self.offset += size
            remaining -= size
        return np.concatenate(parts)


def load_match_breakdown_training_options(
    path: str | Path | None = None, *, overrides: dict[str, Any] | None = None
) -> MatchBreakdownTrainingOptions:
    """Load an explicit structured-objective config while retaining V1 defaults."""

    values: dict[str, Any] = {}
    if path is not None:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values["artifact_version"] = payload.get("artifact_version", "v1")
        sections = {
            "training": {
                "start_season": "start_season",
                "end_season": "end_season",
                "epochs": "epochs",
                "seasons_per_step": "seasons_per_step",
                "rows_per_season": "rows_per_season",
                "learning_rate": "learning_rate",
                "weight_decay": "weight_decay",
                "seed": "seed",
                "validation_fraction": "validation_fraction",
                "max_grad_norm": "max_grad_norm",
                "gradient_accumulation_steps": "gradient_accumulation_steps",
                "scheduler": "scheduler",
                "lr_eta_min": "lr_eta_min",
                "one_cycle_pct_start": "one_cycle_pct_start",
                "one_cycle_div_factor": "one_cycle_div_factor",
                "one_cycle_final_div_factor": "one_cycle_final_div_factor",
                "deterministic_algorithms": "deterministic_algorithms",
                "checkpoint_every_epochs": "checkpoint_every_epochs",
                "optimizer_log_interval": "optimizer_log_interval",
                "device": "device",
            },
            "loss": {
                "reconstruction_weight": "reconstruction_weight",
                "denoising_consistency_weight": "denoising_consistency_weight",
                "winner_margin_rank_weight": "winner_margin_rank_weight",
                "score_consistency_weight": "score_consistency_weight",
            },
            "denoising": {
                "enabled": "denoising_enabled",
                "start_epoch": "denoising_start_epoch",
                "observed_source_dropout": "observed_source_dropout",
                "reconstruct_from_corrupt": "reconstruct_from_corrupt",
            },
            "winner_ranking": {
                "enabled": "winner_ranking_enabled",
                "start_epoch": "winner_ranking_start_epoch",
                "margin": "winner_ranking_margin",
                "pairs_per_season": "pairs_per_season",
                "skip_ties": "skip_ties",
            },
            "score_consistency": {
                "enabled": "score_consistency_enabled",
                "start_epoch": "score_consistency_start_epoch",
                "rule_dir": "score_rule_dir",
                "default_tolerance": "default_rule_tolerance",
                "min_rule_pass_rate": "min_rule_pass_rate",
            },
        }
        for section, mapping in sections.items():
            raw_section = payload.get(section, {})
            unknown = sorted(set(raw_section) - set(mapping))
            if unknown:
                raise ValueError(f"Unsupported {section} config keys: {', '.join(unknown)}.")
            for source, target in mapping.items():
                if source in raw_section:
                    values[target] = raw_section[source]
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    options = replace(MatchBreakdownTrainingOptions(), **values)
    if options.artifact_version not in {"v1", "v2"}:
        raise ValueError("artifact_version must be 'v1' or 'v2'.")
    return options


def corrupt_observed_source_fields(
    values: Tensor,
    masks: Tensor,
    schema: SeasonSchema,
    *,
    dropout: float,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    """Hide whole observed source fields while preserving physical-zero semantics."""

    corrupted_values = values.clone()
    corrupted_masks = masks.clone()
    by_source: dict[str, list[int]] = {}
    for index, field in enumerate(schema.encoded_fields):
        by_source.setdefault(field.source_path, []).append(index)
    for row in range(len(values)):
        for field_indices in by_source.values():
            if torch.any(masks[row, field_indices] > 0) and rng.random() < dropout:
                corrupted_values[row, field_indices] = 0
                corrupted_masks[row, field_indices] = 0
    return corrupted_values, corrupted_masks


def _match_pair_indices(data: SeasonTensors, indices: np.ndarray) -> np.ndarray:
    selected = set(int(index) for index in indices)
    by_match: dict[str, dict[str, int]] = {}
    for index in selected:
        row = data.rows[index]
        by_match.setdefault(row.match_key, {})[row.alliance] = index
    pairs = [
        (alliances["red"], alliances["blue"])
        for _, alliances in sorted(by_match.items())
        if "red" in alliances and "blue" in alliances
    ]
    return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def winner_margin_ranking_loss(
    model: MatchBreakdownAutoencoder,
    season: int,
    data: SeasonTensors,
    pairs: np.ndarray,
    *,
    margin: float,
    skip_ties: bool,
    device: torch.device,
) -> Tensor:
    """Orient one scalar projection of the latent toward the posted match winner."""

    if not len(pairs):
        return next(model.parameters()).sum() * 0
    red_indices = pairs[:, 0]
    blue_indices = pairs[:, 1]
    red_latent = model.encode(
        season, data.values[red_indices].to(device), data.masks[red_indices].to(device)
    )
    blue_latent = model.encode(
        season, data.values[blue_indices].to(device), data.masks[blue_indices].to(device)
    )
    red_quality = model.quality(red_latent)
    blue_quality = model.quality(blue_latent)
    red_scores = torch.as_tensor(
        [data.rows[int(index)].alliance_score for index in red_indices],
        dtype=red_quality.dtype,
        device=device,
    )
    blue_scores = torch.as_tensor(
        [data.rows[int(index)].alliance_score for index in blue_indices],
        dtype=blue_quality.dtype,
        device=device,
    )
    valid = (
        red_scores != blue_scores
        if skip_ties
        else torch.ones_like(red_scores, dtype=torch.bool)
    )
    if not torch.any(valid):
        return red_quality.sum() * 0
    target = torch.where(
        red_scores > blue_scores,
        torch.ones_like(red_quality),
        -torch.ones_like(red_quality),
    )
    return F.margin_ranking_loss(
        red_quality[valid], blue_quality[valid], target[valid], margin=margin
    )


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _eligible_rows(
    matches: list[dict[str, Any]], options: MatchBreakdownTrainingOptions
) -> tuple[list[AllianceBreakdownRow], dict[str, int]]:
    rows = []
    audit = {
        "matches_loaded": len(matches),
        "matches_excluded_incomplete": 0,
        "matches_excluded_2021": 0,
        "alliance_rows": 0,
    }
    for match in matches:
        season = int(str(match.get("event_key", "0"))[:4] or 0)
        if season == 2021:
            audit["matches_excluded_2021"] += 1
            continue
        extracted = extract_alliance_rows(match)
        if not extracted:
            audit["matches_excluded_incomplete"] += 1
            continue
        rows.extend(extracted)
    audit["alliance_rows"] = len(rows)
    return rows, audit


def _season_tensors(rows: list[AllianceBreakdownRow]) -> dict[int, SeasonTensors]:
    by_season: dict[int, list[AllianceBreakdownRow]] = {}
    for row in rows:
        by_season.setdefault(row.season, []).append(row)
    tensors = {}
    for season, season_rows in sorted(by_season.items()):
        schema = discover_season_schema(season_rows, season)
        vectors = [
            vectorize_flat_row(row.flat_breakdown, schema.encoded_fields)
            for row in season_rows
        ]
        values = torch.as_tensor(np.stack([vector[0] for vector in vectors]), dtype=torch.float32)
        masks = torch.as_tensor(np.stack([vector[1] for vector in vectors]), dtype=torch.float32)
        tensors[season] = SeasonTensors(
            season=season,
            rows=tuple(season_rows),
            values=values,
            masks=masks,
            schema=schema,
        )
    if not tensors:
        raise ValueError("The corpus has no eligible match-breakdown alliance rows.")
    return tensors


def _split_indices(
    tensors: dict[int, SeasonTensors], *, fraction: float, seed: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    train = {}
    validation = {}
    for season, data in sorted(tensors.items()):
        match_keys = sorted({row.match_key for row in data.rows})
        shuffled = np.asarray(match_keys, dtype=object)[rng.permutation(len(match_keys))]
        validation_count = 0
        if len(match_keys) > 1:
            validation_count = max(1, int(math.ceil(len(match_keys) * fraction)))
            validation_count = min(validation_count, len(match_keys) - 1)
        validation_keys = set(shuffled[:validation_count])
        validation[season] = np.asarray(
            [index for index, row in enumerate(data.rows) if row.match_key in validation_keys],
            dtype=np.int64,
        )
        train[season] = np.asarray(
            [index for index, row in enumerate(data.rows) if row.match_key not in validation_keys],
            dtype=np.int64,
        )
    return train, validation


def _checkpoint_payload(
    model: MatchBreakdownAutoencoder,
    tensors: dict[int, SeasonTensors],
    options: MatchBreakdownTrainingOptions,
    *,
    provenance: str,
) -> dict[str, Any]:
    return {
        "schema_version": options.checkpoint_schema_version,
        "kind": "v6-lite-match-breakdown-shared-bottleneck",
        "provenance": provenance,
        "latent_dim": LATENT_DIM,
        "season_widths": {season: data.schema.width for season, data in tensors.items()},
        "schemas": {season: data.schema.to_dict() for season, data in tensors.items()},
        "options": asdict(options),
        "quality_head_contract": (
            "diagnostic-only" if model.quality_head_enabled else None
        ),
        "state_dict": model.state_dict(),
    }


def _resolve_rules(
    tensors: dict[int, SeasonTensors],
    indices: dict[int, np.ndarray],
    options: MatchBreakdownTrainingOptions,
    *,
    scope: str,
) -> tuple[dict[int, tuple[LinearRule, ...]], dict[str, Any]]:
    if not options.score_consistency_enabled:
        return {}, {"scope": scope, "enabled": False, "seasons": {}}
    rules_by_season, hashes = load_rules_by_season(
        options.score_rule_dir,
        sorted(tensors),
        default_tolerance=options.default_rule_tolerance,
        default_min_pass_rate=options.min_rule_pass_rate,
    )
    resolved = {}
    audit = {"scope": scope, "enabled": True, "seasons": {}}
    for season, data in sorted(tensors.items()):
        result = audit_rules(
            season=season,
            schema=data.schema,
            rows=data.rows,
            values=data.values,
            masks=data.masks,
            indices=indices[season],
            rules=rules_by_season[season],
            rule_file_hash=hashes[season],
            scope=scope,
        )
        resolved[season] = result.enabled
        audit["seasons"][str(season)] = result.audit
    return resolved, audit


def _train_model(
    tensors: dict[int, SeasonTensors],
    indices: dict[int, np.ndarray],
    options: MatchBreakdownTrainingOptions,
    *,
    seed_offset: int,
    phase: str,
    rules_by_season: dict[int, tuple[LinearRule, ...]] | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_output: str | Path | None = None,
    tensorboard_writer: Any | None = None,
) -> tuple[MatchBreakdownAutoencoder, list[dict[str, Any]]]:
    active_seasons = [season for season, rows in sorted(indices.items()) if len(rows)]
    if not active_seasons:
        raise ValueError(f"{phase} training split has no eligible rows.")
    runtime_config = TrainingRuntimeConfig(
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
    seed_everything(
        options.seed + seed_offset,
        deterministic_algorithms=options.deterministic_algorithms,
    )
    rng = np.random.default_rng(options.seed + seed_offset)
    device = _resolve_device(options.device)
    model = MatchBreakdownAutoencoder(
        {season: tensors[season].schema.width for season in tensors},
        LATENT_DIM,
        quality_head_enabled=options.structured_objective_enabled,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    total_rows = sum(len(rows) for rows in indices.values())
    steps_per_epoch = max(1, math.ceil(total_rows / options.derived_batch_size))
    scheduler = build_lr_scheduler(
        optimizer,
        runtime_config,
        epochs=options.epochs,
        microbatches_per_epoch=steps_per_epoch,
    )
    controller = OptimizationController(
        model,
        optimizer,
        runtime_config,
        scheduler=scheduler,
        tensorboard_writer=tensorboard_writer,
        tensorboard_prefix=f"Optimization/MatchBreakdown/{phase}",
    )
    resolved_config = {"options": asdict(options), "seed_offset": seed_offset}
    digest = hashlib.sha256()
    for season, data in sorted(tensors.items()):
        digest.update(str(season).encode("ascii"))
        digest.update(data.values.contiguous().numpy().tobytes())
        digest.update(data.masks.contiguous().numpy().tobytes())
        digest.update(np.asarray(indices[season], dtype=np.int64).tobytes())
    source_fingerprint = digest.hexdigest()
    history = []
    start_epoch = 1
    rules_by_season = rules_by_season or {}
    if resume_checkpoint is not None:
        payload = load_resume_checkpoint(resume_checkpoint)
        validate_resume_checkpoint(
            payload,
            trainer="match-breakdown",
            phase=phase,
            config=resolved_config,
            source_fingerprint=source_fingerprint,
        )
        load_training_state(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        history = list(payload.get("history", []))
        controller.optimizer_step = int(payload.get("optimizer_step", 0))
        start_epoch = int(payload["completed_epoch"]) + 1
    for epoch in range(start_epoch, options.epochs + 1):
        epoch_started = time.perf_counter()
        epoch_step_start = len(controller.step_results)
        model.train()
        queues = {season: _RowQueue(indices[season], rng) for season in active_seasons}
        pair_queues = {
            season: _RowQueue(_match_pair_indices(tensors[season], indices[season]), rng)
            for season in active_seasons
            if len(_match_pair_indices(tensors[season], indices[season]))
        }
        epoch_losses: dict[str, list[float]] = {
            "reconstruction_loss": [],
            "denoising_consistency_loss": [],
            "winner_margin_rank_loss": [],
            "score_consistency_loss": [],
            "total_loss": [],
        }
        epoch_grad_norms = []
        for step_index in range(steps_per_epoch):
            count = min(options.seasons_per_step, len(active_seasons))
            sampled_seasons = rng.choice(active_seasons, size=count, replace=False)
            season_losses = []
            for season_value in sampled_seasons:
                season = int(season_value)
                selected = queues[season].take(options.rows_per_season)
                values = tensors[season].values[selected].to(device)
                masks = tensors[season].masks[selected].to(device)
                reconstruction, latent = model(season, values, masks)
                reconstruction_loss = grouped_type_aware_loss(
                    reconstruction, values, masks, tensors[season].schema.encoded_fields
                )
                denoising_loss = reconstruction.sum() * 0
                if options.denoising_enabled and epoch >= options.denoising_start_epoch:
                    corrupted_values, corrupted_masks = corrupt_observed_source_fields(
                        tensors[season].values[selected],
                        tensors[season].masks[selected],
                        tensors[season].schema,
                        dropout=options.observed_source_dropout,
                        rng=rng,
                    )
                    corrupted_latent = model.encode(
                        season, corrupted_values.to(device), corrupted_masks.to(device)
                    )
                    denoising_loss = F.mse_loss(corrupted_latent, latent.detach())
                ranking_loss = reconstruction.sum() * 0
                if (
                    options.winner_ranking_enabled
                    and epoch >= options.winner_ranking_start_epoch
                    and season in pair_queues
                ):
                    pairs = pair_queues[season].take(options.pairs_per_season)
                    ranking_loss = winner_margin_ranking_loss(
                        model,
                        season,
                        tensors[season],
                        pairs,
                        margin=options.winner_ranking_margin,
                        skip_ties=options.skip_ties,
                        device=device,
                    )
                consistency_loss = reconstruction.sum() * 0
                if (
                    options.score_consistency_enabled
                    and epoch >= options.score_consistency_start_epoch
                ):
                    consistency_loss = score_consistency_loss(
                        reconstruction,
                        rules_by_season.get(season, ()),
                        numeric_field_to_index(tensors[season].schema),
                    )
                total = (
                    options.reconstruction_weight * reconstruction_loss
                    + options.denoising_consistency_weight * denoising_loss
                    + options.winner_margin_rank_weight * ranking_loss
                    + options.score_consistency_weight * consistency_loss
                )
                season_losses.append(total)
                for name, value in (
                    ("reconstruction_loss", reconstruction_loss),
                    ("denoising_consistency_loss", denoising_loss),
                    ("winner_margin_rank_loss", ranking_loss),
                    ("score_consistency_loss", consistency_loss),
                ):
                    epoch_losses[name].append(float(value.detach().cpu()))
            loss = torch.stack(season_losses).mean()
            step_result = controller.backward(
                loss,
                microbatch_index=step_index,
                microbatch_count=steps_per_epoch,
            )
            epoch_losses["total_loss"].append(float(loss.detach().cpu()))
            if step_result is not None:
                epoch_grad_norms.append(step_result.preclip_grad_norm)
        row = {
                "phase": phase,
                "epoch": epoch,
                "steps": steps_per_epoch,
                **{
                    name: float(np.mean(values)) if values else 0.0
                    for name, values in epoch_losses.items()
                },
                "train_loss": float(np.mean(epoch_losses["total_loss"])),
                "max_preclip_grad_norm": float(np.max(epoch_grad_norms)),
        }
        row.update(
            epoch_runtime_metrics(
                started_at=epoch_started,
                sample_count=total_rows,
                optimizer_summary=controller.epoch_summary(start_index=epoch_step_start),
            )
        )
        history.append(row)
        if tensorboard_writer is not None:
            for name, value in row.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    tensorboard_writer.add_scalar(
                        f"MatchBreakdown/{phase}/{name}", float(value), epoch
                    )
        if (
            resume_output is not None
            and runtime_config.checkpoint_every_epochs > 0
            and epoch % runtime_config.checkpoint_every_epochs == 0
        ):
            save_resume_checkpoint(
                resume_output,
                resume_payload(
                    trainer="match-breakdown",
                    phase=phase,
                    completed_epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_step=controller.optimizer_step,
                    history=history,
                    config=resolved_config,
                    source_fingerprint=source_fingerprint,
                ),
            )
    return model.cpu(), history


def _effective_rank(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    singular = np.linalg.svd(embeddings - embeddings.mean(axis=0), compute_uv=False)
    if not np.any(singular > 0):
        return 0.0
    probability = singular / singular.sum()
    return float(np.exp(-np.sum(probability * np.log(probability + np.finfo(float).eps))))


def _linear_probe_r2(embeddings: np.ndarray, target: np.ndarray) -> float | None:
    if len(embeddings) < 2 or np.allclose(target, target[0]):
        return None
    design = np.column_stack([np.ones(len(embeddings)), embeddings])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else None


def _nearest_neighbors(
    rows: list[AllianceBreakdownRow], embeddings: np.ndarray, *, limit: int = 10
) -> list[dict[str, Any]]:
    if len(embeddings) < 2:
        return []
    sample_indices = np.arange(min(len(embeddings), 2_000))
    reference = embeddings[sample_indices]
    result = []
    for index in sample_indices[:limit]:
        distances = np.linalg.norm(reference - embeddings[index], axis=1)
        distances[index] = np.inf
        neighbor = int(sample_indices[int(np.argmin(distances))])
        result.append(
            {
                "row_id": rows[index].row_id,
                "neighbor_row_id": rows[neighbor].row_id,
                "distance": float(np.min(distances)),
                "alliance_score_gap": float(
                    abs(float(rows[index].alliance_score) - float(rows[neighbor].alliance_score))
                ),
            }
        )
    return result


def _safe_correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool = False) -> float | None:
    if len(left) < 2 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    if ranked:
        left = rankdata(left)
        right = rankdata(right)
    return float(np.corrcoef(left, right)[0, 1])


def _foul_points(row: AllianceBreakdownRow) -> float:
    for field in ("foulPoints", "foul_points"):
        value = row.flat_breakdown.get(field)
        if value is not None:
            return float(value)
    return float("nan")


def _winner_ranking_report(
    model: MatchBreakdownAutoencoder, data: SeasonTensors, indices: np.ndarray
) -> dict[str, Any]:
    pairs = _match_pair_indices(data, indices)
    if not len(pairs):
        return {"pairs": 0, "non_tie_pairs": 0}
    red_indices = pairs[:, 0]
    blue_indices = pairs[:, 1]
    red_quality = model.quality(
        model.encode(data.season, data.values[red_indices], data.masks[red_indices])
    ).numpy()
    blue_quality = model.quality(
        model.encode(data.season, data.values[blue_indices], data.masks[blue_indices])
    ).numpy()
    red_score = np.asarray([data.rows[int(index)].alliance_score for index in red_indices])
    blue_score = np.asarray([data.rows[int(index)].alliance_score for index in blue_indices])
    valid = red_score != blue_score
    winner_gap = np.where(
        red_score > blue_score, red_quality - blue_quality, blue_quality - red_quality
    )
    winner_gap = winner_gap[valid]
    score_margin = np.abs(red_score - blue_score)[valid]
    bucket_report = {}
    if len(winner_gap):
        buckets = pd.qcut(score_margin, q=4, duplicates="drop")
        for bucket in sorted(set(str(value) for value in buckets)):
            selected = np.asarray([str(value) == bucket for value in buckets])
            bucket_report[bucket] = {
                "pairs": int(np.sum(selected)),
                "accuracy": float(np.mean(winner_gap[selected] > 0)),
                "mean_quality_gap": float(np.mean(winner_gap[selected])),
            }
    alliance_indices = pairs.reshape(-1)
    alliance_quality = model.quality(
        model.encode(data.season, data.values[alliance_indices], data.masks[alliance_indices])
    ).numpy()
    alliance_score = np.asarray(
        [data.rows[int(index)].alliance_score for index in alliance_indices]
    )
    foul_points = np.asarray([_foul_points(data.rows[int(index)]) for index in alliance_indices])
    foul_valid = np.isfinite(foul_points)
    return {
        "pairs": int(len(pairs)),
        "non_tie_pairs": int(np.sum(valid)),
        "accuracy": float(np.mean(winner_gap > 0)) if len(winner_gap) else None,
        "mean_quality_gap": float(np.mean(winner_gap)) if len(winner_gap) else None,
        "quality_gap_by_score_margin_quartile": bucket_report,
        "correlations": {
            "quality_total_score_pearson": _safe_correlation(alliance_quality, alliance_score),
            "quality_total_score_spearman": _safe_correlation(
                alliance_quality, alliance_score, ranked=True
            ),
            "quality_foul_points_pearson": (
                _safe_correlation(alliance_quality[foul_valid], foul_points[foul_valid])
                if np.any(foul_valid)
                else None
            ),
            "quality_foul_points_spearman": (
                _safe_correlation(
                    alliance_quality[foul_valid], foul_points[foul_valid], ranked=True
                )
                if np.any(foul_valid)
                else None
            ),
        },
    }


def _evaluate(
    model: MatchBreakdownAutoencoder,
    tensors: dict[int, SeasonTensors],
    validation: dict[int, np.ndarray],
    options: MatchBreakdownTrainingOptions,
    rules_by_season: dict[int, tuple[LinearRule, ...]] | None = None,
) -> dict[str, Any]:
    model.eval()
    report = {"per_season": {}, "all_validation_rows": 0}
    all_latent = []
    all_rows = []
    ranking_by_season = {}
    consistency_by_season = {}
    denoising_by_season = {}
    rules_by_season = rules_by_season or {}
    with torch.inference_mode():
        for season, indices in sorted(validation.items()):
            if not len(indices):
                continue
            data = tensors[season]
            values = data.values[indices]
            masks = data.masks[indices]
            reconstruction, latent = model(season, values, masks)
            latent_np = latent.numpy()
            rows = [data.rows[index] for index in indices]
            report["per_season"][str(season)] = {
                "validation_rows": len(indices),
                "schema_width": data.schema.width,
                "grouped_losses": grouped_type_aware_report(
                    reconstruction, values, masks, data.schema.encoded_fields
                ),
                "effective_rank": _effective_rank(latent_np),
                "linear_probes_r2": {
                    "alliance_score": _linear_probe_r2(
                        latent_np, np.asarray([row.alliance_score for row in rows], dtype=float)
                    ),
                    "opponent_score": _linear_probe_r2(
                        latent_np, np.asarray([row.opponent_score for row in rows], dtype=float)
                    ),
                },
            }
            if options.winner_ranking_enabled:
                ranking_by_season[str(season)] = _winner_ranking_report(model, data, indices)
            if options.score_consistency_enabled:
                consistency_by_season[str(season)] = score_consistency_report(
                    reconstruction,
                    rules_by_season.get(season, ()),
                    numeric_field_to_index(data.schema),
                )
            if options.denoising_enabled:
                corrupted_values, corrupted_masks = corrupt_observed_source_fields(
                    values,
                    masks,
                    data.schema,
                    dropout=options.observed_source_dropout,
                    rng=np.random.default_rng(options.seed + season),
                )
                corrupted_latent = model.encode(season, corrupted_values, corrupted_masks)
                denoising_by_season[str(season)] = float(F.mse_loss(corrupted_latent, latent).cpu())
            all_latent.append(latent_np)
            all_rows.extend(rows)
    if all_latent:
        embeddings = np.concatenate(all_latent)
        report["all_validation_rows"] = len(all_rows)
        report["effective_rank"] = _effective_rank(embeddings)
        report["nearest_neighbors"] = _nearest_neighbors(all_rows, embeddings)
    else:
        report["effective_rank"] = 0.0
        report["nearest_neighbors"] = []
    report["winner_ranking"] = {
        "enabled": options.winner_ranking_enabled,
        "margin": options.winner_ranking_margin,
        "quality_head_contract": "diagnostic-only",
        "by_season": ranking_by_season,
    }
    ranking_rows = [
        values for values in ranking_by_season.values() if values.get("non_tie_pairs", 0)
    ]
    ranking_pairs = sum(values["non_tie_pairs"] for values in ranking_rows)
    if ranking_pairs:
        report["winner_ranking"].update(
            {
                "non_tie_pairs": ranking_pairs,
                "accuracy": float(
                    sum(values["accuracy"] * values["non_tie_pairs"] for values in ranking_rows)
                    / ranking_pairs
                ),
                "mean_quality_gap": float(
                    sum(
                        values["mean_quality_gap"] * values["non_tie_pairs"]
                        for values in ranking_rows
                    )
                    / ranking_pairs
                ),
            }
        )
    report["score_consistency"] = {
        "enabled": options.score_consistency_enabled,
        "by_season": consistency_by_season,
    }
    consistency_rows = [
        values
        for values in consistency_by_season.values()
        if values.get("mean_abs_residual") is not None
    ]
    if consistency_rows:
        report["score_consistency"].update(
            {
                "mean_abs_residual": float(
                    np.mean([values["mean_abs_residual"] for values in consistency_rows])
                ),
                "max_abs_residual": float(
                    np.max([values["max_abs_residual"] for values in consistency_rows])
                ),
            }
        )
    report["denoising_consistency"] = {
        "enabled": options.denoising_enabled,
        "observed_source_dropout": options.observed_source_dropout,
        "by_season_mse": denoising_by_season,
    }
    if denoising_by_season:
        report["denoising_consistency"]["mean_mse"] = float(
            np.mean(list(denoising_by_season.values()))
        )
    return report


def _alliance_frame(rows: list[AllianceBreakdownRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "row_id": row.row_id,
                "season": row.season,
                "event_key": row.event_key,
                "match_key": row.match_key,
                "comp_level": row.comp_level,
                "set_number": row.set_number,
                "match_number": row.match_number,
                "alliance": row.alliance,
                "alliance_score": row.alliance_score,
                "opponent_score": row.opponent_score,
                "winning_alliance": row.winning_alliance,
                "flat_breakdown_json": json.dumps(row.flat_breakdown, sort_keys=True),
            }
            for row in rows
        ]
    )


def _production_embeddings(
    model: MatchBreakdownAutoencoder, tensors: dict[int, SeasonTensors]
) -> pd.DataFrame:
    model.eval()
    frames = []
    with torch.inference_mode():
        for season, data in sorted(tensors.items()):
            _, latent = model(season, data.values, data.masks)
            frame = _alliance_frame(list(data.rows)).drop(columns=["flat_breakdown_json"])
            for index in range(LATENT_DIM):
                frame[f"z_{index:03d}"] = latent[:, index].numpy()
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def train_match_breakdown_encoder(
    corpus_path: str | Path,
    output_dir: str | Path,
    alliance_features_path: str | Path,
    options: MatchBreakdownTrainingOptions | None = None,
    *,
    resume_checkpoint: str | Path | None = None,
    tensorboard_logdir: str | Path | None = None,
    tensorboard_run_name: str | None = None,
) -> TrainingResult:
    """Train holdout and production encoders, then write the offline artifact bundle."""

    options = options or MatchBreakdownTrainingOptions()
    if options.start_season > options.end_season:
        raise ValueError("start_season must be less than or equal to end_season.")
    if options.epochs <= 0 or options.seasons_per_step <= 0 or options.rows_per_season <= 0:
        raise ValueError("epochs, seasons_per_step, and rows_per_season must be positive.")
    if options.artifact_version not in {"v1", "v2"}:
        raise ValueError("artifact_version must be 'v1' or 'v2'.")
    if not 0 <= options.observed_source_dropout <= 1:
        raise ValueError("observed_source_dropout must be between 0 and 1.")
    if options.pairs_per_season <= 0:
        raise ValueError("pairs_per_season must be positive.")
    if options.reconstruct_from_corrupt:
        raise ValueError("V2 does not reconstruct decoder targets from corrupted inputs.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    if tensorboard_logdir is not None:
        from torch.utils.tensorboard import SummaryWriter

        run_name = tensorboard_run_name or f"match_breakdown_{int(time.time())}"
        writer = SummaryWriter(str(Path(tensorboard_logdir) / run_name))
    resume_phase = None
    if resume_checkpoint is not None:
        resume_phase = str(load_resume_checkpoint(resume_checkpoint).get("phase"))
    corpus = MatchBreakdownCorpus(corpus_path)
    matches = corpus.raw_matches(
        start_season=options.start_season,
        end_season=options.end_season,
        event_types=set(options.event_types),
    )
    rows, audit = _eligible_rows(matches, options)
    tensors = _season_tensors(rows)
    train_indices, validation_indices = _split_indices(
        tensors, fraction=options.validation_fraction, seed=options.seed
    )
    eval_rules, eval_rule_audit = _resolve_rules(
        tensors, train_indices, options, scope="eval-training-split"
    )
    eval_model, eval_history = _train_model(
        tensors,
        train_indices,
        options,
        seed_offset=0,
        phase="eval",
        rules_by_season=eval_rules,
        resume_checkpoint=(
            resume_checkpoint
            if resume_phase == "eval"
            else (
                out / "resume" / "eval" / "latest.ckpt"
                if resume_phase == "all_data"
                and (out / "resume" / "eval" / "latest.ckpt").exists()
                else None
            )
        ),
        resume_output=out / "resume" / "eval" / "latest.ckpt",
        tensorboard_writer=writer,
    )
    validation_report = _evaluate(
        eval_model, tensors, validation_indices, options, rules_by_season=eval_rules
    )
    all_indices = {
        season: np.arange(len(data.rows), dtype=np.int64) for season, data in tensors.items()
    }
    production_rules, production_rule_audit = _resolve_rules(
        tensors, all_indices, options, scope="all-eligible-rows"
    )
    production_model, full_history = _train_model(
        tensors,
        all_indices,
        options,
        seed_offset=1,
        phase="all_data",
        rules_by_season=production_rules,
        resume_checkpoint=resume_checkpoint if resume_phase == "all_data" else None,
        resume_output=out / "resume" / "all_data" / "latest.ckpt",
        tensorboard_writer=writer,
    )
    features_path = Path(alliance_features_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    _alliance_frame(rows).to_parquet(features_path, engine=PARQUET_ENGINE, index=False)
    embeddings = _production_embeddings(production_model, tensors)
    embeddings_path = out / "embeddings.parquet"
    embeddings.to_parquet(embeddings_path, engine=PARQUET_ENGINE, index=False)
    schema_payload = {
        "schema_version": options.checkpoint_schema_version,
        "season_schemas": {
            str(season): data.schema.to_dict() for season, data in sorted(tensors.items())
        },
    }
    _json_write(out / "schema.json", schema_payload)
    _json_write(out / "union_schema_audit.json", union_schema_audit(
        {season: data.schema for season, data in tensors.items()}
    ))
    optional_files = []
    if options.structured_objective_enabled:
        _json_write(
            out / "rule_audit.json",
            {
                "schema_version": 1,
                "eval_model": eval_rule_audit,
                "production_model": production_rule_audit,
            },
        )
        _json_write(out / "resolved_config.json", asdict(options))
        optional_files.extend(["rule_audit.json", "resolved_config.json"])
    torch.save(
        _checkpoint_payload(eval_model, tensors, options, provenance="holdout-evaluation-only"),
        out / "eval_model.pt",
    )
    torch.save(
        _checkpoint_payload(production_model, tensors, options, provenance="all-data-embeddings"),
        out / "model.pt",
    )
    history = pd.DataFrame([*eval_history, *full_history])
    history.to_csv(out / "training_history.csv", index=False)
    validation_report["audit"] = audit
    validation_report["metrics_checkpoint"] = "eval_model.pt"
    _json_write(out / "validation_report.json", validation_report)
    bundle = {
        "schema_version": options.checkpoint_schema_version,
        "kind": "v6-lite-match-breakdown-shared-bottleneck",
        "promotion_eligible": False,
        "promotion_note": (
            "All-years offline representation artifact; not valid as leakage-safe "
            "2026 walk-forward evidence."
        ),
        "git_commit": _git_head_commit(),
        "corpus": {"path": str(corpus.path), "sha256": _sha256_file(corpus.path)},
        "alliance_features": {
            "path": str(features_path),
            "sha256": _sha256_file(features_path),
        },
        "seasons_requested": [options.start_season, options.end_season],
        "event_types": list(options.event_types),
        "seasons_trained": sorted(tensors),
        "excluded_schema_seasons": (
            [2021] if options.start_season <= 2021 <= options.end_season else []
        ),
        "latent_dim": LATENT_DIM,
        "artifact_version": options.artifact_version,
        "quality_head_contract": (
            "diagnostic-only" if options.structured_objective_enabled else None
        ),
        "season_widths": {
            str(season): data.schema.width for season, data in sorted(tensors.items())
        },
        "derived_batch_size": options.derived_batch_size,
        "options": asdict(options),
        "score_consistency_rule_files": {
            season: payload["rule_file_sha256"]
            for season, payload in production_rule_audit.get("seasons", {}).items()
        },
        "audit": audit,
        "provenance": {
            "validation_metrics": "eval_model.pt",
            "exported_embeddings": "model.pt",
        },
        "files": {
            name: _sha256_file(out / name)
            for name in (
                "eval_model.pt",
                "model.pt",
                "embeddings.parquet",
                "schema.json",
                "union_schema_audit.json",
                "validation_report.json",
                "training_history.csv",
                *optional_files,
            )
        },
    }
    _json_write(out / "bundle.json", bundle)
    write_artifact_manifest(
        out,
        workflow="pretraining.match-breakdown.train",
        artifact_version=options.artifact_version,
        resolved_config=asdict(options),
        source_files={"corpus": corpus.path, "alliance_features": features_path},
        promotion_eligible=False,
        promotion_note=bundle["promotion_note"],
    )
    if writer is not None:
        writer.flush()
        writer.close()
    return TrainingResult(
        output_dir=out,
        alliance_features_path=features_path,
        embeddings_path=embeddings_path,
        bundle=bundle,
        validation_report=validation_report,
    )
