"""Train and export the offline multi-season match-breakdown encoder."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from latentstrat.world_model.match_breakdown.corpus import (
    DEFAULT_EVENT_TYPES,
    FOC_EVENT_TYPE,
    REMOTE_EVENT_TYPE,
    MatchBreakdownCorpus,
)
from latentstrat.world_model.match_breakdown.loss import (
    grouped_type_aware_loss,
    grouped_type_aware_report,
)
from latentstrat.world_model.match_breakdown.model import MatchBreakdownAutoencoder
from latentstrat.world_model.match_breakdown.parse import (
    AllianceBreakdownRow,
    extract_alliance_rows,
)
from latentstrat.world_model.match_breakdown.vectorize import (
    SeasonSchema,
    discover_season_schema,
    union_schema_audit,
    vectorize_flat_row,
)

PARQUET_ENGINE = "pyarrow"
LATENT_DIM = 16


@dataclass(frozen=True)
class MatchBreakdownTrainingOptions:
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
    device: str = "auto"
    include_foc: bool = False
    include_remote: bool = False

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
        "schema_version": 1,
        "kind": "v6-lite-match-breakdown-shared-bottleneck",
        "provenance": provenance,
        "latent_dim": LATENT_DIM,
        "season_widths": {season: data.schema.width for season, data in tensors.items()},
        "schemas": {season: data.schema.to_dict() for season, data in tensors.items()},
        "options": asdict(options),
        "state_dict": model.state_dict(),
    }


def _train_model(
    tensors: dict[int, SeasonTensors],
    indices: dict[int, np.ndarray],
    options: MatchBreakdownTrainingOptions,
    *,
    seed_offset: int,
    phase: str,
) -> tuple[MatchBreakdownAutoencoder, list[dict[str, Any]]]:
    active_seasons = [season for season, rows in sorted(indices.items()) if len(rows)]
    if not active_seasons:
        raise ValueError(f"{phase} training split has no eligible rows.")
    torch.manual_seed(options.seed + seed_offset)
    rng = np.random.default_rng(options.seed + seed_offset)
    device = _resolve_device(options.device)
    model = MatchBreakdownAutoencoder(
        {season: tensors[season].schema.width for season in tensors}, LATENT_DIM
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    total_rows = sum(len(rows) for rows in indices.values())
    steps_per_epoch = max(1, math.ceil(total_rows / options.derived_batch_size))
    history = []
    for epoch in range(1, options.epochs + 1):
        model.train()
        queues = {season: _RowQueue(indices[season], rng) for season in active_seasons}
        epoch_losses = []
        epoch_grad_norms = []
        for _ in range(steps_per_epoch):
            count = min(options.seasons_per_step, len(active_seasons))
            sampled_seasons = rng.choice(active_seasons, size=count, replace=False)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for season_value in sampled_seasons:
                season = int(season_value)
                selected = queues[season].take(options.rows_per_season)
                values = tensors[season].values[selected].to(device)
                masks = tensors[season].masks[selected].to(device)
                reconstruction, _ = model(season, values, masks)
                losses.append(
                    grouped_type_aware_loss(
                        reconstruction, values, masks, tensors[season].schema.encoded_fields
                    )
                )
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=options.max_grad_norm
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_grad_norms.append(float(grad_norm.detach().cpu()))
        history.append(
            {
                "phase": phase,
                "epoch": epoch,
                "steps": steps_per_epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "max_preclip_grad_norm": float(np.max(epoch_grad_norms)),
            }
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
            }
        )
    return result


def _evaluate(
    model: MatchBreakdownAutoencoder,
    tensors: dict[int, SeasonTensors],
    validation: dict[int, np.ndarray],
) -> dict[str, Any]:
    model.eval()
    report = {"per_season": {}, "all_validation_rows": 0}
    all_latent = []
    all_rows = []
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
) -> TrainingResult:
    """Train holdout and production encoders, then write the offline artifact bundle."""

    options = options or MatchBreakdownTrainingOptions()
    if options.start_season > options.end_season:
        raise ValueError("start_season must be less than or equal to end_season.")
    if options.epochs <= 0 or options.seasons_per_step <= 0 or options.rows_per_season <= 0:
        raise ValueError("epochs, seasons_per_step, and rows_per_season must be positive.")
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
    eval_model, eval_history = _train_model(
        tensors, train_indices, options, seed_offset=0, phase="eval"
    )
    validation_report = _evaluate(eval_model, tensors, validation_indices)
    all_indices = {
        season: np.arange(len(data.rows), dtype=np.int64) for season, data in tensors.items()
    }
    production_model, full_history = _train_model(
        tensors, all_indices, options, seed_offset=1, phase="all_data"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features_path = Path(alliance_features_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    _alliance_frame(rows).to_parquet(features_path, engine=PARQUET_ENGINE, index=False)
    embeddings = _production_embeddings(production_model, tensors)
    embeddings_path = out / "embeddings.parquet"
    embeddings.to_parquet(embeddings_path, engine=PARQUET_ENGINE, index=False)
    schema_payload = {
        "schema_version": 1,
        "season_schemas": {
            str(season): data.schema.to_dict() for season, data in sorted(tensors.items())
        },
    }
    _json_write(out / "schema.json", schema_payload)
    _json_write(out / "union_schema_audit.json", union_schema_audit(
        {season: data.schema for season, data in tensors.items()}
    ))
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
        "schema_version": 1,
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
        "season_widths": {
            str(season): data.schema.width for season, data in sorted(tensors.items())
        },
        "derived_batch_size": options.derived_batch_size,
        "options": asdict(options),
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
            )
        },
    }
    _json_write(out / "bundle.json", bundle)
    return TrainingResult(
        output_dir=out,
        alliance_features_path=features_path,
        embeddings_path=embeddings_path,
        bundle=bundle,
        validation_report=validation_report,
    )
