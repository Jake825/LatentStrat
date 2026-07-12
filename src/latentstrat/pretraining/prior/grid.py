"""Latent-dimension elbow experiments for the V5.6 prior distiller."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from latentstrat.artifacts import write_artifact_manifest
from latentstrat.config import PriorOpts
from latentstrat.pretraining.prior.features import (
    CULTURE_TARGET_COLUMNS,
    NORM_EPA_OBSERVED_COLUMNS,
    NORM_EPA_TARGET_COLUMNS,
    read_prior_feature_table,
)
from latentstrat.pretraining.prior.model import TeamPriorDistiller, prior_distillation_losses
from latentstrat.training_runtime import (
    OptimizationController,
    TrainingRuntimeConfig,
    build_lr_scheduler,
    epoch_runtime_metrics,
    load_resume_checkpoint,
    load_training_state,
    resume_payload,
    save_resume_checkpoint,
    seed_dataloader_worker,
    seed_everything,
    validate_resume_checkpoint,
)


@dataclass(frozen=True)
class PriorGridSplit:
    train_mask: np.ndarray
    validation_mask: np.ndarray


@dataclass
class PriorGridExperimentResult:
    results: pd.DataFrame
    history: pd.DataFrame
    failures: pd.DataFrame
    output_dir: Path


class PriorGridDataset(Dataset):
    """Tensor-backed prior feature table for latent elbow experiments."""

    def __init__(self, table: pd.DataFrame, *, llm_dim: int = 256) -> None:
        if table.empty:
            raise ValueError("Prior grid feature table is empty.")
        if "team_number" not in table.columns:
            raise KeyError("Prior grid features are missing team_number.")
        if "openai_narrative_vector" not in table.columns:
            raise KeyError("Prior grid features are missing openai_narrative_vector.")
        missing_norm_epa = [
            column
            for column in (*NORM_EPA_TARGET_COLUMNS, *NORM_EPA_OBSERVED_COLUMNS)
            if column not in table.columns
        ]
        if missing_norm_epa:
            raise KeyError(
                "Prior grid features are missing V5.6.4 normalized EPA trajectory "
                f"columns; rebuild prior features. Missing: {missing_norm_epa}."
            )
        missing_culture = [
            column for column in CULTURE_TARGET_COLUMNS if column not in table.columns
        ]
        if missing_culture:
            raise KeyError(
                "Prior grid features are missing V5.6.4 cultural target columns; "
                f"rebuild prior features. Missing: {missing_culture}."
            )
        team_numbers = [int(value) for value in table["team_number"].tolist()]
        if any(value < 0 for value in team_numbers):
            raise ValueError("Prior grid team_number values must be non-negative.")
        vectors = [
            np.asarray(value, dtype=np.float32)
            for value in table["openai_narrative_vector"].tolist()
        ]
        bad_widths = sorted(
            {int(vector.shape[0]) for vector in vectors if vector.shape[0] != llm_dim}
        )
        if bad_widths:
            raise ValueError(f"Expected prior vectors of width {llm_dim}, saw {bad_widths}.")
        self.team_numbers = torch.as_tensor(team_numbers, dtype=torch.long)
        self.vectors = torch.as_tensor(np.stack(vectors, axis=0), dtype=torch.float32)
        norm_epa = table[list(NORM_EPA_TARGET_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        norm_epa_observed = table[list(NORM_EPA_OBSERVED_COLUMNS)].astype(bool)
        observed_count = int(norm_epa_observed.to_numpy(dtype=bool).sum())
        if observed_count == 0:
            raise ValueError(
                "Prior grid features have zero observed normalized EPA trajectory values; "
                "rebuild V5.6.4 prior features with Statbotics normalized EPA from `epa.norm`."
            )
        observed_values = norm_epa.to_numpy(dtype=np.float32)[
            norm_epa_observed.to_numpy(dtype=bool)
        ]
        if observed_values.size and not np.isfinite(observed_values).all():
            raise ValueError("Observed prior grid normalized EPA values must be finite.")
        culture = table[list(CULTURE_TARGET_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(culture.to_numpy(dtype=np.float32)).all():
            raise ValueError("Prior grid cultural target values must be finite.")
        self.target_norm_epa = torch.as_tensor(norm_epa.to_numpy(dtype=np.float32, copy=True))
        self.norm_epa_observed = torch.as_tensor(
            norm_epa_observed.to_numpy(dtype=bool, copy=True),
            dtype=torch.bool,
        )
        self.target_culture = torch.as_tensor(culture.to_numpy(dtype=np.float32, copy=True))
        self.max_team_number = int(max(team_numbers))
        self.llm_dim = int(llm_dim)

    def __len__(self) -> int:
        return int(self.team_numbers.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.team_numbers[index],
            self.vectors[index],
            self.target_norm_epa[index],
            self.norm_epa_observed[index],
            self.target_culture[index],
        )


def resolve_grid_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_prior_grid_split(
    table: pd.DataFrame,
    *,
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> PriorGridSplit:
    if len(table) < 2:
        raise ValueError("At least two prior rows are required for a grid split.")
    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be > 0 and < 1.")
    rng = np.random.default_rng(seed)
    train_mask = np.ones(len(table), dtype=bool)
    validation_mask = np.zeros(len(table), dtype=bool)
    groups = (
        table["archetype"].astype(str)
        if "archetype" in table.columns
        else pd.Series("all", index=table.index)
    )
    ghost_indices = (
        np.flatnonzero(table["team_number"].astype(int).to_numpy() == 0)
        if "team_number" in table.columns
        else np.array([], dtype=int)
    )
    for _, index_values in groups.groupby(groups).groups.items():
        indices = np.asarray(list(index_values), dtype=int)
        if len(indices) <= 1:
            continue
        shuffled = rng.permutation(indices)
        holdout = max(1, int(round(len(indices) * validation_fraction)))
        holdout = min(holdout, len(indices) - 1)
        validation_mask[shuffled[:holdout]] = True
    if len(ghost_indices):
        validation_mask[ghost_indices] = False
    if not np.any(validation_mask):
        candidates = np.setdiff1d(np.arange(len(table)), ghost_indices, assume_unique=False)
        if len(candidates) == 0:
            raise ValueError("At least one non-ghost prior row is required for validation.")
        shuffled = rng.permutation(candidates)
        validation_mask[shuffled[0]] = True
    train_mask[validation_mask] = False
    if len(ghost_indices):
        train_mask[ghost_indices] = True
    if not np.any(train_mask):
        raise ValueError("Grid split produced no training rows.")
    return PriorGridSplit(train_mask=train_mask, validation_mask=validation_mask)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _estimated_memory(parameter_count: int) -> dict[str, float]:
    mib = 1024 * 1024
    model_mb = parameter_count * 4 / mib
    optimizer_mb = parameter_count * 8 / mib
    return {
        "estimated_model_mb": model_mb,
        "estimated_optimizer_mb": optimizer_mb,
        "estimated_training_state_mb": model_mb + optimizer_mb,
    }


def _loader(
    dataset: PriorGridDataset,
    mask: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = np.flatnonzero(mask)
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_dataloader_worker,
    )


def _batch_metrics(
    model: TeamPriorDistiller, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_openai_mse = 0.0
    total_norm_epa_mse = 0.0
    total_culture_mse = 0.0
    total_rows = 0
    with torch.inference_mode():
        for team_numbers, vectors, target_norm_epa, norm_epa_observed, target_culture in loader:
            team_numbers = team_numbers.to(device)
            vectors = vectors.to(device)
            target_norm_epa = target_norm_epa.to(device)
            norm_epa_observed = norm_epa_observed.to(device)
            target_culture = target_culture.to(device)
            losses = prior_distillation_losses(
                model,
                team_numbers,
                vectors,
                target_norm_epa,
                norm_epa_observed,
                target_culture,
            )
            batch_rows = int(vectors.shape[0])
            total_loss += float(losses.total.detach().cpu()) * batch_rows
            total_openai_mse += float(losses.openai_mse.detach().cpu()) * batch_rows
            total_norm_epa_mse += float(losses.norm_epa_mse.detach().cpu()) * batch_rows
            total_culture_mse += float(losses.culture_mse.detach().cpu()) * batch_rows
            total_rows += batch_rows
    if was_training:
        model.train()
    return {
        "loss": total_loss / max(total_rows, 1),
        "openai_mse": total_openai_mse / max(total_rows, 1),
        "norm_epa_mse": total_norm_epa_mse / max(total_rows, 1),
        "culture_mse": total_culture_mse / max(total_rows, 1),
    }


def _epochs_to_converge(history: pd.DataFrame) -> int | float:
    if history.empty or "validation_loss" not in history.columns:
        return np.nan
    best = float(history["validation_loss"].min())
    threshold = best * 1.01 if best > 0 else best + np.finfo(float).eps
    hits = history[history["validation_loss"] <= threshold]
    return int(hits["epoch"].iloc[0]) if not hits.empty else np.nan


def train_prior_grid_run(
    dataset: PriorGridDataset,
    split: PriorGridSplit,
    *,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    runtime_config: TrainingRuntimeConfig | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_output: str | Path | None = None,
    tensorboard_writer: Any | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    runtime_config = runtime_config or TrainingRuntimeConfig()
    seed_everything(seed, deterministic_algorithms=runtime_config.deterministic_algorithms)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    opts = PriorOpts(
        max_team_number=dataset.max_team_number,
        latent_dim=int(latent_dim),
        llm_dim=dataset.llm_dim,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        epochs=int(epochs),
        random_seed=int(seed),
        device=str(device),
    )
    model = TeamPriorDistiller(opts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    train_loader = _loader(
        dataset,
        split.train_mask,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _loader(
        dataset,
        split.validation_mask,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        runtime_config,
        epochs=epochs,
        microbatches_per_epoch=len(train_loader),
    )
    controller = OptimizationController(
        model,
        optimizer,
        runtime_config,
        scheduler=scheduler,
        tensorboard_writer=tensorboard_writer,
        tensorboard_prefix=f"Optimization/PriorGrid/{latent_dim}",
    )
    resolved_config = {
        **opts.model_dump(mode="json"),
        "runtime": runtime_config.__dict__,
    }
    digest = hashlib.sha256()
    for tensor in (
        dataset.team_numbers,
        dataset.vectors,
        dataset.target_norm_epa,
        dataset.norm_epa_observed,
        dataset.target_culture,
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(np.asarray(split.train_mask, dtype=np.bool_).tobytes())
    digest.update(np.asarray(split.validation_mask, dtype=np.bool_).tobytes())
    source_fingerprint = digest.hexdigest()
    start = time.perf_counter()
    history_rows = []
    start_epoch = 1
    if resume_checkpoint is not None:
        payload = load_resume_checkpoint(resume_checkpoint)
        validate_resume_checkpoint(
            payload,
            trainer="prior-grid",
            phase=str(latent_dim),
            config=resolved_config,
            source_fingerprint=source_fingerprint,
        )
        load_training_state(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader_generator=train_loader.generator,
        )
        history_rows = list(payload.get("history", []))
        controller.optimizer_step = int(payload.get("optimizer_step", 0))
        start_epoch = int(payload["completed_epoch"]) + 1
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        epoch_step_start = len(controller.step_results)
        model.train()
        total_loss = 0.0
        total_openai_mse = 0.0
        total_norm_epa_mse = 0.0
        total_culture_mse = 0.0
        total_rows = 0
        for batch_index, (
            team_numbers,
            vectors,
            target_norm_epa,
            norm_epa_observed,
            target_culture,
        ) in enumerate(train_loader):
            team_numbers = team_numbers.to(device)
            vectors = vectors.to(device)
            target_norm_epa = target_norm_epa.to(device)
            norm_epa_observed = norm_epa_observed.to(device)
            target_culture = target_culture.to(device)
            losses = prior_distillation_losses(
                model,
                team_numbers,
                vectors,
                target_norm_epa,
                norm_epa_observed,
                target_culture,
            )
            controller.backward(
                losses.total,
                microbatch_index=batch_index,
                microbatch_count=len(train_loader),
            )
            batch_rows = int(vectors.shape[0])
            total_loss += float(losses.total.detach().cpu()) * batch_rows
            total_openai_mse += float(losses.openai_mse.detach().cpu()) * batch_rows
            total_norm_epa_mse += float(losses.norm_epa_mse.detach().cpu()) * batch_rows
            total_culture_mse += float(losses.culture_mse.detach().cpu()) * batch_rows
            total_rows += batch_rows
        train_loss = total_loss / max(total_rows, 1)
        validation = _batch_metrics(model, validation_loader, device)
        row = {
            "latent_dim": latent_dim,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_openai_mse": total_openai_mse / max(total_rows, 1),
            "train_norm_epa_mse": total_norm_epa_mse / max(total_rows, 1),
            "train_culture_mse": total_culture_mse / max(total_rows, 1),
            "validation_loss": validation["loss"],
            "validation_openai_mse": validation["openai_mse"],
            "validation_norm_epa_mse": validation["norm_epa_mse"],
            "validation_culture_mse": validation["culture_mse"],
            "log_var_openai": float(model.log_var_openai.detach().cpu()),
            "log_var_epa": float(model.log_var_epa.detach().cpu()),
            "log_var_culture_mean": float(model.log_var_culture.detach().cpu().mean()),
        }
        row.update(
            epoch_runtime_metrics(
                started_at=epoch_started,
                sample_count=total_rows,
                optimizer_summary=controller.epoch_summary(start_index=epoch_step_start),
            )
        )
        history_rows.append(row)
        if tensorboard_writer is not None:
            for name, value in row.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    tensorboard_writer.add_scalar(
                        f"PriorGrid/{latent_dim}/{name}", float(value), epoch
                    )
        if (
            resume_output is not None
            and runtime_config.checkpoint_every_epochs > 0
            and epoch % runtime_config.checkpoint_every_epochs == 0
        ):
            save_resume_checkpoint(
                resume_output,
                resume_payload(
                    trainer="prior-grid",
                    phase=str(latent_dim),
                    completed_epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_step=controller.optimizer_step,
                    history=history_rows,
                    config=resolved_config,
                    source_fingerprint=source_fingerprint,
                    loader_generator=train_loader.generator,
                ),
            )
    history = pd.DataFrame(history_rows)
    elapsed = time.perf_counter() - start
    parameter_count = trainable_parameter_count(model)
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else np.nan
    )
    memory = _estimated_memory(parameter_count)
    result = {
        "latent_dim": latent_dim,
        "status": "ok",
        "epochs": epochs,
        "train_rows": int(np.sum(split.train_mask)),
        "validation_rows": int(np.sum(split.validation_mask)),
        "final_train_loss": float(history["train_loss"].iloc[-1]),
        "final_validation_loss": float(history["validation_loss"].iloc[-1]),
        "best_validation_loss": float(history["validation_loss"].min()),
        "final_train_mse": float(history["train_loss"].iloc[-1]),
        "final_validation_mse": float(history["validation_loss"].iloc[-1]),
        "best_validation_mse": float(history["validation_loss"].min()),
        "final_train_openai_mse": float(history["train_openai_mse"].iloc[-1]),
        "final_validation_openai_mse": float(history["validation_openai_mse"].iloc[-1]),
        "best_validation_openai_mse": float(history["validation_openai_mse"].min()),
        "final_train_norm_epa_mse": float(history["train_norm_epa_mse"].iloc[-1]),
        "final_validation_norm_epa_mse": float(history["validation_norm_epa_mse"].iloc[-1]),
        "best_validation_norm_epa_mse": float(history["validation_norm_epa_mse"].min()),
        "final_train_culture_mse": float(history["train_culture_mse"].iloc[-1]),
        "final_validation_culture_mse": float(history["validation_culture_mse"].iloc[-1]),
        "best_validation_culture_mse": float(history["validation_culture_mse"].min()),
        "final_log_var_openai": float(history["log_var_openai"].iloc[-1]),
        "final_log_var_epa": float(history["log_var_epa"].iloc[-1]),
        "final_log_var_culture_mean": float(history["log_var_culture_mean"].iloc[-1]),
        "epochs_to_converge": _epochs_to_converge(history),
        "trainable_parameters": parameter_count,
        "elapsed_seconds": elapsed,
        "peak_vram_mb": peak_vram,
        **memory,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, history


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _failure_row(latent_dim: int, exc: Exception) -> dict[str, Any]:
    status = "oom" if _is_oom(exc) else "failed"
    return {
        "latent_dim": latent_dim,
        "status": status,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _write_plots(results: pd.DataFrame, output_dir: Path) -> None:
    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = ok.sort_values("latent_dim")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        ordered["latent_dim"],
        ordered["best_validation_mse"],
        marker="o",
        color="#2563eb",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Best validation MSE")
    ax.set_title("V5.6 Prior Latent-Dimension Elbow")
    fig.tight_layout()
    fig.savefig(output_dir / "prior_elbow_curve.png", dpi=160)
    plt.close(fig)


def run_prior_grid(
    features_path: str | Path,
    output_dir: str | Path,
    *,
    latent_dims: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256),
    epochs: int = 500,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 2026,
    device: str = "cpu",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    scheduler: str = "cosine",
    lr_eta_min: float = 1e-5,
    deterministic_algorithms: bool = False,
    checkpoint_every_epochs: int = 1,
    resume_checkpoint: str | Path | None = None,
    tensorboard_logdir: str | Path | None = None,
    tensorboard_run_name: str | None = None,
) -> PriorGridExperimentResult:
    table = read_prior_feature_table(features_path)
    dataset = PriorGridDataset(table)
    split = make_prior_grid_split(table, validation_fraction=validation_fraction, seed=seed)
    torch_device = resolve_grid_device(device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    if tensorboard_logdir is not None:
        from torch.utils.tensorboard import SummaryWriter

        run_name = tensorboard_run_name or f"prior_grid_{int(time.time())}"
        writer = SummaryWriter(str(Path(tensorboard_logdir) / run_name))
    result_rows: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []
    runtime_config = TrainingRuntimeConfig(
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
        lr_eta_min=lr_eta_min,
        deterministic_algorithms=deterministic_algorithms,
        checkpoint_every_epochs=checkpoint_every_epochs,
    )
    resume_phase = None
    if resume_checkpoint is not None:
        resume_phase = str(load_resume_checkpoint(resume_checkpoint).get("phase"))
    for latent_dim in latent_dims:
        try:
            result, history = train_prior_grid_run(
                dataset,
                split,
                latent_dim=int(latent_dim),
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                seed=int(seed),
                device=torch_device,
                runtime_config=runtime_config,
                resume_checkpoint=(resume_checkpoint if resume_phase == str(latent_dim) else None),
                resume_output=output / "resume" / str(latent_dim) / "latest.ckpt",
                tensorboard_writer=writer,
            )
            result_rows.append(result)
            history_frames.append(history)
        except RuntimeError as exc:
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
            failure = _failure_row(int(latent_dim), exc)
            failure_rows.append(failure)
            result_rows.append(
                {
                    **failure,
                    "epochs": epochs,
                    "train_rows": int(np.sum(split.train_mask)),
                    "validation_rows": int(np.sum(split.validation_mask)),
                    "final_train_loss": np.nan,
                    "final_validation_loss": np.nan,
                    "best_validation_loss": np.nan,
                    "final_train_mse": np.nan,
                    "final_validation_mse": np.nan,
                    "best_validation_mse": np.nan,
                    "final_train_openai_mse": np.nan,
                    "final_validation_openai_mse": np.nan,
                    "best_validation_openai_mse": np.nan,
                    "final_train_norm_epa_mse": np.nan,
                    "final_validation_norm_epa_mse": np.nan,
                    "best_validation_norm_epa_mse": np.nan,
                    "final_train_culture_mse": np.nan,
                    "final_validation_culture_mse": np.nan,
                    "best_validation_culture_mse": np.nan,
                    "final_log_var_openai": np.nan,
                    "final_log_var_epa": np.nan,
                    "final_log_var_culture_mean": np.nan,
                    "epochs_to_converge": np.nan,
                    "trainable_parameters": np.nan,
                    "elapsed_seconds": np.nan,
                    "peak_vram_mb": np.nan,
                    **_estimated_memory(0),
                }
            )
    results = pd.DataFrame(result_rows)
    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    failures = pd.DataFrame(
        failure_rows,
        columns=["latent_dim", "status", "error_type", "error_message"],
    )
    results.to_csv(output / "prior_grid_results.csv", index=False)
    history.to_csv(output / "prior_grid_history.csv", index=False)
    failures.to_csv(output / "prior_grid_failures.csv", index=False)
    _write_plots(results, output)
    write_artifact_manifest(
        output,
        workflow="pretraining.prior.grid",
        artifact_version="v6.1",
        resolved_config={
            "latent_dims": list(latent_dims),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_fraction": validation_fraction,
            "seed": seed,
            "device": device,
        },
        source_files={"features": features_path},
        promotion_note="Prior bottleneck selection experiment.",
    )
    if writer is not None:
        writer.flush()
        writer.close()
    return PriorGridExperimentResult(
        results=results,
        history=history,
        failures=failures,
        output_dir=output,
    )
