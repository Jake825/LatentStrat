"""Shared, reproducible PyTorch training runtime utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

RESUME_SCHEMA_VERSION = 1
SCHEDULERS = ("none", "cosine", "one-cycle")


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    """Optimizer-loop behavior shared by all LatentStrat trainers."""

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

    def validate(self) -> TrainingRuntimeConfig:
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative.")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}.")
        if self.lr_eta_min < 0:
            raise ValueError("lr_eta_min must be non-negative.")
        if self.checkpoint_every_epochs < 0:
            raise ValueError("checkpoint_every_epochs must be non-negative.")
        if self.optimizer_log_interval <= 0:
            raise ValueError("optimizer_log_interval must be positive.")
        return self


@dataclass(frozen=True)
class OptimizerStepResult:
    optimizer_step: int
    preclip_grad_norm: float
    clipped: bool
    learning_rates: tuple[float, ...]
    elapsed_seconds: float


def stable_seed(seed: int, *parts: object) -> int:
    """Derive a process-stable PyTorch seed from a root seed and stream labels."""

    payload = json.dumps([int(seed), *map(str, parts)], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic kernels by default."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by DataLoader."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seeded_generator(seed: int, *parts: object) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(stable_seed(seed, *parts))
    return generator


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingRuntimeConfig,
    *,
    epochs: int,
    microbatches_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create an optimizer-step scheduler with accumulation-aware total steps."""

    config.validate()
    if config.scheduler == "none":
        return None
    steps_per_epoch = max(
        1, math.ceil(microbatches_per_epoch / config.gradient_accumulation_steps)
    )
    total_steps = max(1, int(epochs) * steps_per_epoch)
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.lr_eta_min
        )
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[float(group["lr"]) for group in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=config.one_cycle_pct_start,
        div_factor=config.one_cycle_div_factor,
        final_div_factor=config.one_cycle_final_div_factor,
    )


class OptimizationController:
    """Own backward/step ordering, accumulation, clipping, scheduling, and telemetry."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingRuntimeConfig,
        *,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: Any | None = None,
        tensorboard_writer: Any | None = None,
        tensorboard_prefix: str = "Optimization",
        optimizer_step: int = 0,
        after_step: Callable[[], None] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config.validate()
        self.scheduler = scheduler
        self.scaler = scaler
        self.tensorboard_writer = tensorboard_writer
        self.tensorboard_prefix = tensorboard_prefix.rstrip("/")
        self.optimizer_step = int(optimizer_step)
        self.after_step = after_step
        self.step_results: list[OptimizerStepResult] = []
        self.skipped_nonfinite_steps = 0
        self._parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        self.optimizer.zero_grad(set_to_none=True)

    def backward(
        self,
        loss: Tensor,
        *,
        microbatch_index: int,
        microbatch_count: int,
    ) -> OptimizerStepResult | None:
        """Accumulate one loss and step at the effective-batch boundary."""

        if not bool(torch.isfinite(loss).detach().cpu()):
            self.skipped_nonfinite_steps += 1
            raise FloatingPointError("Training loss is non-finite before backward().")
        accumulation = self.config.gradient_accumulation_steps
        window_start = (microbatch_index // accumulation) * accumulation
        window_size = min(accumulation, microbatch_count - window_start)
        scaled_loss = loss / max(window_size, 1)
        if self.scaler is None:
            scaled_loss.backward()
        else:
            self.scaler.scale(scaled_loss).backward()
        at_boundary = (
            (microbatch_index + 1) % accumulation == 0
            or microbatch_index + 1 == microbatch_count
        )
        return self.step() if at_boundary else None

    def step(self) -> OptimizerStepResult:
        started = time.perf_counter()
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._parameters,
            max_norm=(
                self.config.max_grad_norm
                if self.config.max_grad_norm > 0
                else float("inf")
            ),
            error_if_nonfinite=True,
        )
        grad_norm_value = float(grad_norm.detach().cpu())
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.after_step is not None:
            self.after_step()
        self.optimizer_step += 1
        learning_rates = tuple(float(group["lr"]) for group in self.optimizer.param_groups)
        result = OptimizerStepResult(
            optimizer_step=self.optimizer_step,
            preclip_grad_norm=grad_norm_value,
            clipped=(
                self.config.max_grad_norm > 0
                and grad_norm_value > self.config.max_grad_norm
            ),
            learning_rates=learning_rates,
            elapsed_seconds=time.perf_counter() - started,
        )
        self.step_results.append(result)
        self._log_step(result)
        self.optimizer.zero_grad(set_to_none=True)
        return result

    def _log_step(self, result: OptimizerStepResult) -> None:
        writer = self.tensorboard_writer
        if writer is None or result.optimizer_step % self.config.optimizer_log_interval:
            return
        writer.add_scalar(
            f"{self.tensorboard_prefix}/PreclipGradientNorm",
            result.preclip_grad_norm,
            result.optimizer_step,
        )
        writer.add_scalar(
            f"{self.tensorboard_prefix}/Clipped",
            float(result.clipped),
            result.optimizer_step,
        )
        for index, learning_rate in enumerate(result.learning_rates):
            writer.add_scalar(
                f"{self.tensorboard_prefix}/LearningRate/group_{index}",
                learning_rate,
                result.optimizer_step,
            )

    def epoch_summary(self, *, start_index: int = 0) -> dict[str, float | int]:
        results = self.step_results[start_index:]
        if not results:
            return {
                "optimizer_steps": 0,
                "preclip_grad_norm_mean": 0.0,
                "preclip_grad_norm_max": 0.0,
                "clipping_fraction": 0.0,
            }
        return {
            "optimizer_steps": len(results),
            "preclip_grad_norm_mean": float(
                np.mean([result.preclip_grad_norm for result in results])
            ),
            "preclip_grad_norm_max": float(
                np.max([result.preclip_grad_norm for result in results])
            ),
            "clipping_fraction": float(np.mean([result.clipped for result in results])),
        }


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def runtime_provenance(device: torch.device | str = "cpu") -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "executable": sys.executable,
    }


def collect_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "torch_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
        "numpy_random_state": {
            "bit_generator": numpy_state[0],
            "keys": torch.as_tensor(numpy_state[1].copy(), dtype=torch.int64),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    torch_state = payload.get("torch_rng_state")
    if isinstance(torch_state, Tensor):
        torch.set_rng_state(torch_state.cpu())
    python_state = payload.get("python_random_state")
    if isinstance(python_state, (tuple, list)):
        random.setstate(_nested_tuple(python_state))
    numpy_state = payload.get("numpy_random_state")
    if isinstance(numpy_state, Mapping):
        keys = numpy_state["keys"]
        keys_array = (
            keys.cpu().numpy().astype("uint32", copy=False)
            if isinstance(keys, Tensor)
            else np.asarray(keys, dtype="uint32")
        )
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                keys_array,
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_nested_tuple(item) for item in value)
    return value


def save_resume_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a local epoch-boundary resume checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    complete = dict(payload)
    complete["resume_schema_version"] = RESUME_SCHEMA_VERSION
    complete.setdefault("runtime_provenance", runtime_provenance())
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(complete, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Resume checkpoint must contain a dictionary.")
    if payload.get("resume_schema_version") != RESUME_SCHEMA_VERSION:
        raise ValueError("Unsupported resume checkpoint schema.")
    return payload


def validate_resume_checkpoint(
    payload: Mapping[str, Any],
    *,
    trainer: str,
    phase: str,
    config: Mapping[str, Any],
    source_fingerprint: str,
) -> None:
    expected = {
        "trainer": trainer,
        "phase": phase,
        "config_fingerprint": config_fingerprint(config),
        "source_fingerprint": source_fingerprint,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Resume checkpoint is incompatible: {mismatches}")


def resume_payload(
    *,
    trainer: str,
    phase: str,
    completed_epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    optimizer_step: int,
    history: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    source_fingerprint: str,
    scaler: Any | None = None,
    loader_generator: torch.Generator | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "trainer": trainer,
        "phase": phase,
        "completed_epoch": int(completed_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "optimizer_step": int(optimizer_step),
        "history": [dict(row) for row in history],
        "resolved_config": dict(config),
        "config_fingerprint": config_fingerprint(config),
        "source_fingerprint": source_fingerprint,
        "loader_generator_state": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
        "rng_state": collect_rng_state(),
        "extra_state": dict(extra_state or {}),
        "runtime_provenance": runtime_provenance(next(model.parameters()).device),
    }


def load_training_state(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: Any | None = None,
    loader_generator: torch.Generator | None = None,
) -> None:
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    saved_scheduler = payload.get("scheduler_state_dict")
    if scheduler is None and saved_scheduler is not None:
        raise ValueError("Resume checkpoint contains a scheduler but this run does not.")
    if scheduler is not None:
        if saved_scheduler is None:
            raise ValueError("Resume checkpoint is missing scheduler state.")
        scheduler.load_state_dict(saved_scheduler)
    saved_scaler = payload.get("scaler_state_dict")
    if scaler is not None and saved_scaler is not None:
        scaler.load_state_dict(saved_scaler)
    generator_state = payload.get("loader_generator_state")
    if loader_generator is not None and isinstance(generator_state, Tensor):
        loader_generator.set_state(generator_state)
    rng_state = payload.get("rng_state")
    if isinstance(rng_state, Mapping):
        restore_rng_state(rng_state)


def epoch_runtime_metrics(
    *,
    started_at: float,
    sample_count: int,
    optimizer_summary: Mapping[str, float | int],
) -> dict[str, float | int]:
    elapsed = max(time.perf_counter() - started_at, np.finfo(float).eps)
    return {
        **optimizer_summary,
        "epoch_seconds": float(elapsed),
        "samples_per_second": float(sample_count / elapsed),
    }


def runtime_config_dict(config: TrainingRuntimeConfig) -> dict[str, Any]:
    return asdict(config.validate())
