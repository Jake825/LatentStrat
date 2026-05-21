"""Training utilities for LatentStrat's native PyTorch workflow."""

from __future__ import annotations

import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from latentstrat.config import LatentStratOptions, default_options
from latentstrat.data import Split, target_matrix
from latentstrat.model import SetTransformerModel, init_model, optimizer_parameter_groups


@dataclass
class LossMetrics:
    continuous_loss: float
    binary_loss: float
    l2_loss: float
    embedding_l2_loss: float
    head_l2_loss: float
    set_l2_loss: float


@dataclass
class TrainingDiagnostics:
    initial_loss: float
    final_loss: float
    final_validation_loss: float
    iterations: int
    bin_positive_weights: np.ndarray
    stopped_early: bool
    best_epoch: int | None
    stop_epoch: int
    best_validation_loss: float
    restored_best_validation_model: bool
    device: str
    amp_enabled: bool
    compiled: bool


class MatchTensorDataset(Dataset):
    """Tensor-backed match dataset for DataLoader-based training."""

    def __init__(
        self,
        red_team_idx: Tensor,
        blue_team_idx: Tensor,
        cont_targets: Tensor,
        bin_targets: Tensor,
    ) -> None:
        self.red_team_idx = red_team_idx
        self.blue_team_idx = blue_team_idx
        self.cont_targets = cont_targets
        self.bin_targets = bin_targets

    @classmethod
    def from_table(
        cls, table: pd.DataFrame, opts: LatentStratOptions | None = None
    ) -> MatchTensorDataset:
        opts = opts or default_options()
        red, blue = match_team_matrices(table)
        cont_targets = target_matrix(
            table, [f"{mapping.target_name}_z" for mapping in opts.target_map]
        )
        bin_targets = target_matrix(table, list(opts.binary_targets))
        return cls(
            red_team_idx=torch.as_tensor(red, dtype=torch.long),
            blue_team_idx=torch.as_tensor(blue, dtype=torch.long),
            cont_targets=torch.as_tensor(cont_targets, dtype=torch.float32),
            bin_targets=torch.as_tensor(bin_targets, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return int(self.red_team_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.red_team_idx[index],
            self.blue_team_idx[index],
            self.cont_targets[index],
            self.bin_targets[index],
        )


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_enabled(opts: LatentStratOptions, device: torch.device) -> bool:
    if isinstance(opts.use_amp, bool):
        return opts.use_amp and device.type == "cuda"
    if opts.use_amp != "auto":
        raise ValueError('use_amp must be a bool or "auto".')
    return device.type == "cuda"


def resolve_compile_enabled(opts: LatentStratOptions, device: torch.device) -> bool:
    if isinstance(opts.compile_model, bool):
        return opts.compile_model
    if opts.compile_model != "auto":
        raise ValueError('compile_model must be a bool or "auto".')
    return device.type == "cuda" and sys.platform != "win32"


def compile_forward_model(
    model: SetTransformerModel, opts: LatentStratOptions, device: torch.device
) -> tuple[torch.nn.Module, bool]:
    enabled = resolve_compile_enabled(opts, device)
    if not enabled:
        return model, False
    try:
        return torch.compile(model), True
    except Exception as exc:
        if opts.compile_model is True:
            raise
        warnings.warn(
            f"torch.compile failed in auto mode; continuing uncompiled: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return model, False


def match_team_matrices(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    red_columns = ["red_team_1_idx", "red_team_2_idx", "red_team_3_idx"]
    blue_columns = ["blue_team_1_idx", "blue_team_2_idx", "blue_team_3_idx"]
    values = table[red_columns + blue_columns].to_numpy(dtype=object)
    if pd.isna(values).any():
        raise ValueError("Training/evaluation requires nonmissing team indices.")
    return table[red_columns].to_numpy(dtype=np.int64), table[blue_columns].to_numpy(dtype=np.int64)


def resolve_positive_weights(bin_targets: np.ndarray, opts: LatentStratOptions) -> np.ndarray:
    if bin_targets.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(opts.bin_positive_weights, list):
        return np.asarray(opts.bin_positive_weights, dtype=np.float32)
    if opts.bin_positive_weights != "auto":
        raise ValueError('bin_positive_weights must be numeric or "auto".')
    positives = np.sum(bin_targets == 1, axis=0)
    negatives = np.sum(bin_targets == 0, axis=0)
    weights = negatives / np.maximum(positives, 1)
    weights[(positives == 0) | (negatives == 0)] = 1
    return weights.astype(np.float32)


def batch_to_device(
    batch: tuple[Tensor, Tensor, Tensor, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    non_blocking = device.type == "cuda"
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)


def active_embedding_l2(
    model: SetTransformerModel, red_team_idx: Tensor, blue_team_idx: Tensor, coefficient: float
) -> Tensor:
    if coefficient == 0:
        return torch.zeros((), dtype=model.team_embedding.weight.dtype, device=red_team_idx.device)
    active_teams = torch.unique(torch.cat([red_team_idx.flatten(), blue_team_idx.flatten()]))
    return coefficient * model.team_embedding(active_teams).pow(2).sum()


def model_loss(
    model: SetTransformerModel,
    red_team_idx: Tensor,
    blue_team_idx: Tensor,
    cont_targets: Tensor,
    bin_targets: Tensor,
    opts: LatentStratOptions | None = None,
    positive_weights: Tensor | None = None,
    forward_model: torch.nn.Module | None = None,
) -> tuple[Tensor, LossMetrics, object]:
    opts = opts or default_options()
    pred = (forward_model or model)(red_team_idx, blue_team_idx)
    if cont_targets.numel() == 0:
        cont_loss = torch.zeros((), dtype=pred.cont_z.dtype, device=pred.cont_z.device)
    else:
        cont_loss = F.mse_loss(pred.cont_z, cont_targets)
    if bin_targets.numel() == 0:
        bin_loss = torch.zeros((), dtype=pred.cont_z.dtype, device=pred.cont_z.device)
    else:
        bin_loss = F.binary_cross_entropy_with_logits(
            pred.bin_logits,
            bin_targets,
            pos_weight=positive_weights,
        )
    emb_l2 = active_embedding_l2(model, red_team_idx, blue_team_idx, opts.l2_embedding)
    total = cont_loss + bin_loss + emb_l2
    metrics = LossMetrics(
        continuous_loss=float(cont_loss.detach().cpu()),
        binary_loss=float(bin_loss.detach().cpu()),
        l2_loss=float(emb_l2.detach().cpu()),
        embedding_l2_loss=float(emb_l2.detach().cpu()),
        head_l2_loss=0.0,
        set_l2_loss=0.0,
    )
    return total, metrics, pred


def create_optimizer(model: SetTransformerModel, opts: LatentStratOptions) -> torch.optim.AdamW:
    return torch.optim.AdamW(optimizer_parameter_groups(model, opts), lr=opts.learning_rate)


def _make_loader(
    dataset: MatchTensorDataset,
    indices: np.ndarray,
    opts: LatentStratOptions,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> DataLoader:
    pin_memory = bool(device is not None and device.type == "cuda")
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=opts.mini_batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=opts.dataloader_num_workers,
        pin_memory=pin_memory,
    )


def evaluate_loss(
    model: SetTransformerModel,
    data_loader: DataLoader,
    opts: LatentStratOptions,
    positive_weights: Tensor,
    device: torch.device,
    *,
    forward_model: torch.nn.Module | None = None,
    amp_enabled: bool = False,
) -> float:
    was_training = model.training
    model.eval()
    active_model = forward_model or model
    active_model.eval()
    total_loss = 0.0
    total_rows = 0
    with torch.inference_mode():
        for batch in data_loader:
            red, blue, cont, binary = batch_to_device(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                loss, _, _ = model_loss(
                    model,
                    red,
                    blue,
                    cont,
                    binary,
                    opts,
                    positive_weights,
                    active_model,
                )
            batch_size = int(red.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_rows += batch_size
    if was_training:
        model.train()
        active_model.train()
    return total_loss / max(total_rows, 1)


def train_model(
    table: pd.DataFrame,
    split: Split,
    opts: LatentStratOptions | None = None,
    *,
    initial_model: SetTransformerModel | None = None,
    verbose: bool = True,
) -> tuple[SetTransformerModel, pd.DataFrame, TrainingDiagnostics]:
    opts = opts or default_options()
    device = resolve_device(opts.device)
    dataset = MatchTensorDataset.from_table(table, opts)
    train_rows = np.flatnonzero(split.train_mask)
    validation_rows = np.flatnonzero(split.validation_mask)
    if len(train_rows) == 0:
        raise ValueError("At least one training row is required.")

    positive_weights_np = resolve_positive_weights(dataset.bin_targets[train_rows].numpy(), opts)
    positive_weights = torch.as_tensor(positive_weights_np, dtype=torch.float32, device=device)
    num_teams = int(max(dataset.red_team_idx.max(), dataset.blue_team_idx.max()).item()) + 1
    model = initial_model or init_model(
        num_teams,
        opts.latent_dim,
        dataset.cont_targets.shape[1],
        dataset.bin_targets.shape[1],
        opts,
    )
    model.to(device)
    amp_enabled = resolve_amp_enabled(opts, device)
    forward_model, compiled = compile_forward_model(model, opts, device)
    optimizer = create_optimizer(model, opts)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    generator = torch.Generator()
    generator.manual_seed(opts.random_seed)
    train_loader = _make_loader(
        dataset, train_rows, opts, shuffle=True, generator=generator, device=device
    )
    train_eval_loader = _make_loader(dataset, train_rows, opts, shuffle=False, device=device)
    validation_loader = _make_loader(dataset, validation_rows, opts, shuffle=False, device=device)

    initial_loss = evaluate_loss(
        model,
        train_eval_loader,
        opts,
        positive_weights,
        device,
        forward_model=forward_model,
        amp_enabled=amp_enabled,
    )
    best_model = None
    best_epoch = None
    best_validation = float("inf")
    patience_counter = 0
    stopped = False
    iteration = 0
    rows = []

    for epoch in range(1, opts.epochs + 1):
        model.train()
        forward_model.train()
        for batch in train_loader:
            red, blue, cont, binary = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                loss, _, _ = model_loss(
                    model,
                    red,
                    blue,
                    cont,
                    binary,
                    opts,
                    positive_weights,
                    forward_model,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iteration += 1

        train_loss = evaluate_loss(
            model,
            train_eval_loader,
            opts,
            positive_weights,
            device,
            forward_model=forward_model,
            amp_enabled=amp_enabled,
        )
        validation_loss = np.nan
        if len(validation_rows):
            validation_loss = evaluate_loss(
                model,
                validation_loader,
                opts,
                positive_weights,
                device,
                forward_model=forward_model,
                amp_enabled=amp_enabled,
            )
        rows.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})

        if verbose and (epoch == 1 or epoch == opts.epochs or epoch % 25 == 0):
            print(f"Epoch {epoch}/{opts.epochs}: train loss {train_loss:.4f}")

        if opts.use_early_stopping and len(validation_rows):
            if validation_loss < best_validation - opts.early_stopping_min_delta:
                best_validation = float(validation_loss)
                best_epoch = epoch
                best_model = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                stopped = patience_counter >= opts.early_stopping_patience
                if stopped:
                    break

    restored = False
    if opts.restore_best_validation_model and best_model is not None:
        model.load_state_dict(best_model)
        restored = True

    final_loss = evaluate_loss(
        model,
        train_eval_loader,
        opts,
        positive_weights,
        device,
        forward_model=forward_model,
        amp_enabled=amp_enabled,
    )
    final_validation = np.nan
    if len(validation_rows):
        final_validation = evaluate_loss(
            model,
            validation_loader,
            opts,
            positive_weights,
            device,
            forward_model=forward_model,
            amp_enabled=amp_enabled,
        )
    history = pd.DataFrame(rows)
    diagnostics = TrainingDiagnostics(
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_validation_loss=float(final_validation),
        iterations=iteration,
        bin_positive_weights=positive_weights_np,
        stopped_early=stopped,
        best_epoch=best_epoch,
        stop_epoch=int(history["epoch"].iloc[-1]),
        best_validation_loss=float(best_validation if np.isfinite(best_validation) else np.nan),
        restored_best_validation_model=restored,
        device=device.type,
        amp_enabled=amp_enabled,
        compiled=compiled,
    )
    return model, history, diagnostics
