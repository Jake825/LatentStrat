"""Training utilities for LatentStrat's native PyTorch V5 workflow."""

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
    endgame_loss: float
    award_loss: float
    l2_loss: float
    embedding_l2_loss: float
    event_embedding_l2_loss: float
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
    """Tensor-backed V5 match dataset for DataLoader-based training."""

    def __init__(
        self,
        red_team_idx: Tensor,
        blue_team_idx: Tensor,
        red_event_idx: Tensor,
        blue_event_idx: Tensor,
        red_missing_mask: Tensor,
        blue_missing_mask: Tensor,
        cont_targets: Tensor,
        bin_targets: Tensor,
        endgame_targets: Tensor,
        award_targets: Tensor,
    ) -> None:
        self.red_team_idx = red_team_idx
        self.blue_team_idx = blue_team_idx
        self.red_event_idx = red_event_idx
        self.blue_event_idx = blue_event_idx
        self.red_missing_mask = red_missing_mask
        self.blue_missing_mask = blue_missing_mask
        self.cont_targets = cont_targets
        self.bin_targets = bin_targets
        self.endgame_targets = endgame_targets
        self.award_targets = award_targets

    @classmethod
    def from_table(
        cls, table: pd.DataFrame, opts: LatentStratOptions | None = None
    ) -> MatchTensorDataset:
        opts = opts or default_options()
        matrices = match_v5_matrices(table)
        cont_targets = target_matrix(
            table, [f"{mapping.target_name}_z" for mapping in opts.target_map]
        )
        bin_targets = target_matrix(table, list(opts.binary_targets))
        endgame_targets = endgame_target_matrix(table, opts)
        award_targets = award_target_tensor(table, opts)
        return cls(
            red_team_idx=torch.as_tensor(matrices[0], dtype=torch.long),
            blue_team_idx=torch.as_tensor(matrices[1], dtype=torch.long),
            red_event_idx=torch.as_tensor(matrices[2], dtype=torch.long),
            blue_event_idx=torch.as_tensor(matrices[3], dtype=torch.long),
            red_missing_mask=torch.as_tensor(matrices[4], dtype=torch.bool),
            blue_missing_mask=torch.as_tensor(matrices[5], dtype=torch.bool),
            cont_targets=torch.as_tensor(cont_targets, dtype=torch.float32),
            bin_targets=torch.as_tensor(bin_targets, dtype=torch.float32),
            endgame_targets=torch.as_tensor(endgame_targets, dtype=torch.long),
            award_targets=torch.as_tensor(award_targets, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return int(self.red_team_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return (
            self.red_team_idx[index],
            self.blue_team_idx[index],
            self.red_event_idx[index],
            self.blue_event_idx[index],
            self.red_missing_mask[index],
            self.blue_missing_mask[index],
            self.cont_targets[index],
            self.bin_targets[index],
            self.endgame_targets[index],
            self.award_targets[index],
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


def _slot_columns(suffix: str) -> list[str]:
    return [f"{color}_team_{slot}_{suffix}" for color in ("red", "blue") for slot in (1, 2, 3)]


def _matrix(table: pd.DataFrame, columns: list[str], *, dtype, default=0) -> np.ndarray:
    values = []
    for column in columns:
        if column in table.columns:
            values.append(table[column].fillna(default).to_numpy(dtype=dtype))
        else:
            values.append(np.full(len(table), default, dtype=dtype))
    return np.stack(values, axis=1)


def match_v5_matrices(
    table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    red_base_columns = [f"red_team_{slot}_base_idx" for slot in (1, 2, 3)]
    blue_base_columns = [f"blue_team_{slot}_base_idx" for slot in (1, 2, 3)]
    if not all(column in table.columns for column in red_base_columns + blue_base_columns):
        red_base_columns = [f"red_team_{slot}_idx" for slot in (1, 2, 3)]
        blue_base_columns = [f"blue_team_{slot}_idx" for slot in (1, 2, 3)]
    red_event_columns = [f"red_team_{slot}_event_idx" for slot in (1, 2, 3)]
    blue_event_columns = [f"blue_team_{slot}_event_idx" for slot in (1, 2, 3)]
    red_missing_columns = [f"red_team_{slot}_missing_team_mask" for slot in (1, 2, 3)]
    blue_missing_columns = [f"blue_team_{slot}_missing_team_mask" for slot in (1, 2, 3)]
    red_base = _matrix(table, red_base_columns, dtype=np.int64)
    blue_base = _matrix(table, blue_base_columns, dtype=np.int64)
    red_event = _matrix(table, red_event_columns, dtype=np.int64)
    blue_event = _matrix(table, blue_event_columns, dtype=np.int64)
    red_missing = _matrix(table, red_missing_columns, dtype=bool, default=False) | (red_base == 0)
    blue_missing = _matrix(table, blue_missing_columns, dtype=bool, default=False) | (
        blue_base == 0
    )
    return red_base, blue_base, red_event, blue_event, red_missing, blue_missing


def match_team_matrices(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    red, blue, *_ = match_v5_matrices(table)
    return red, blue


def endgame_target_matrix(table: pd.DataFrame, opts: LatentStratOptions) -> np.ndarray:
    mapping = {name: idx for idx, name in enumerate(opts.endgame_class_order)}
    columns = _slot_columns("endgame_status")
    result = np.zeros((len(table), 6), dtype=np.int64)
    for idx, column in enumerate(columns):
        if column not in table.columns:
            continue
        result[:, idx] = [
            mapping.get(str(value), 0) if pd.notna(value) else 0 for value in table[column]
        ]
    return result


def award_target_tensor(table: pd.DataFrame, opts: LatentStratOptions) -> np.ndarray:
    result = np.full((len(table), 6, len(opts.award_targets)), np.nan, dtype=np.float32)
    slot_prefixes = [f"{color}_team_{slot}" for color in ("red", "blue") for slot in (1, 2, 3)]
    for slot_idx, prefix in enumerate(slot_prefixes):
        for award_idx, axis in enumerate(opts.award_targets):
            column = f"{prefix}_award_{axis}"
            if column in table.columns:
                result[:, slot_idx, award_idx] = table[column].astype(float).to_numpy()
    return result


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


def batch_to_device(batch: tuple[Tensor, ...], device: torch.device) -> tuple[Tensor, ...]:
    non_blocking = device.type == "cuda"
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)


def _active_embedding_l2(
    embedding: torch.nn.Embedding, indices: Tensor, coefficient: float
) -> Tensor:
    if coefficient == 0:
        return torch.zeros((), dtype=embedding.weight.dtype, device=indices.device)
    active = torch.unique(indices.flatten())
    active = active[active > 0]
    if active.numel() == 0:
        return torch.zeros((), dtype=embedding.weight.dtype, device=indices.device)
    return coefficient * embedding(active).pow(2).sum()


def active_embedding_l2(
    model: SetTransformerModel,
    red_team_idx: Tensor,
    blue_team_idx: Tensor,
    coefficient: float,
) -> Tensor:
    active_indices = torch.cat([red_team_idx, blue_team_idx], dim=1)
    return _active_embedding_l2(model.Z_base, active_indices, coefficient)


def active_event_embedding_l2(
    model: SetTransformerModel,
    red_event_idx: Tensor,
    blue_event_idx: Tensor,
    coefficient: float,
) -> Tensor:
    return _active_embedding_l2(
        model.Z_event, torch.cat([red_event_idx, blue_event_idx], dim=1), coefficient
    )


def ordinal_targets_to_cumulative(targets: Tensor, num_classes: int) -> Tensor:
    thresholds = torch.arange(num_classes - 1, device=targets.device).view(1, 1, -1)
    return (targets.unsqueeze(-1) > thresholds).to(torch.float32)


def _homoscedastic(loss: Tensor, log_var: Tensor, valid: bool) -> Tensor:
    if not valid:
        return torch.zeros((), dtype=loss.dtype, device=loss.device)
    return torch.exp(-log_var) * loss + log_var


def model_loss(
    model: SetTransformerModel,
    red_team_idx: Tensor,
    blue_team_idx: Tensor,
    cont_targets: Tensor,
    bin_targets: Tensor,
    opts: LatentStratOptions | None = None,
    positive_weights: Tensor | None = None,
    forward_model: torch.nn.Module | None = None,
    *,
    red_event_idx: Tensor | None = None,
    blue_event_idx: Tensor | None = None,
    red_missing_mask: Tensor | None = None,
    blue_missing_mask: Tensor | None = None,
    endgame_targets: Tensor | None = None,
    award_targets: Tensor | None = None,
) -> tuple[Tensor, LossMetrics, object]:
    opts = opts or default_options()
    pred = (forward_model or model)(
        red_team_idx,
        blue_team_idx,
        red_event_idx=red_event_idx,
        blue_event_idx=blue_event_idx,
        red_missing_mask=red_missing_mask,
        blue_missing_mask=blue_missing_mask,
    )
    zero = torch.zeros((), dtype=pred.cont_z.dtype, device=pred.cont_z.device)
    if cont_targets.numel() == 0:
        raw_cont_loss = zero
        cont_term = zero
    else:
        raw_cont_loss = F.mse_loss(pred.cont_z, cont_targets)
        cont_term = _homoscedastic(raw_cont_loss, model.log_var_continuous, True)
    if bin_targets.numel() == 0:
        raw_bin_loss = zero
        bin_term = zero
    else:
        raw_bin_loss = F.binary_cross_entropy_with_logits(
            pred.bin_logits,
            bin_targets,
            pos_weight=positive_weights,
        )
        bin_term = _homoscedastic(raw_bin_loss, model.log_var_win, True)

    slot_missing = torch.cat([pred.red_missing_mask, pred.blue_missing_mask], dim=1)
    if endgame_targets is None or pred.endgame_logits.numel() == 0:
        raw_endgame_loss = zero
        endgame_term = zero
    else:
        cumulative = ordinal_targets_to_cumulative(endgame_targets, model.num_endgame_classes)
        valid = ~slot_missing
        if torch.any(valid):
            raw_endgame_loss = F.binary_cross_entropy_with_logits(
                pred.endgame_logits[valid], cumulative[valid]
            )
            endgame_term = _homoscedastic(raw_endgame_loss, model.log_var_endgame, True)
        else:
            raw_endgame_loss = zero
            endgame_term = zero

    if award_targets is None or pred.award_logits.numel() == 0:
        raw_award_loss = zero
        award_term = zero
    else:
        finite = torch.isfinite(award_targets) & ~slot_missing.unsqueeze(-1)
        if torch.any(finite):
            raw_award_loss = F.binary_cross_entropy_with_logits(
                pred.award_logits[finite], award_targets[finite]
            )
            award_term = _homoscedastic(raw_award_loss, model.log_var_awards, True)
        else:
            raw_award_loss = zero
            award_term = zero

    emb_l2 = active_embedding_l2(model, red_team_idx, blue_team_idx, opts.l2_embedding)
    event_l2 = zero
    if red_event_idx is not None and blue_event_idx is not None:
        event_l2 = active_event_embedding_l2(
            model, red_event_idx, blue_event_idx, opts.event_delta_l2
        )
    total = cont_term + bin_term + endgame_term + award_term + emb_l2 + event_l2
    metrics = LossMetrics(
        continuous_loss=float(raw_cont_loss.detach().cpu()),
        binary_loss=float(raw_bin_loss.detach().cpu()),
        endgame_loss=float(raw_endgame_loss.detach().cpu()),
        award_loss=float(raw_award_loss.detach().cpu()),
        l2_loss=float((emb_l2 + event_l2).detach().cpu()),
        embedding_l2_loss=float(emb_l2.detach().cpu()),
        event_embedding_l2_loss=float(event_l2.detach().cpu()),
        head_l2_loss=0.0,
        set_l2_loss=0.0,
    )
    return total, metrics, pred


def create_optimizer(model: SetTransformerModel, opts: LatentStratOptions) -> torch.optim.AdamW:
    return torch.optim.AdamW(optimizer_parameter_groups(model, opts), lr=opts.learning_rate)


def freeze_for_venue_mode(model: SetTransformerModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.Z_event.weight.requires_grad = True


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


def _loss_from_batch(
    model: SetTransformerModel,
    batch: tuple[Tensor, ...],
    opts: LatentStratOptions,
    positive_weights: Tensor,
    forward_model: torch.nn.Module | None,
) -> tuple[Tensor, LossMetrics, object]:
    (
        red,
        blue,
        red_event,
        blue_event,
        red_missing,
        blue_missing,
        cont,
        binary,
        endgame,
        awards,
    ) = batch
    return model_loss(
        model,
        red,
        blue,
        cont,
        binary,
        opts,
        positive_weights,
        forward_model,
        red_event_idx=red_event,
        blue_event_idx=blue_event,
        red_missing_mask=red_missing,
        blue_missing_mask=blue_missing,
        endgame_targets=endgame,
        award_targets=awards,
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
            batch = batch_to_device(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                loss, _, _ = _loss_from_batch(
                    model, batch, opts, positive_weights, active_model
                )
            batch_size = int(batch[0].shape[0])
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
    venue_mode: bool = False,
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
    num_event_teams = int(max(dataset.red_event_idx.max(), dataset.blue_event_idx.max()).item()) + 1
    model = initial_model or init_model(
        num_teams,
        opts.latent_dim,
        dataset.cont_targets.shape[1],
        dataset.bin_targets.shape[1],
        opts,
        num_event_teams=num_event_teams,
        num_endgame_classes=len(opts.endgame_class_order),
        num_awards=len(opts.award_targets),
    )
    if venue_mode:
        freeze_for_venue_mode(model)
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
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                loss, _, _ = _loss_from_batch(
                    model, batch, opts, positive_weights, forward_model
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
