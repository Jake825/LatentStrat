"""Training loop and loss functions for LatentStrat V4."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from latentstrat.config import LatentStratOptions, default_options
from latentstrat.data import Split, target_matrix
from latentstrat.model import SetTransformerModel, init_model, require_torch


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


def match_team_matrices(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    red_columns = ["red_team_1_idx", "red_team_2_idx", "red_team_3_idx"]
    blue_columns = ["blue_team_1_idx", "blue_team_2_idx", "blue_team_3_idx"]
    values = table[red_columns + blue_columns].to_numpy(dtype=object)
    if pd.isna(values).any():
        raise ValueError("Training/evaluation requires nonmissing team indices.")
    red = table[red_columns].to_numpy(dtype=np.int64)
    blue = table[blue_columns].to_numpy(dtype=np.int64)
    return red, blue


def resolve_positive_weights(bin_targets: np.ndarray, opts: LatentStratOptions) -> np.ndarray:
    if bin_targets.size == 0:
        return np.zeros((0,), dtype=float)
    if isinstance(opts.bin_positive_weights, list):
        return np.asarray(opts.bin_positive_weights, dtype=float)
    if opts.bin_positive_weights != "auto":
        raise ValueError('bin_positive_weights must be numeric or "auto".')
    positives = np.sum(bin_targets == 1, axis=0)
    negatives = np.sum(bin_targets == 0, axis=0)
    weights = negatives / np.maximum(positives, 1)
    weights[(positives == 0) | (negatives == 0)] = 1
    return weights.astype(float)


def model_loss(
    model: SetTransformerModel,
    red_team_idx,
    blue_team_idx,
    cont_targets,
    bin_targets,
    opts: LatentStratOptions | None = None,
    positive_weights=None,
) -> tuple[object, LossMetrics, object]:
    torch = require_torch()
    opts = opts or default_options()
    pred = model(red_team_idx, blue_team_idx)
    if cont_targets.numel() == 0:
        cont_loss = torch.zeros((), dtype=pred.cont_z.dtype, device=pred.cont_z.device)
    else:
        cont_error = pred.cont_z - cont_targets
        cont_loss = torch.mean(torch.sum(cont_error**2, dim=1))
    if bin_targets.numel() == 0:
        bin_loss = torch.zeros((), dtype=pred.cont_z.dtype, device=pred.cont_z.device)
    else:
        weights = positive_weights
        if weights is None:
            weights = torch.ones((bin_targets.shape[1],), dtype=bin_targets.dtype, device=bin_targets.device)
        logits = pred.bin_logits
        bce = torch.clamp(logits, min=0) - logits * bin_targets + torch.log1p(torch.exp(-torch.abs(logits)))
        target_weights = 1 + (weights.view(1, -1) - 1) * bin_targets
        bin_loss = torch.mean(bce * target_weights)
    active_teams = torch.unique(torch.cat([red_team_idx.reshape(-1), blue_team_idx.reshape(-1)]))
    embedding_l2 = opts.l2_embedding * torch.sum(model.team_embedding(active_teams) ** 2)
    head_l2 = opts.l2_heads * (torch.sum(model.w_cont**2) + torch.sum(model.w_bin**2))
    set_l2 = torch.zeros((), dtype=pred.cont_z.dtype, device=pred.cont_z.device)
    for parameter in model.set_l2_parameters():
        set_l2 = set_l2 + opts.l2_set * torch.sum(parameter**2)
    l2_loss = embedding_l2 + head_l2 + set_l2
    total = cont_loss + bin_loss + l2_loss
    metrics = LossMetrics(
        continuous_loss=float(cont_loss.detach().cpu()),
        binary_loss=float(bin_loss.detach().cpu()),
        l2_loss=float(l2_loss.detach().cpu()),
        embedding_l2_loss=float(embedding_l2.detach().cpu()),
        head_l2_loss=float(head_l2.detach().cpu()),
        set_l2_loss=float(set_l2.detach().cpu()),
    )
    return total, metrics, pred


def _tensor(data, dtype=None):
    torch = require_torch()
    return torch.as_tensor(data, dtype=dtype)


def evaluate_loss(
    model: SetTransformerModel,
    red: np.ndarray,
    blue: np.ndarray,
    cont: np.ndarray,
    binary: np.ndarray,
    opts: LatentStratOptions,
    positive_weights,
) -> float:
    torch = require_torch()
    with torch.no_grad():
        loss, _, _ = model_loss(
            model,
            _tensor(red, torch.long),
            _tensor(blue, torch.long),
            _tensor(cont),
            _tensor(binary),
            opts,
            positive_weights,
        )
    return float(loss.detach().cpu())


def train_model(
    table: pd.DataFrame,
    split: Split,
    opts: LatentStratOptions | None = None,
    *,
    initial_model: SetTransformerModel | None = None,
    verbose: bool = True,
    batch_order: list[np.ndarray] | None = None,
) -> tuple[SetTransformerModel, pd.DataFrame, TrainingDiagnostics]:
    torch = require_torch()
    opts = opts or default_options()
    red, blue = match_team_matrices(table)
    cont_targets = target_matrix(table, [f"{mapping.target_name}_z" for mapping in opts.target_map])
    bin_targets = target_matrix(table, list(opts.binary_targets))
    train_rows = np.flatnonzero(split.train_mask)
    validation_rows = np.flatnonzero(split.validation_mask)
    positive_weights_np = resolve_positive_weights(bin_targets[train_rows], opts)
    positive_weights = _tensor(positive_weights_np)
    num_teams = int(max(red.max(), blue.max())) + 1
    model = initial_model or init_model(
        num_teams, opts.latent_dim, cont_targets.shape[1], bin_targets.shape[1], opts
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, weight_decay=0.0)
    initial_loss = evaluate_loss(
        model,
        red[train_rows],
        blue[train_rows],
        cont_targets[train_rows],
        bin_targets[train_rows],
        opts,
        positive_weights,
    )
    rng = np.random.default_rng(opts.random_seed)
    best_model = None
    best_epoch = None
    best_validation = float("inf")
    patience_counter = 0
    stopped = False
    iteration = 0
    rows = []
    for epoch in range(1, opts.epochs + 1):
        if batch_order is not None and epoch <= len(batch_order):
            shuffled = np.asarray(batch_order[epoch - 1], dtype=int)
        else:
            shuffled = rng.permutation(train_rows)
        for start in range(0, len(shuffled), opts.mini_batch_size):
            batch = shuffled[start : start + opts.mini_batch_size]
            iteration += 1
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model_loss(
                model,
                _tensor(red[batch], torch.long),
                _tensor(blue[batch], torch.long),
                _tensor(cont_targets[batch]),
                _tensor(bin_targets[batch]),
                opts,
                positive_weights,
            )
            loss.backward()
            optimizer.step()
        train_loss = evaluate_loss(
            model,
            red[train_rows],
            blue[train_rows],
            cont_targets[train_rows],
            bin_targets[train_rows],
            opts,
            positive_weights,
        )
        validation_loss = np.nan
        if len(validation_rows):
            validation_loss = evaluate_loss(
                model,
                red[validation_rows],
                blue[validation_rows],
                cont_targets[validation_rows],
                bin_targets[validation_rows],
                opts,
                positive_weights,
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
        red[train_rows],
        blue[train_rows],
        cont_targets[train_rows],
        bin_targets[train_rows],
        opts,
        positive_weights,
    )
    final_validation = np.nan
    if len(validation_rows):
        final_validation = evaluate_loss(
            model,
            red[validation_rows],
            blue[validation_rows],
            cont_targets[validation_rows],
            bin_targets[validation_rows],
            opts,
            positive_weights,
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
    )
    return model, history, diagnostics
