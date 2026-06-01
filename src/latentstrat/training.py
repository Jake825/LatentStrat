"""Training utilities for LatentStrat's native PyTorch V5 workflow."""

from __future__ import annotations

import math
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from latentstrat.config import LatentStratOptions, default_options
from latentstrat.data import (
    Split,
    optional_target_matrix,
    target_matrix,
    v57_binary_target_names,
    v57_continuous_target_names,
)
from latentstrat.model import SetTransformerModel, init_model, optimizer_parameter_groups
from latentstrat.world_model import WorldModelOptions


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
    atomic_loss: float = 0.0
    foul_loss: float = 0.0
    bonus_loss: float = 0.0
    special_loss: float = 0.0
    rank_loss: float = 0.0
    playoff_loss: float = 0.0
    selection_loss: float = 0.0
    wm_award_loss: float = 0.0
    wm_rank_loss: float = 0.0
    wm_pick_loss: float = 0.0


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


TASK_NAMES = (
    "continuous",
    "win",
    "endgame",
    "awards",
    "atomic",
    "foul",
    "bonus",
    "special",
    "rank",
    "playoff",
    "selection",
    "wm_award",
    "wm_rank",
    "wm_pick",
)


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
        v57_cont_targets: Tensor | None = None,
        v57_bin_targets: Tensor | None = None,
        wm_award_targets: Tensor | None = None,
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
        self.v57_cont_targets = (
            v57_cont_targets
            if v57_cont_targets is not None
            else torch.empty((len(red_team_idx), 0), dtype=torch.float32)
        )
        self.v57_bin_targets = (
            v57_bin_targets
            if v57_bin_targets is not None
            else torch.empty((len(red_team_idx), 0), dtype=torch.float32)
        )
        self.wm_award_targets = (
            wm_award_targets
            if wm_award_targets is not None
            else torch.empty((len(red_team_idx), 6, 0), dtype=torch.float32)
        )

    @classmethod
    def from_table(
        cls,
        table: pd.DataFrame,
        opts: LatentStratOptions | None = None,
        world_model_opts: WorldModelOptions | None = None,
    ) -> MatchTensorDataset:
        opts = opts or default_options()
        matrices = match_v5_matrices(table)
        cont_targets = target_matrix(
            table, [f"{mapping.target_name}_z" for mapping in opts.target_map]
        )
        bin_targets = target_matrix(table, list(opts.binary_targets))
        v57_cont_columns = [f"{name}_z" for name in v57_continuous_target_names(opts)]
        v57_bin_columns = v57_binary_target_names(opts)
        v57_cont_targets = (
            optional_target_matrix(table, v57_cont_columns)
            if any(column in table.columns for column in v57_cont_columns)
            else np.empty((len(table), 0), dtype=float)
        )
        v57_bin_targets = (
            optional_target_matrix(table, v57_bin_columns)
            if any(column in table.columns for column in v57_bin_columns)
            else np.empty((len(table), 0), dtype=float)
        )
        endgame_targets = endgame_target_matrix(table, opts)
        award_targets = award_target_tensor(table, opts)
        world_model_opts = world_model_opts or WorldModelOptions()
        wm_award_targets = world_award_target_tensor(
            table, world_model_opts.award_embedding.width
        )
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
            v57_cont_targets=torch.as_tensor(v57_cont_targets, dtype=torch.float32),
            v57_bin_targets=torch.as_tensor(v57_bin_targets, dtype=torch.float32),
            wm_award_targets=torch.as_tensor(wm_award_targets, dtype=torch.float32),
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
            self.v57_cont_targets[index],
            self.v57_bin_targets[index],
            self.wm_award_targets[index],
        )


class RankPairDataset(Dataset):
    def __init__(
        self,
        higher_base_idx: Tensor,
        lower_base_idx: Tensor,
        higher_event_idx: Tensor,
        lower_event_idx: Tensor,
    ) -> None:
        self.higher_base_idx = higher_base_idx
        self.lower_base_idx = lower_base_idx
        self.higher_event_idx = higher_event_idx
        self.lower_event_idx = lower_event_idx

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> RankPairDataset:
        rows = []
        if table is not None and not table.empty:
            for _, event_rows in table.dropna(subset=["qual_rank"]).groupby("event_key"):
                ordered = event_rows.sort_values("qual_rank")
                left_rows = ordered.iloc[:-1].itertuples()
                right_rows = ordered.iloc[1:].itertuples()
                for left, right in zip(left_rows, right_rows, strict=False):
                    rows.append(
                        (
                            int(left.team_base_idx),
                            int(right.team_base_idx),
                            int(getattr(left, "team_event_idx", 0)),
                            int(getattr(right, "team_event_idx", 0)),
                        )
                    )
        data = np.asarray(rows, dtype=np.int64).reshape((-1, 4))
        return cls(
            torch.as_tensor(data[:, 0], dtype=torch.long),
            torch.as_tensor(data[:, 1], dtype=torch.long),
            torch.as_tensor(data[:, 2], dtype=torch.long),
            torch.as_tensor(data[:, 3], dtype=torch.long),
        )

    def __len__(self) -> int:
        return int(self.higher_base_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return (
            self.higher_base_idx[index],
            self.lower_base_idx[index],
            self.higher_event_idx[index],
            self.lower_event_idx[index],
        )


class AlliancePairDataset(Dataset):
    def __init__(
        self,
        better_base_idx: Tensor,
        worse_base_idx: Tensor,
        better_event_idx: Tensor,
        worse_event_idx: Tensor,
    ) -> None:
        self.better_base_idx = better_base_idx
        self.worse_base_idx = worse_base_idx
        self.better_event_idx = better_event_idx
        self.worse_event_idx = worse_event_idx

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> AlliancePairDataset:
        rows = []
        if table is not None and not table.empty:
            for _, event_rows in table.dropna(subset=["playoff_finish_order"]).groupby("event_key"):
                ordered = event_rows.sort_values("playoff_finish_order")
                left_rows = ordered.iloc[:-1].itertuples()
                right_rows = ordered.iloc[1:].itertuples()
                for left, right in zip(left_rows, right_rows, strict=False):
                    rows.append(
                        (
                            [int(getattr(left, f"team_{slot}_base_idx")) for slot in (1, 2, 3)],
                            [int(getattr(right, f"team_{slot}_base_idx")) for slot in (1, 2, 3)],
                            [int(getattr(left, f"team_{slot}_event_idx")) for slot in (1, 2, 3)],
                            [int(getattr(right, f"team_{slot}_event_idx")) for slot in (1, 2, 3)],
                        )
                    )
        if rows:
            better_base, worse_base, better_event, worse_event = zip(*rows, strict=True)
        else:
            better_base = worse_base = better_event = worse_event = []
        return cls(
            torch.as_tensor(better_base, dtype=torch.long).reshape((-1, 3)),
            torch.as_tensor(worse_base, dtype=torch.long).reshape((-1, 3)),
            torch.as_tensor(better_event, dtype=torch.long).reshape((-1, 3)),
            torch.as_tensor(worse_event, dtype=torch.long).reshape((-1, 3)),
        )

    def __len__(self) -> int:
        return int(self.better_base_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return (
            self.better_base_idx[index],
            self.worse_base_idx[index],
            self.better_event_idx[index],
            self.worse_event_idx[index],
        )


class SelectionTripletDataset(Dataset):
    def __init__(
        self,
        captain_base_idx: Tensor,
        pick_base_idx: Tensor,
        passed_base_idx: Tensor,
        captain_event_idx: Tensor,
        pick_event_idx: Tensor,
        passed_event_idx: Tensor,
    ) -> None:
        self.captain_base_idx = captain_base_idx
        self.pick_base_idx = pick_base_idx
        self.passed_base_idx = passed_base_idx
        self.captain_event_idx = captain_event_idx
        self.pick_event_idx = pick_event_idx
        self.passed_event_idx = passed_event_idx

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> SelectionTripletDataset:
        required = ["captain_base_idx", "pick_base_idx", "passed_over_base_idx"]
        rows = []
        if table is not None and not table.empty and all(c in table.columns for c in required):
            valid = table.dropna(subset=required)
            for row in valid.itertuples():
                if int(row.passed_over_base_idx) <= 0:
                    continue
                rows.append(
                    (
                        int(row.captain_base_idx),
                        int(row.pick_base_idx),
                        int(row.passed_over_base_idx),
                        int(getattr(row, "captain_event_idx", 0)),
                        int(getattr(row, "pick_event_idx", 0)),
                        int(getattr(row, "passed_over_event_idx", 0)),
                    )
                )
        data = np.asarray(rows, dtype=np.int64).reshape((-1, 6))
        return cls(
            torch.as_tensor(data[:, 0], dtype=torch.long),
            torch.as_tensor(data[:, 1], dtype=torch.long),
            torch.as_tensor(data[:, 2], dtype=torch.long),
            torch.as_tensor(data[:, 3], dtype=torch.long),
            torch.as_tensor(data[:, 4], dtype=torch.long),
            torch.as_tensor(data[:, 5], dtype=torch.long),
        )

    def __len__(self) -> int:
        return int(self.captain_base_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return (
            self.captain_base_idx[index],
            self.pick_base_idx[index],
            self.passed_base_idx[index],
            self.captain_event_idx[index],
            self.pick_event_idx[index],
            self.passed_event_idx[index],
        )


class RankEmbeddingDataset(Dataset):
    def __init__(self, team_base_idx: Tensor, team_event_idx: Tensor, targets: Tensor) -> None:
        self.team_base_idx = team_base_idx
        self.team_event_idx = team_event_idx
        self.targets = targets

    @classmethod
    def from_table(cls, table: pd.DataFrame, width: int) -> RankEmbeddingDataset:
        required = ["team_base_idx", "team_event_idx"]
        if table is None or table.empty or not all(column in table.columns for column in required):
            return cls(
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty((0, width), dtype=torch.float32),
            )
        targets = world_embedding_matrix(table, "", width)
        return cls(
            torch.as_tensor(table["team_base_idx"].to_numpy(), dtype=torch.long),
            torch.as_tensor(table["team_event_idx"].to_numpy(), dtype=torch.long),
            torch.as_tensor(targets, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return int(self.team_base_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return self.team_base_idx[index], self.team_event_idx[index], self.targets[index]


class PickEmbeddingDataset(Dataset):
    def __init__(
        self,
        captain_base_idx: Tensor,
        candidate_base_idx: Tensor,
        captain_event_idx: Tensor,
        candidate_event_idx: Tensor,
        targets: Tensor,
    ) -> None:
        self.captain_base_idx = captain_base_idx
        self.candidate_base_idx = candidate_base_idx
        self.captain_event_idx = captain_event_idx
        self.candidate_event_idx = candidate_event_idx
        self.targets = targets

    @classmethod
    def from_table(cls, table: pd.DataFrame, width: int) -> PickEmbeddingDataset:
        required = [
            "captain_base_idx",
            "candidate_base_idx",
            "captain_event_idx",
            "candidate_event_idx",
        ]
        if table is None or table.empty or not all(column in table.columns for column in required):
            empty = torch.empty(0, dtype=torch.long)
            return cls(empty, empty, empty, empty, torch.empty((0, width), dtype=torch.float32))
        return cls(
            torch.as_tensor(table["captain_base_idx"].to_numpy(), dtype=torch.long),
            torch.as_tensor(table["candidate_base_idx"].to_numpy(), dtype=torch.long),
            torch.as_tensor(table["captain_event_idx"].to_numpy(), dtype=torch.long),
            torch.as_tensor(table["candidate_event_idx"].to_numpy(), dtype=torch.long),
            torch.as_tensor(world_embedding_matrix(table, "", width), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return int(self.captain_base_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return (
            self.captain_base_idx[index],
            self.candidate_base_idx[index],
            self.captain_event_idx[index],
            self.candidate_event_idx[index],
            self.targets[index],
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


def world_embedding_matrix(table: pd.DataFrame, prefix: str, width: int) -> np.ndarray:
    separator = "_" if prefix else ""
    columns = [f"{prefix}{separator}z_{idx:03d}" for idx in range(width)]
    return optional_target_matrix(table, columns)


def world_award_target_tensor(table: pd.DataFrame, width: int) -> np.ndarray:
    result = np.full((len(table), 6, width), np.nan, dtype=np.float32)
    slot_prefixes = [f"{color}_team_{slot}" for color in ("red", "blue") for slot in (1, 2, 3)]
    for slot_idx, prefix in enumerate(slot_prefixes):
        result[:, slot_idx, :] = world_embedding_matrix(table, f"{prefix}_wm_award", width)
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
    embedding: torch.nn.Embedding,
    indices: Tensor,
    coefficient: float,
    *,
    include_zero: bool = False,
) -> Tensor:
    if coefficient == 0 or not embedding.weight.requires_grad:
        return torch.zeros((), dtype=embedding.weight.dtype, device=indices.device)
    active = torch.unique(indices.flatten())
    active = active[active >= 0] if include_zero else active[active > 0]
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
    return _active_embedding_l2(model.Z_base, active_indices, coefficient, include_zero=True)


def active_event_embedding_l2(
    model: SetTransformerModel,
    red_event_idx: Tensor,
    blue_event_idx: Tensor,
    coefficient: float,
) -> Tensor:
    return _active_embedding_l2(
        model.Z_event,
        torch.cat([red_event_idx, blue_event_idx], dim=1),
        coefficient,
        include_zero=False,
    )


def ordinal_targets_to_cumulative(targets: Tensor, num_classes: int) -> Tensor:
    thresholds = torch.arange(num_classes - 1, device=targets.device).view(1, 1, -1)
    return (targets.unsqueeze(-1) > thresholds).to(torch.float32)


def _masked_mse(prediction: Tensor, target: Tensor) -> tuple[Tensor, bool]:
    valid = torch.isfinite(target)
    if not torch.any(valid):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device), False
    return F.mse_loss(prediction[valid], target[valid]), True


def _masked_bce_with_logits(prediction: Tensor, target: Tensor) -> tuple[Tensor, bool]:
    valid = torch.isfinite(target)
    if not torch.any(valid):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device), False
    return F.binary_cross_entropy_with_logits(prediction[valid], target[valid]), True


def _split_v57_targets(
    values: Tensor,
    opts: LatentStratOptions,
    *,
    continuous: bool,
) -> tuple[Tensor, Tensor]:
    if values.numel() == 0:
        empty = values.new_empty((values.shape[0], 0))
        return empty, empty
    first_width = 2 * (
        len(opts.atomic_count_targets) if continuous else len(opts.bonus_binary_targets)
    )
    return values[:, :first_width], values[:, first_width:]


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
    v57_cont_targets: Tensor | None = None,
    v57_bin_targets: Tensor | None = None,
    wm_award_targets: Tensor | None = None,
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
        raw_cont_loss, cont_active = _masked_mse(pred.cont_z, cont_targets)
        cont_term = model.balance_loss("continuous", raw_cont_loss, cont_active)
    if bin_targets.numel() == 0:
        raw_bin_loss = zero
        bin_term = zero
    else:
        raw_bin_loss = F.binary_cross_entropy_with_logits(
            pred.bin_logits,
            bin_targets,
            pos_weight=positive_weights,
        )
        bin_term = model.balance_loss("win", raw_bin_loss, True)

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
            endgame_term = model.balance_loss("endgame", raw_endgame_loss, True)
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
            award_term = model.balance_loss("awards", raw_award_loss, True)
        else:
            raw_award_loss = zero
            award_term = zero

    atomic_targets, foul_targets = (
        _split_v57_targets(v57_cont_targets, opts, continuous=True)
        if v57_cont_targets is not None
        else (zero.new_empty((pred.cont_z.shape[0], 0)), zero.new_empty((pred.cont_z.shape[0], 0)))
    )
    bonus_targets, special_targets = (
        _split_v57_targets(v57_bin_targets, opts, continuous=False)
        if v57_bin_targets is not None
        else (zero.new_empty((pred.cont_z.shape[0], 0)), zero.new_empty((pred.cont_z.shape[0], 0)))
    )
    if atomic_targets.numel() and pred.atomic_z.numel():
        raw_atomic_loss, atomic_active = _masked_mse(pred.atomic_z, atomic_targets)
    else:
        raw_atomic_loss, atomic_active = zero, False
    atomic_term = model.balance_loss("atomic", raw_atomic_loss, atomic_active)
    if foul_targets.numel() and pred.foul_z.numel():
        raw_foul_loss, foul_active = _masked_mse(pred.foul_z, foul_targets)
    else:
        raw_foul_loss, foul_active = zero, False
    foul_term = model.balance_loss("foul", raw_foul_loss, foul_active)
    if bonus_targets.numel() and pred.bonus_logits.numel():
        raw_bonus_loss, bonus_active = _masked_bce_with_logits(
            pred.bonus_logits, bonus_targets
        )
    else:
        raw_bonus_loss, bonus_active = zero, False
    bonus_term = model.balance_loss("bonus", raw_bonus_loss, bonus_active)
    if special_targets.numel() and pred.special_logits.numel():
        raw_special_loss, special_active = _masked_bce_with_logits(
            pred.special_logits, special_targets
        )
    else:
        raw_special_loss, special_active = zero, False
    special_term = model.balance_loss("special", raw_special_loss, special_active)

    world_options = model.world_model_opts
    if (
        world_options.enabled
        and world_options.award_embedding.enabled
        and wm_award_targets is not None
        and wm_award_targets.numel()
    ):
        valid_awards = torch.isfinite(wm_award_targets).all(dim=-1) & ~slot_missing
        if torch.any(valid_awards):
            raw_wm_award_loss = (
                1
                - F.cosine_similarity(
                    pred.wm_award_embedding[valid_awards],
                    wm_award_targets[valid_awards],
                    dim=-1,
                )
            ).mean()
            wm_award_active = True
        else:
            raw_wm_award_loss, wm_award_active = zero, False
    else:
        raw_wm_award_loss, wm_award_active = zero, False
    wm_award_term = model.balance_loss("wm_award", raw_wm_award_loss, wm_award_active)

    emb_l2 = active_embedding_l2(model, red_team_idx, blue_team_idx, opts.l2_embedding)
    event_l2 = zero
    if red_event_idx is not None and blue_event_idx is not None:
        event_l2 = active_event_embedding_l2(
            model, red_event_idx, blue_event_idx, opts.event_delta_l2
        )
    total = (
        cont_term
        + bin_term
        + endgame_term
        + award_term
        + atomic_term
        + foul_term
        + bonus_term
        + special_term
        + wm_award_term
        + emb_l2
        + event_l2
    )
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
        atomic_loss=float(raw_atomic_loss.detach().cpu()),
        foul_loss=float(raw_foul_loss.detach().cpu()),
        bonus_loss=float(raw_bonus_loss.detach().cpu()),
        special_loss=float(raw_special_loss.detach().cpu()),
        wm_award_loss=float(raw_wm_award_loss.detach().cpu()),
    )
    return total, metrics, pred


def create_optimizer(model: SetTransformerModel, opts: LatentStratOptions) -> torch.optim.AdamW:
    return torch.optim.AdamW(optimizer_parameter_groups(model, opts), lr=opts.learning_rate)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, opts: LatentStratOptions
) -> torch.optim.lr_scheduler.CosineAnnealingLR | None:
    if not opts.use_lr_scheduler:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(opts.epochs), 1),
        eta_min=float(opts.lr_eta_min),
    )


def clamp_loss_log_vars(model: SetTransformerModel, opts: LatentStratOptions) -> None:
    model.loss_balancer.clamp_(opts.loss_log_var_min, opts.loss_log_var_max)


def create_tensorboard_writer(log_dir: str) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled, but tensorboard is not installed. "
            'Run `pip install -e ".[dev]"` or pass `--no-tensorboard`.'
        ) from exc
    return SummaryWriter(str(log_dir))


def freeze_for_venue_mode(model: SetTransformerModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.Z_event.weight.requires_grad = True


def freeze_team_embedding_tables(model: SetTransformerModel) -> None:
    model.Z_base.weight.requires_grad = False
    model.Z_event.weight.requires_grad = False


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
        v57_cont,
        v57_bin,
        wm_award,
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
        v57_cont_targets=v57_cont,
        v57_bin_targets=v57_bin,
        wm_award_targets=wm_award,
    )


def _rank_loss_from_batch(
    model: SetTransformerModel, batch: tuple[Tensor, ...], opts: LatentStratOptions
) -> Tensor:
    higher, lower, higher_event, lower_event = batch
    higher_score = model.team_value(higher, higher_event)
    lower_score = model.team_value(lower, lower_event)
    target = torch.ones_like(higher_score)
    return F.margin_ranking_loss(
        higher_score, lower_score, target, margin=float(opts.rank_margin)
    )


def _playoff_loss_from_batch(
    model: SetTransformerModel, batch: tuple[Tensor, ...], opts: LatentStratOptions
) -> Tensor:
    better, worse, better_event, worse_event = batch
    better_score = model.alliance_value(better, better_event)
    worse_score = model.alliance_value(worse, worse_event)
    target = torch.ones_like(better_score)
    return F.margin_ranking_loss(
        better_score, worse_score, target, margin=float(opts.playoff_margin)
    )


def _selection_loss_from_batch(
    model: SetTransformerModel, batch: tuple[Tensor, ...], opts: LatentStratOptions
) -> Tensor:
    captain, pick, passed, captain_event, pick_event, passed_event = batch
    anchor = model.team_latent(captain, captain_event)
    positive = model.team_latent(pick, pick_event)
    negative = model.team_latent(passed, passed_event)
    return F.triplet_margin_loss(
        anchor, positive, negative, margin=float(opts.selection_triplet_margin)
    )


def _rank_embedding_loss_from_batch(
    model: SetTransformerModel, batch: tuple[Tensor, ...]
) -> Tensor:
    team, event, target = batch
    prediction = model.predict_rank_embedding(team, event)
    return _masked_mse(prediction, target)[0]


def _pick_embedding_loss_from_batch(
    model: SetTransformerModel, batch: tuple[Tensor, ...]
) -> Tensor:
    captain, candidate, captain_event, candidate_event, target = batch
    prediction = model.predict_selection_embedding(
        captain, candidate, captain_event, candidate_event
    )
    return _masked_mse(prediction, target)[0]


def _metrics_to_losses(metrics: LossMetrics) -> dict[str, float]:
    return {
        "continuous": metrics.continuous_loss,
        "win": metrics.binary_loss,
        "endgame": metrics.endgame_loss,
        "awards": metrics.award_loss,
        "atomic": metrics.atomic_loss,
        "foul": metrics.foul_loss,
        "bonus": metrics.bonus_loss,
        "special": metrics.special_loss,
        "wm_award": metrics.wm_award_loss,
    }


def _empty_epoch_accumulators() -> tuple[dict[str, float], dict[str, float]]:
    return ({name: 0.0 for name in TASK_NAMES}, {name: 0.0 for name in TASK_NAMES})


def _accumulate_loss(
    sums: dict[str, float],
    weights: dict[str, float],
    task_name: str,
    loss_value: float,
    weight: float,
) -> None:
    if not math.isfinite(loss_value) or weight <= 0:
        return
    sums[task_name] += float(loss_value) * float(weight)
    weights[task_name] += float(weight)


def _epoch_task_means(sums: dict[str, float], weights: dict[str, float]) -> dict[str, float]:
    return {
        name: (sums[name] / weights[name] if weights[name] > 0 else math.nan)
        for name in TASK_NAMES
    }


def _write_feature_tensorboard_epoch(
    writer: Any,
    *,
    model: SetTransformerModel,
    row: dict[str, float | int],
    task_means: dict[str, float],
    active_sidecars: set[str],
) -> None:
    epoch = int(row["epoch"])
    writer.add_scalar("Loss/Train_Total", float(row["train_loss"]), epoch)
    validation = float(row["validation_loss"])
    if math.isfinite(validation):
        writer.add_scalar("Loss/Validation_Total", validation, epoch)
    learning_rate = float(row.get("learning_rate", math.nan))
    if math.isfinite(learning_rate):
        writer.add_scalar("LR/base", learning_rate, epoch)
    for task_name, value in task_means.items():
        if math.isfinite(value):
            writer.add_scalar(f"LossRaw/{task_name}", value, epoch)
        writer.add_scalar(
            f"Active/{task_name}",
            1.0 if (math.isfinite(value) or task_name in active_sidecars) else 0.0,
            epoch,
        )
    for task_name, parameter in model.loss_balancer.log_vars.items():
        log_var = float(parameter.detach().cpu())
        writer.add_scalar(f"LogVar/{task_name}", log_var, epoch)
        writer.add_scalar(f"Weights/{task_name}_precision", math.exp(-log_var), epoch)


def _sidecar_loader(
    dataset: Dataset | None,
    opts: LatentStratOptions,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> DataLoader | None:
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=opts.mini_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=opts.dataloader_num_workers,
        pin_memory=device.type == "cuda",
    )


def _prepare_sidecar_loaders(
    sidecar_tables: dict[str, pd.DataFrame] | None,
    opts: LatentStratOptions,
    *,
    generator: torch.Generator,
    device: torch.device,
    world_model_opts: WorldModelOptions,
) -> dict[str, DataLoader]:
    if not sidecar_tables:
        return {}
    datasets: dict[str, Dataset] = {}
    if "rankings" in sidecar_tables:
        datasets["rank"] = RankPairDataset.from_table(sidecar_tables["rankings"])
    if "playoffs" in sidecar_tables:
        datasets["playoff"] = AlliancePairDataset.from_table(sidecar_tables["playoffs"])
    if "selections" in sidecar_tables:
        datasets["selection"] = SelectionTripletDataset.from_table(sidecar_tables["selections"])
    if "world_rank" in sidecar_tables and world_model_opts.rank_embedding.enabled:
        datasets["wm_rank"] = RankEmbeddingDataset.from_table(
            sidecar_tables["world_rank"], world_model_opts.rank_embedding.width
        )
    if "world_pick" in sidecar_tables and world_model_opts.pick_embedding.enabled:
        datasets["wm_pick"] = PickEmbeddingDataset.from_table(
            sidecar_tables["world_pick"], world_model_opts.pick_embedding.width
        )
    loaders = {}
    for name, dataset in datasets.items():
        loader = _sidecar_loader(dataset, opts, generator=generator, device=device)
        if loader is not None:
            loaders[name] = loader
    return loaders


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
    freeze_team_embeddings: bool = False,
    sidecar_tables: dict[str, pd.DataFrame] | None = None,
    tensorboard_writer: Any | None = None,
    world_model_opts: WorldModelOptions | None = None,
) -> tuple[SetTransformerModel, pd.DataFrame, TrainingDiagnostics]:
    opts = opts or default_options()
    world_model_opts = world_model_opts or (
        initial_model.world_model_opts if initial_model is not None else WorldModelOptions()
    )
    device = resolve_device(opts.device)
    dataset = MatchTensorDataset.from_table(table, opts, world_model_opts)
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
        world_model_opts=world_model_opts,
    )
    if venue_mode and freeze_team_embeddings:
        raise ValueError("venue_mode and freeze_team_embeddings cannot be used together.")
    if venue_mode:
        freeze_for_venue_mode(model)
    elif freeze_team_embeddings:
        freeze_team_embedding_tables(model)
    model.to(device)
    amp_enabled = resolve_amp_enabled(opts, device)
    forward_model, compiled = compile_forward_model(model, opts, device)
    optimizer = create_optimizer(model, opts)
    scheduler = create_lr_scheduler(optimizer, opts)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    generator = torch.Generator()
    generator.manual_seed(opts.random_seed)
    train_loader = _make_loader(
        dataset, train_rows, opts, shuffle=True, generator=generator, device=device
    )
    sidecar_loaders = _prepare_sidecar_loaders(
        sidecar_tables,
        opts,
        generator=generator,
        device=device,
        world_model_opts=world_model_opts,
    )
    sidecar_iters = {name: cycle(loader) for name, loader in sidecar_loaders.items()}
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
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        task_sums, task_weights = _empty_epoch_accumulators()
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                loss, metrics, _ = _loss_from_batch(
                    model, batch, opts, positive_weights, forward_model
                )
                batch_rows = int(batch[0].shape[0])
                for task_name, loss_value in _metrics_to_losses(metrics).items():
                    _accumulate_loss(
                        task_sums, task_weights, task_name, loss_value, batch_rows
                    )
                for name, iterator in sidecar_iters.items():
                    sidecar_batch = batch_to_device(next(iterator), device)
                    if name == "rank":
                        raw_loss = _rank_loss_from_batch(model, sidecar_batch, opts)
                    elif name == "playoff":
                        raw_loss = _playoff_loss_from_batch(model, sidecar_batch, opts)
                    elif name == "selection":
                        raw_loss = _selection_loss_from_batch(model, sidecar_batch, opts)
                    elif name == "wm_rank":
                        raw_loss = _rank_embedding_loss_from_batch(model, sidecar_batch)
                    elif name == "wm_pick":
                        raw_loss = _pick_embedding_loss_from_batch(model, sidecar_batch)
                    else:
                        continue
                    _accumulate_loss(
                        task_sums,
                        task_weights,
                        name,
                        float(raw_loss.detach().cpu()),
                        int(sidecar_batch[0].shape[0]),
                    )
                    loss = loss + model.balance_loss(name, raw_loss, True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            clamp_loss_log_vars(model, opts)
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
        task_means = _epoch_task_means(task_sums, task_weights)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": epoch_learning_rate,
        }
        for task_name, value in task_means.items():
            row[f"{task_name}_loss"] = value
        for task_name, parameter in model.loss_balancer.log_vars.items():
            log_var = float(parameter.detach().cpu())
            row[f"{task_name}_log_var"] = log_var
            row[f"{task_name}_precision"] = math.exp(-log_var)
        rows.append(row)
        if tensorboard_writer is not None:
            _write_feature_tensorboard_epoch(
                tensorboard_writer,
                model=model,
                row=row,
                task_means=task_means,
                active_sidecars=set(sidecar_loaders),
            )

        if verbose and (epoch == 1 or epoch == opts.epochs or epoch % 25 == 0):
            print(f"Epoch {epoch}/{opts.epochs}: train loss {train_loss:.4f}")

        if len(validation_rows) and np.isfinite(validation_loss):
            if validation_loss < best_validation - opts.early_stopping_min_delta:
                best_validation = float(validation_loss)
                best_epoch = epoch
                best_model = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                stopped = (
                    opts.use_early_stopping
                    and patience_counter >= opts.early_stopping_patience
                )
                if stopped:
                    break
        if scheduler is not None:
            scheduler.step()

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
