"""Training utilities for LatentStrat's native PyTorch V5 workflow."""

from __future__ import annotations

import hashlib
import math
import sys
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from latentstrat.config import LatentStratOptions, default_options
from latentstrat.experimental.frozen_targets import WorldModelOptions
from latentstrat.season.data import (
    Split,
    optional_target_matrix,
    target_matrix,
    v57_binary_target_names,
    v57_continuous_target_names,
)
from latentstrat.season.model import SetTransformerModel, init_model, optimizer_parameter_groups
from latentstrat.season.physics import (
    compose_predicted_scores,
    ordered_atomic_columns,
    ordered_bonus_columns,
    ordered_foul_columns,
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
    seed_dataloader_worker,
    seed_everything,
    validate_resume_checkpoint,
)


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
    auto_loss: float = 0.0
    rank_loss: float = 0.0
    playoff_loss: float = 0.0
    selection_loss: float = 0.0
    wm_award_loss: float = 0.0
    wm_rank_loss: float = 0.0
    wm_pick_loss: float = 0.0
    score_consistency_loss: float = 0.0
    score_ordering_loss: float = 0.0
    winner_consistency_loss: float = 0.0
    composed_score_rmse: float = 0.0
    score_disagreement_mae: float = 0.0
    winner_logit_disagreement_mae: float = 0.0


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
    "embedding",
    "endgame",
    "auto",
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
    "score_consistency",
    "score_ordering",
    "winner_consistency",
    "award_probe",
    "composed_score_rmse",
    "direct_composed_score_mae",
    "winner_logit_disagreement_mae",
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
        auto_targets: Tensor | None = None,
        official_total_targets: Tensor | None = None,
        dq_mask: Tensor | None = None,
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
        self.auto_targets = (
            auto_targets
            if auto_targets is not None
            else torch.empty((len(red_team_idx), 0), dtype=torch.long)
        )
        self.official_total_targets = (
            official_total_targets
            if official_total_targets is not None
            else torch.full((len(red_team_idx), 2), float("nan"), dtype=torch.float32)
        )
        self.dq_mask = (
            dq_mask if dq_mask is not None else torch.zeros(len(red_team_idx), dtype=torch.bool)
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
        if opts.study_arm != "none":
            v57_cont_columns = [
                *ordered_atomic_columns(standardized=True),
                *ordered_foul_columns(standardized=True),
            ]
            v57_bin_columns = ordered_bonus_columns()
        else:
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
        auto_targets = auto_target_matrix(table, opts)
        award_targets = award_target_tensor(table, opts)
        world_model_opts = world_model_opts or WorldModelOptions()
        wm_award_targets = world_award_target_tensor(table, world_model_opts.award_embedding.width)
        red_dq = (
            table["red_has_dq"].astype(bool)
            if "red_has_dq" in table
            else pd.Series(False, index=table.index)
        )
        blue_dq = (
            table["blue_has_dq"].astype(bool)
            if "blue_has_dq" in table
            else pd.Series(False, index=table.index)
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
            auto_targets=torch.as_tensor(auto_targets, dtype=torch.long),
            official_total_targets=torch.as_tensor(
                optional_target_matrix(table, ["red_total_score", "blue_total_score"]),
                dtype=torch.float32,
            ),
            dq_mask=torch.as_tensor((red_dq | blue_dq).to_numpy(copy=True), dtype=torch.bool),
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
            self.auto_targets[index],
            self.official_total_targets[index],
            self.dq_mask[index],
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


class AwardCandidateDataset(Dataset):
    """One explicit team-event candidate row per supported robot-award category."""

    def __init__(self, team_idx: Tensor, event_idx: Tensor, targets: Tensor) -> None:
        self.team_idx = team_idx
        self.event_idx = event_idx
        self.targets = targets

    @classmethod
    def from_table(cls, table: pd.DataFrame, categories: tuple[str, ...]) -> AwardCandidateDataset:
        columns = [f"award_{category}" for category in categories]
        if table is None or table.empty:
            values = np.empty((0, len(columns)), dtype=np.float32)
            teams = events = np.empty((0,), dtype=np.int64)
        else:
            missing = set(columns) - set(table)
            if missing:
                raise ValueError(
                    "Award candidate sidecar is missing: " + ", ".join(sorted(missing))
                )
            values = (
                table[columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32, copy=True)
            )
            teams = table["team_base_idx"].to_numpy(dtype=np.int64, copy=True)
            events = table.get("team_event_idx", pd.Series(0, index=table.index)).to_numpy(
                dtype=np.int64, copy=True
            )
        return cls(
            torch.as_tensor(teams, dtype=torch.long),
            torch.as_tensor(events, dtype=torch.long),
            torch.as_tensor(values, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return int(self.team_idx.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return self.team_idx[index], self.event_idx[index], self.targets[index]


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


def auto_target_matrix(table: pd.DataFrame, opts: LatentStratOptions) -> np.ndarray:
    mapping = {name: idx for idx, name in enumerate(opts.auto_class_order)}
    columns = _slot_columns("auto_status")
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
    normalized: bool = False,
) -> Tensor:
    if coefficient == 0 or not embedding.weight.requires_grad:
        return torch.zeros((), dtype=embedding.weight.dtype, device=indices.device)
    active = torch.unique(indices.flatten())
    active = active[active >= 0] if include_zero else active[active > 0]
    if active.numel() == 0:
        return torch.zeros((), dtype=embedding.weight.dtype, device=indices.device)
    values = embedding(active).pow(2)
    return coefficient * (values.mean() if normalized else values.sum())


def active_embedding_l2(
    model: SetTransformerModel,
    red_team_idx: Tensor,
    blue_team_idx: Tensor,
    coefficient: float,
    *,
    normalized: bool = False,
) -> Tensor:
    active_indices = torch.cat([red_team_idx, blue_team_idx], dim=1)
    return _active_embedding_l2(
        model.Z_base,
        active_indices,
        coefficient,
        include_zero=True,
        normalized=normalized,
    )


def active_event_embedding_l2(
    model: SetTransformerModel,
    red_event_idx: Tensor,
    blue_event_idx: Tensor,
    coefficient: float,
) -> Tensor:
    if not hasattr(model, "Z_event"):
        return torch.zeros((), dtype=model.Z_base.weight.dtype, device=red_event_idx.device)
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


def _masked_physical_mse(
    raw_prediction: Tensor,
    standardized_target: Tensor,
    mu: Tensor,
    sigma: Tensor,
) -> tuple[Tensor, bool]:
    """Compare nonnegative physical predictions using train-standardized residuals."""

    valid = torch.isfinite(standardized_target)
    if not torch.any(valid):
        return raw_prediction.sum() * 0.0, False
    physical_prediction = F.softplus(raw_prediction)
    physical_target = standardized_target * sigma + mu
    residual = (physical_prediction - physical_target) / sigma.clamp_min(1e-6)
    return torch.mean(residual[valid] ** 2), True


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
    auto_targets: Tensor | None = None,
    official_total_targets: Tensor | None = None,
    dq_mask: Tensor | None = None,
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
    predictions = pred.predictions
    zero = torch.zeros(
        (),
        dtype=predictions["continuous"].dtype,
        device=predictions["continuous"].device,
    )
    if cont_targets.numel() == 0:
        raw_cont_loss = zero
        cont_term = zero
    else:
        raw_cont_loss, cont_active = _masked_mse(predictions["continuous"], cont_targets)
        cont_term = model.balance_loss("continuous", raw_cont_loss, cont_active)
    if bin_targets.numel() == 0:
        raw_bin_loss = zero
        bin_term = zero
    else:
        finite = torch.isfinite(bin_targets)
        if torch.any(finite):
            raw_bin_loss = F.binary_cross_entropy_with_logits(
                predictions["win"][finite],
                bin_targets[finite],
                pos_weight=(positive_weights if positive_weights.numel() == 1 else None),
            )
            bin_term = model.balance_loss("win", raw_bin_loss, True)
        else:
            raw_bin_loss = zero
            bin_term = zero

    slot_missing = torch.cat([pred.masks.red_missing_mask, pred.masks.blue_missing_mask], dim=1)
    clean_rows = (
        torch.ones(slot_missing.shape[0], dtype=torch.bool, device=slot_missing.device)
        if dq_mask is None
        else ~dq_mask.bool()
    )
    if endgame_targets is None or predictions["endgame"].numel() == 0:
        raw_endgame_loss = zero
        endgame_term = zero
    else:
        valid = ~slot_missing & clean_rows.unsqueeze(1)
        if torch.any(valid):
            if opts.study_arm != "none":
                raw_endgame_loss = F.cross_entropy(
                    predictions["endgame"][valid], endgame_targets[valid]
                )
            else:
                cumulative = ordinal_targets_to_cumulative(
                    endgame_targets, model.num_endgame_classes
                )
                raw_endgame_loss = F.binary_cross_entropy_with_logits(
                    predictions["endgame"][valid], cumulative[valid]
                )
            endgame_term = model.balance_loss("endgame", raw_endgame_loss, True)
        else:
            raw_endgame_loss = zero
            endgame_term = zero

    if auto_targets is None or predictions["auto"].numel() == 0:
        raw_auto_loss = zero
        auto_term = zero
        auto_active = False
    else:
        valid = ~slot_missing & clean_rows.unsqueeze(1)
        if torch.any(valid):
            if opts.study_arm != "none":
                raw_auto_loss = F.cross_entropy(predictions["auto"][valid], auto_targets[valid])
            else:
                cumulative = ordinal_targets_to_cumulative(auto_targets, len(opts.auto_class_order))
                raw_auto_loss = F.binary_cross_entropy_with_logits(
                    predictions["auto"][valid], cumulative[valid]
                )
            auto_term = model.balance_loss("auto", raw_auto_loss, True)
            auto_active = True
        else:
            raw_auto_loss = zero
            auto_term = zero
            auto_active = False

    if opts.study_arm != "none" or award_targets is None or predictions["award"].numel() == 0:
        raw_award_loss = zero
        award_term = zero
    else:
        finite = torch.isfinite(award_targets) & ~slot_missing.unsqueeze(-1)
        if torch.any(finite):
            raw_award_loss = F.binary_cross_entropy_with_logits(
                predictions["award"][finite], award_targets[finite]
            )
            award_term = model.balance_loss("awards", raw_award_loss, True)
        else:
            raw_award_loss = zero
            award_term = zero

    atomic_targets, foul_targets = (
        _split_v57_targets(v57_cont_targets, opts, continuous=True)
        if v57_cont_targets is not None
        else (
            zero.new_empty((predictions["continuous"].shape[0], 0)),
            zero.new_empty((predictions["continuous"].shape[0], 0)),
        )
    )
    bonus_targets, special_targets = (
        _split_v57_targets(v57_bin_targets, opts, continuous=False)
        if v57_bin_targets is not None
        else (
            zero.new_empty((predictions["continuous"].shape[0], 0)),
            zero.new_empty((predictions["continuous"].shape[0], 0)),
        )
    )
    if atomic_targets.numel() and predictions["atomic"].numel():
        if opts.study_arm != "none":
            raw_atomic_loss, atomic_active = _masked_physical_mse(
                predictions["atomic"],
                atomic_targets,
                model.physics_atomic_mu,
                model.physics_atomic_sigma,
            )
        else:
            raw_atomic_loss, atomic_active = _masked_mse(predictions["atomic"], atomic_targets)
    else:
        raw_atomic_loss, atomic_active = zero, False
    atomic_term = model.balance_loss("atomic", raw_atomic_loss, atomic_active)
    if foul_targets.numel() and predictions["foul"].numel():
        if opts.study_arm != "none":
            raw_foul_loss, foul_active = _masked_physical_mse(
                predictions["foul"],
                foul_targets,
                model.physics_foul_mu,
                model.physics_foul_sigma,
            )
        else:
            raw_foul_loss, foul_active = _masked_mse(predictions["foul"], foul_targets)
    else:
        raw_foul_loss, foul_active = zero, False
    foul_term = model.balance_loss("foul", raw_foul_loss, foul_active)
    if bonus_targets.numel() and predictions["bonus"].numel():
        raw_bonus_loss, bonus_active = _masked_bce_with_logits(predictions["bonus"], bonus_targets)
    else:
        raw_bonus_loss, bonus_active = zero, False
    bonus_term = model.balance_loss("bonus", raw_bonus_loss, bonus_active)
    if special_targets.numel() and predictions["special"].numel():
        raw_special_loss, special_active = _masked_bce_with_logits(
            predictions["special"], special_targets
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
                    predictions["wm_award"][valid_awards],
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

    raw_score_consistency = zero
    raw_score_ordering = zero
    raw_winner_consistency = zero
    composed_score_rmse = math.nan
    score_disagreement_mae = math.nan
    winner_logit_disagreement_mae = math.nan
    score_consistency_term = zero
    score_ordering_term = zero
    winner_consistency_term = zero
    consistency_active = opts.study_arm in {"physics-consistent", "full-structured"}
    if consistency_active:
        composed_scores = compose_predicted_scores(
            predictions["atomic"],
            predictions["foul"],
            predictions["auto"],
            predictions["endgame"],
        )
        direct_scores = (
            predictions["continuous"] * model.physics_cont_sigma[:2] + model.physics_cont_mu[:2]
        )
        clean = (
            torch.ones(direct_scores.shape[0], dtype=torch.bool, device=direct_scores.device)
            if dq_mask is None
            else ~dq_mask.bool()
        )
        score_valid = (
            clean.unsqueeze(1) & torch.isfinite(direct_scores) & torch.isfinite(composed_scores)
        )
        if torch.any(score_valid):
            score_residual = (
                direct_scores - composed_scores
            ) / model.physics_score_sigma.clamp_min(1e-6)
            raw_score_consistency = F.smooth_l1_loss(
                score_residual[score_valid], torch.zeros_like(score_residual[score_valid]), beta=1.0
            )
            score_consistency_term = (
                float(opts.score_consistency_loss_weight) * raw_score_consistency
            )
            score_disagreement_mae = float(
                torch.mean(torch.abs(direct_scores[score_valid] - composed_scores[score_valid]))
                .detach()
                .cpu()
            )
        score_logit = model.physics_score_logit_alpha * (direct_scores[:, 0] - direct_scores[:, 1])
        winner_logit = predictions["win"][:, 0]
        winner_target = (
            bin_targets[:, 0] if bin_targets.shape[1] else zero.new_full((len(clean),), math.nan)
        )
        winner_valid = clean & torch.isfinite(winner_target)
        if torch.any(winner_valid):
            raw_score_ordering = F.binary_cross_entropy_with_logits(
                score_logit[winner_valid], winner_target[winner_valid]
            )
            raw_winner_consistency = F.smooth_l1_loss(
                ((winner_logit - score_logit) / model.physics_score_logit_sigma.clamp_min(1e-6))[
                    winner_valid
                ],
                torch.zeros_like(winner_logit[winner_valid]),
                beta=1.0,
            )
            score_ordering_term = float(opts.score_ordering_loss_weight) * raw_score_ordering
            winner_consistency_term = (
                float(opts.winner_consistency_loss_weight) * raw_winner_consistency
            )
            winner_logit_disagreement_mae = float(
                torch.mean(torch.abs(winner_logit[winner_valid] - score_logit[winner_valid]))
                .detach()
                .cpu()
            )
        if official_total_targets is not None:
            official_valid = clean.unsqueeze(1) & torch.isfinite(official_total_targets)
            if torch.any(official_valid):
                composed_score_rmse = float(
                    torch.sqrt(
                        torch.mean(
                            (
                                composed_scores[official_valid]
                                - official_total_targets[official_valid]
                            )
                            ** 2
                        )
                    )
                    .detach()
                    .cpu()
                )

    emb_l2 = active_embedding_l2(
        model,
        red_team_idx,
        blue_team_idx,
        opts.embedding_loss_weight if opts.core_objective else opts.l2_embedding,
        normalized=opts.core_objective,
    )
    event_l2 = zero
    if red_event_idx is not None and blue_event_idx is not None:
        event_l2 = active_event_embedding_l2(
            model, red_event_idx, blue_event_idx, opts.event_delta_l2
        )
    total = (
        cont_term
        + bin_term
        + endgame_term
        + auto_term
        + award_term
        + atomic_term
        + foul_term
        + bonus_term
        + special_term
        + wm_award_term
        + score_consistency_term
        + score_ordering_term
        + winner_consistency_term
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
        auto_loss=float(raw_auto_loss.detach().cpu()) if auto_active else math.nan,
        wm_award_loss=float(raw_wm_award_loss.detach().cpu()),
        score_consistency_loss=float(raw_score_consistency.detach().cpu()),
        score_ordering_loss=float(raw_score_ordering.detach().cpu()),
        winner_consistency_loss=float(raw_winner_consistency.detach().cpu()),
        composed_score_rmse=composed_score_rmse,
        score_disagreement_mae=score_disagreement_mae,
        winner_logit_disagreement_mae=winner_logit_disagreement_mae,
    )
    return total, metrics, pred


def create_optimizer(model: SetTransformerModel, opts: LatentStratOptions) -> torch.optim.AdamW:
    return torch.optim.AdamW(optimizer_parameter_groups(model, opts), lr=opts.learning_rate)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, opts: LatentStratOptions
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Compatibility helper for callers that do not have a loader length."""

    return build_lr_scheduler(
        optimizer,
        _runtime_config(opts),
        epochs=opts.epochs,
        microbatches_per_epoch=1,
    )


def _runtime_config(opts: LatentStratOptions) -> TrainingRuntimeConfig:
    return TrainingRuntimeConfig(
        gradient_accumulation_steps=opts.gradient_accumulation_steps,
        max_grad_norm=opts.max_grad_norm,
        scheduler=opts.scheduler if opts.use_lr_scheduler else "none",
        lr_eta_min=opts.lr_eta_min,
        one_cycle_pct_start=opts.one_cycle_pct_start,
        one_cycle_div_factor=opts.one_cycle_div_factor,
        one_cycle_final_div_factor=opts.one_cycle_final_div_factor,
        deterministic_algorithms=opts.deterministic_algorithms,
        checkpoint_every_epochs=opts.checkpoint_every_epochs,
        optimizer_log_interval=opts.optimizer_log_interval,
        log_optimizer_steps=opts.log_optimizer_steps,
    )


def clamp_loss_log_vars(model: SetTransformerModel, opts: LatentStratOptions) -> None:
    model.loss_balancer.clamp_(opts.loss_log_var_min, opts.loss_log_var_max)


def create_tensorboard_writer(log_dir: str, *, purge_step: int | None = None) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled, but tensorboard is not installed. "
            'Run `pip install -e ".[dev]"` or pass `--no-tensorboard`.'
        ) from exc
    return SummaryWriter(str(log_dir), purge_step=purge_step, flush_secs=30)


def freeze_for_venue_mode(model: SetTransformerModel) -> None:
    if not hasattr(model, "Z_event"):
        raise ValueError("venue_mode is unavailable for static-z-base models.")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.Z_event.weight.requires_grad = True


def freeze_team_embedding_tables(model: SetTransformerModel) -> None:
    model.Z_base.weight.requires_grad = False
    if hasattr(model, "Z_event"):
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
        worker_init_fn=seed_dataloader_worker if opts.dataloader_num_workers else None,
        persistent_workers=opts.dataloader_num_workers > 0,
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
    ) = batch[:13]
    auto = batch[13] if len(batch) > 13 else None
    official_total = batch[14] if len(batch) > 14 else None
    dq_mask = batch[15] if len(batch) > 15 else None
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
        auto_targets=auto,
        official_total_targets=official_total,
        dq_mask=dq_mask,
    )


def _rank_loss_from_batch(
    model: SetTransformerModel, batch: tuple[Tensor, ...], opts: LatentStratOptions
) -> Tensor:
    higher, lower, higher_event, lower_event = batch
    higher_score = model.team_value(higher, higher_event)
    lower_score = model.team_value(lower, lower_event)
    target = torch.ones_like(higher_score)
    return F.margin_ranking_loss(higher_score, lower_score, target, margin=float(opts.rank_margin))


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


def _award_probe_loss_from_batch(model: SetTransformerModel, batch: tuple[Tensor, ...]) -> Tensor:
    team, event, targets = batch
    logits = model.award_probe(team, event)
    finite = torch.isfinite(targets)
    if not torch.any(finite):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[finite], targets[finite])


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
        "embedding": metrics.embedding_l2_loss,
        "endgame": metrics.endgame_loss,
        "auto": metrics.auto_loss,
        "awards": metrics.award_loss,
        "atomic": metrics.atomic_loss,
        "foul": metrics.foul_loss,
        "bonus": metrics.bonus_loss,
        "special": metrics.special_loss,
        "wm_award": metrics.wm_award_loss,
        "score_consistency": metrics.score_consistency_loss,
        "score_ordering": metrics.score_ordering_loss,
        "winner_consistency": metrics.winner_consistency_loss,
        "composed_score_rmse": metrics.composed_score_rmse,
        "direct_composed_score_mae": metrics.score_disagreement_mae,
        "winner_logit_disagreement_mae": metrics.winner_logit_disagreement_mae,
    }


def _core_batch_task_weights(
    batch: tuple[Tensor, ...], opts: LatentStratOptions | None = None
) -> dict[str, float]:
    """Count rows that actually contribute to each fixed core objective."""

    continuous = batch[6]
    binary = batch[7]
    opts = opts or default_options().model_copy(update={"core_objective": True})
    counts = {
        "continuous": float(torch.isfinite(continuous).any(dim=1).sum().item()),
        "win": float(torch.isfinite(binary).any(dim=1).sum().item()),
        "embedding": float(batch[0].shape[0]),
    }
    if opts.study_arm != "none":
        missing = torch.cat([batch[4], batch[5]], dim=1)
        v57_cont = batch[10]
        v57_bin = batch[11]
        atomic_width = 2 * len(opts.atomic_count_targets)
        counts.update(
            {
                "atomic": float(torch.isfinite(v57_cont[:, :atomic_width]).sum().item()),
                "foul": float(torch.isfinite(v57_cont[:, atomic_width:]).sum().item()),
                "bonus": float(torch.isfinite(v57_bin).sum().item()),
                "auto": float((~missing).sum().item()),
                "endgame": float((~missing).sum().item()),
                "score_consistency": float(torch.isfinite(batch[14]).all(dim=1).sum().item()),
                "score_ordering": counts["win"],
                "winner_consistency": counts["win"],
                "composed_score_rmse": float(torch.isfinite(batch[14]).all(dim=1).sum().item()),
                "direct_composed_score_mae": float(
                    torch.isfinite(batch[14]).all(dim=1).sum().item()
                ),
                "winner_logit_disagreement_mae": counts["win"],
            }
        )
    return counts


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
        name: (sums[name] / weights[name] if weights[name] > 0 else math.nan) for name in TASK_NAMES
    }


def _write_feature_tensorboard_epoch(
    writer: Any,
    *,
    model: SetTransformerModel,
    row: dict[str, float | int],
    task_means: dict[str, float],
    task_counts: dict[str, float],
    validation_task_means: dict[str, float],
    validation_task_counts: dict[str, float],
    active_sidecars: set[str],
) -> None:
    epoch = int(row["epoch"])
    if model.opts.core_objective:
        writer.add_scalar("Loss/Train", float(row["train_loss"]), epoch)
        validation = float(row["validation_loss"])
        if math.isfinite(validation):
            writer.add_scalar("Loss/Validation", validation, epoch)
        component_tasks = (
            ("Score", "continuous"),
            ("Win", "win"),
            ("Embedding", "embedding"),
            ("AtomicCounts", "atomic"),
            ("AutoStatus", "auto"),
            ("EndgameStatus", "endgame"),
            ("Fouls", "foul"),
            ("Bonuses", "bonus"),
            ("ScoreConsistency", "score_consistency"),
            ("ScoreOrdering", "score_ordering"),
            ("WinnerConsistency", "winner_consistency"),
            ("CompetitionRank", "rank"),
            ("CompetitionPlayoff", "playoff"),
            ("AwardProbe", "award_probe"),
        )
        for label, task in component_tasks:
            value = float(task_means.get(task, math.nan))
            if math.isfinite(value):
                writer.add_scalar(f"LossComponent/{label}", value, epoch)
            validation_value = float(validation_task_means.get(task, math.nan))
            if math.isfinite(validation_value):
                writer.add_scalar(f"ValidationLossComponent/{label}", validation_value, epoch)
            count = float(task_counts.get(task, 0.0))
            if count > 0 or task in active_sidecars:
                writer.add_scalar(f"ActiveLabelCount/{label}", count, epoch)
        breakdown_values = [
            float(task_means.get(task, math.nan))
            for task in ("atomic", "auto", "endgame", "foul", "bonus")
        ]
        active_breakdown = [value for value in breakdown_values if math.isfinite(value)]
        if active_breakdown:
            writer.add_scalar(
                "LossComponent/BreakdownMacro", float(np.mean(active_breakdown)), epoch
            )
        diagnostics = {
            "Physics/ComposedScoreRMSE": "composed_score_rmse",
            "Physics/DirectComposedScoreMAE": "direct_composed_score_mae",
            "Physics/WinnerLogitDisagreementMAE": "winner_logit_disagreement_mae",
        }
        for tag, task in diagnostics.items():
            value = float(task_means.get(task, math.nan))
            if math.isfinite(value):
                writer.add_scalar(tag, value, epoch)
        fields = {
            "Forecast/Train/ScoreMAE": "train_score_mae",
            "Forecast/Train/ScoreRMSE": "train_score_rmse",
            "Forecast/Train/ScoreDifferentialMAE": "train_score_differential_mae",
            "Forecast/Train/ScoreDifferentialRMSE": "train_score_differential_rmse",
            "Forecast/Train/WinnerBrier": "train_winner_brier",
            "Forecast/Train/WinnerLogLoss": "train_winner_log_loss",
            "Forecast/Train/WinnerECE": "train_winner_ece",
            "Forecast/Validation/ScoreMAE": "validation_score_mae",
            "Forecast/Validation/ScoreRMSE": "validation_score_rmse",
            "Forecast/Validation/ScoreDifferentialMAE": "validation_score_differential_mae",
            "Forecast/Validation/ScoreDifferentialRMSE": "validation_score_differential_rmse",
            "Forecast/Validation/WinnerBrier": "validation_winner_brier",
            "Forecast/Validation/WinnerLogLoss": "validation_winner_log_loss",
            "Forecast/Validation/WinnerECE": "validation_winner_ece",
            "Optimization/LearningRate": "learning_rate_group_0",
            "Optimization/GradientNormMean": "preclip_grad_norm_mean",
            "Optimization/GradientNormMax": "preclip_grad_norm_max",
            "Optimization/ClippingFraction": "clipping_fraction",
            "Runtime/EpochSeconds": "epoch_seconds",
            "Runtime/SamplesPerSecond": "samples_per_second",
            "State/ZBaseNormMean": "z_base_norm_mean",
            "State/ZBaseUpdateMean": "z_base_update_norm_mean",
        }
        for tag, field in fields.items():
            value = float(row.get(field, math.nan))
            if math.isfinite(value):
                writer.add_scalar(tag, value, epoch)
        return
    writer.add_scalar("Loss/Train/Total", float(row["train_loss"]), epoch)
    writer.add_scalar("Loss/Train_Total", float(row["train_loss"]), epoch)
    validation = float(row["validation_loss"])
    if math.isfinite(validation):
        writer.add_scalar("Loss/Validation/Total", validation, epoch)
        writer.add_scalar("Loss/Validation_Total", validation, epoch)
    for key, raw_value in row.items():
        if not key.startswith("learning_rate_group_"):
            continue
        value = float(raw_value)
        if math.isfinite(value):
            writer.add_scalar(
                f"LearningRate/group_{key.removeprefix('learning_rate_group_')}",
                value,
                epoch,
            )
            if key == "learning_rate_group_0":
                writer.add_scalar("LR/base", value, epoch)
    for task_name, value in task_means.items():
        if math.isfinite(value):
            writer.add_scalar(f"LossRaw/{task_name}", value, epoch)
            writer.add_scalar(f"Loss/Train/{task_name}", value, epoch)
        validation_value = validation_task_means.get(task_name, math.nan)
        if math.isfinite(validation_value):
            writer.add_scalar(f"Loss/Validation/{task_name}", validation_value, epoch)
        writer.add_scalar(
            f"Active/{task_name}",
            1.0 if (math.isfinite(value) or task_name in active_sidecars) else 0.0,
            epoch,
        )
        writer.add_scalar(f"ActiveCount/Train/{task_name}", task_counts.get(task_name, 0), epoch)
        writer.add_scalar(
            f"ActiveCount/Validation/{task_name}",
            validation_task_counts.get(task_name, 0),
            epoch,
        )
    for task_name, parameter in model.loss_balancer.log_vars.items():
        log_var = float(parameter.detach().cpu())
        writer.add_scalar(f"LogVar/{task_name}", log_var, epoch)
        writer.add_scalar(f"Weights/{task_name}_precision", math.exp(-log_var), epoch)
    scalar_fields = {
        "Runtime/EpochSeconds": "epoch_seconds",
        "Runtime/SamplesPerSecond": "samples_per_second",
        "Runtime/OptimizerSteps": "optimizer_steps",
        "Optimization/Season/PreclipGradientNormMean": "preclip_grad_norm_mean",
        "Optimization/Season/PreclipGradientNormMax": "preclip_grad_norm_max",
        "Optimization/Season/ClippingFraction": "clipping_fraction",
        "Parameters/Nominal": "nominal_parameter_count",
        "Parameters/Trainable": "trainable_parameter_count",
        "Parameters/GradientReceiving": "gradient_receiving_parameter_count",
        "Data/TrainRows": "train_rows",
        "Data/ValidationRows": "validation_rows",
        "Data/TrainEvents": "train_events",
        "Data/ValidationEvents": "validation_events",
        "Data/ActiveTeams": "active_team_rows",
        "State/ZBase/ActiveRows": "active_team_rows",
        "State/ZBase/NormMean": "z_base_norm_mean",
        "State/ZBase/NormStd": "z_base_norm_std",
        "State/ZBase/NormP05": "z_base_norm_p05",
        "State/ZBase/NormP50": "z_base_norm_p50",
        "State/ZBase/NormP95": "z_base_norm_p95",
        "State/ZBase/GhostNorm": "z_base_ghost_norm",
        "State/ZBase/UpdateNormMean": "z_base_update_norm_mean",
        "Metrics/Train/ScoreMAE": "train_score_mae",
        "Metrics/Train/ScoreRMSE": "train_score_rmse",
        "Metrics/Train/ScoreDifferentialMAE": "train_score_differential_mae",
        "Metrics/Train/ScoreDifferentialRMSE": "train_score_differential_rmse",
        "Metrics/Train/WinnerAccuracy": "train_winner_accuracy",
        "Metrics/Train/WinnerBrier": "train_winner_brier",
        "Metrics/Train/WinnerLogLoss": "train_winner_log_loss",
        "Metrics/Train/WinnerECE": "train_winner_ece",
        "Metrics/Validation/ScoreMAE": "validation_score_mae",
        "Metrics/Validation/ScoreRMSE": "validation_score_rmse",
        "Metrics/Validation/ScoreDifferentialMAE": "validation_score_differential_mae",
        "Metrics/Validation/ScoreDifferentialRMSE": "validation_score_differential_rmse",
        "Metrics/Validation/WinnerAccuracy": "validation_winner_accuracy",
        "Metrics/Validation/WinnerBrier": "validation_winner_brier",
        "Metrics/Validation/WinnerLogLoss": "validation_winner_log_loss",
        "Metrics/Validation/WinnerECE": "validation_winner_ece",
    }
    for tag, field in scalar_fields.items():
        value = float(row.get(field, math.nan))
        if math.isfinite(value):
            writer.add_scalar(tag, value, epoch)


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
        worker_init_fn=seed_dataloader_worker if opts.dataloader_num_workers else None,
        persistent_workers=opts.dataloader_num_workers > 0,
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
    if "award_candidates" in sidecar_tables:
        datasets["award_probe"] = AwardCandidateDataset.from_table(
            sidecar_tables["award_candidates"], opts.award_targets
        )
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
                loss, _, _ = _loss_from_batch(model, batch, opts, positive_weights, active_model)
            batch_size = int(batch[0].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_rows += batch_size
    if was_training:
        model.train()
        active_model.train()
    return total_loss / max(total_rows, 1)


def evaluate_task_means(
    model: SetTransformerModel,
    data_loader: DataLoader,
    opts: LatentStratOptions,
    positive_weights: Tensor,
    device: torch.device,
    *,
    forward_model: torch.nn.Module | None = None,
    amp_enabled: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    sums, weights = _empty_epoch_accumulators()
    was_training = model.training
    model.eval()
    active_model = forward_model or model
    active_model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            batch = batch_to_device(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                _, metrics, _ = _loss_from_batch(model, batch, opts, positive_weights, active_model)
            batch_rows = int(batch[0].shape[0])
            core_weights = _core_batch_task_weights(batch, opts) if opts.core_objective else None
            for task_name, value in _metrics_to_losses(metrics).items():
                weight = (
                    core_weights.get(task_name, 0.0) if core_weights is not None else batch_rows
                )
                _accumulate_loss(sums, weights, task_name, value, weight)
    if was_training:
        model.train()
        active_model.train()
    return _epoch_task_means(sums, weights), weights


def evaluate_forecast_metrics(
    model: SetTransformerModel,
    data_loader: DataLoader,
    device: torch.device,
    target_mu: Tensor,
    target_sigma: Tensor,
    opts: LatentStratOptions,
    *,
    score_residual: float = 0.0,
    forward_model: torch.nn.Module | None = None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    predicted_scores = []
    actual_scores = []
    probabilities = []
    outcomes = []
    was_training = model.training
    model.eval()
    active_model = forward_model or model
    active_model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            batch = batch_to_device(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                output = active_model(
                    batch[0],
                    batch[1],
                    red_event_idx=batch[2],
                    blue_event_idx=batch[3],
                    red_missing_mask=batch[4],
                    blue_missing_mask=batch[5],
                )
            continuous = output.cont_z * target_sigma + target_mu
            actual_continuous = batch[6] * target_sigma + target_mu
            if opts.score_target_mode == "official-total-core" and continuous.shape[1] >= 2:
                predicted_scores.append(continuous[:, :2].cpu())
                official = batch[14].clone()
                official[batch[15].bool()] = float("nan")
                actual_scores.append(official.cpu())
            elif continuous.shape[1] >= 4:
                predicted_scores.append(
                    torch.stack(
                        [
                            continuous[:, :2].sum(dim=1) + score_residual,
                            continuous[:, 2:4].sum(dim=1) + score_residual,
                        ],
                        dim=1,
                    ).cpu()
                )
                if opts.core_objective:
                    official = batch[14].clone()
                    official[batch[15].bool()] = float("nan")
                    actual_scores.append(official.cpu())
                else:
                    actual_scores.append(
                        torch.stack(
                            [
                                actual_continuous[:, :2].sum(dim=1),
                                actual_continuous[:, 2:4].sum(dim=1),
                            ],
                            dim=1,
                        ).cpu()
                    )
            if output.bin_logits.shape[1] and batch[7].shape[1]:
                probabilities.append(torch.sigmoid(output.bin_logits[:, 0]).cpu())
                outcomes.append(batch[7][:, 0].cpu())
    if was_training:
        model.train()
        active_model.train()
    result = {
        "score_mae": math.nan,
        "score_rmse": math.nan,
        "score_differential_mae": math.nan,
        "score_differential_rmse": math.nan,
        "winner_accuracy": math.nan,
        "winner_brier": math.nan,
        "winner_log_loss": math.nan,
        "winner_ece": math.nan,
    }
    if predicted_scores:
        predicted = torch.cat(predicted_scores).numpy()
        actual = torch.cat(actual_scores).numpy()
        valid = np.isfinite(predicted) & np.isfinite(actual)
        errors = predicted - actual
        if valid.any():
            result["score_mae"] = float(np.mean(np.abs(errors[valid])))
            result["score_rmse"] = float(np.sqrt(np.mean(errors[valid] ** 2)))
        predicted_difference = predicted[:, 0] - predicted[:, 1]
        actual_difference = actual[:, 0] - actual[:, 1]
        diff_valid = np.isfinite(predicted_difference) & np.isfinite(actual_difference)
        if diff_valid.any():
            diff_error = predicted_difference[diff_valid] - actual_difference[diff_valid]
            result["score_differential_mae"] = float(np.mean(np.abs(diff_error)))
            result["score_differential_rmse"] = float(np.sqrt(np.mean(diff_error**2)))
    if probabilities:
        probability = torch.cat(probabilities).numpy()
        outcome = torch.cat(outcomes).numpy()
        valid = np.isfinite(probability) & np.isfinite(outcome)
        if valid.any():
            probability = np.clip(probability[valid], 1e-7, 1 - 1e-7)
            outcome = outcome[valid]
            result["winner_accuracy"] = float(np.mean((probability >= 0.5) == (outcome >= 0.5)))
            result["winner_brier"] = float(np.mean((probability - outcome) ** 2))
            result["winner_log_loss"] = float(
                np.mean(-(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability)))
            )
            edges = np.linspace(0, 1, 11)
            bin_index = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, 9)
            result["winner_ece"] = float(
                sum(
                    float((bin_index == index).mean())
                    * abs(
                        float(probability[bin_index == index].mean())
                        - float(outcome[bin_index == index].mean())
                    )
                    for index in range(10)
                    if np.any(bin_index == index)
                )
            )
    return result


def _z_base_statistics(
    model: SetTransformerModel,
    active_indices: Tensor,
    initial_z_base: Tensor,
) -> dict[str, float | int]:
    active = torch.unique(active_indices.flatten()).long()
    active = active[(active >= 0) & (active < model.Z_base.num_embeddings)]
    values = model.Z_base.weight.detach()[active]
    initial = initial_z_base.to(values.device)[active]
    norms = torch.linalg.vector_norm(values, dim=1)
    updates = torch.linalg.vector_norm(values - initial, dim=1)
    quantiles = torch.quantile(norms.float(), torch.tensor([0.05, 0.5, 0.95], device=norms.device))
    return {
        "active_team_rows": int(active.numel()),
        "z_base_norm_mean": float(norms.mean().cpu()),
        "z_base_norm_std": float(norms.std(unbiased=False).cpu()),
        "z_base_norm_p05": float(quantiles[0].cpu()),
        "z_base_norm_p50": float(quantiles[1].cpu()),
        "z_base_norm_p95": float(quantiles[2].cpu()),
        "z_base_ghost_norm": float(torch.linalg.vector_norm(model.Z_base.weight[0].detach()).cpu()),
        "z_base_update_norm_mean": float(updates.mean().cpu()),
    }


def _write_parameter_histograms(
    writer: Any,
    model: SetTransformerModel,
    active_indices: Tensor,
    initial_z_base: Tensor,
    *,
    step: int,
    gradients: dict[str, Tensor] | None = None,
) -> None:
    if not hasattr(writer, "add_histogram"):
        return
    active = torch.unique(active_indices.flatten()).long()
    active = active[(active >= 0) & (active < model.Z_base.num_embeddings)]
    values = model.Z_base.weight.detach()[active].cpu()
    initial = initial_z_base[active.cpu()].cpu()
    writer.add_histogram("State/ZBase/ActiveNorms", torch.linalg.vector_norm(values, dim=1), step)
    writer.add_histogram(
        "State/ZBase/UpdateNorms",
        torch.linalg.vector_norm(values - initial, dim=1),
        step,
    )
    for module_name in ("sab", "cross", "pma", "cont_head", "bin_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            writer.add_histogram(
                f"Parameters/{module_name}/{name.replace('.', '/')}",
                parameter.detach().cpu(),
                step,
            )
    for name, gradient in (gradients or {}).items():
        writer.add_histogram(f"Gradients/{name.replace('.', '/')}", gradient.detach().cpu(), step)


def _parameter_utilization_by_module(
    model: SetTransformerModel, gradient_receiving_names: set[str]
) -> dict[str, dict[str, int]]:
    modules: dict[str, dict[str, int]] = {}
    head_prefixes = {
        "cont_head",
        "atomic_head",
        "foul_head",
        "bonus_head",
        "special_head",
        "bin_head",
        "endgame_head",
        "award_head",
        "team_value_head",
        "alliance_value_head",
    }
    for name, parameter in model.named_parameters():
        prefix = name.split(".", 1)[0]
        module = "prediction_heads" if prefix in head_prefixes else prefix
        row = modules.setdefault(module, {"nominal": 0, "trainable": 0, "gradient_receiving": 0})
        row["nominal"] += parameter.numel()
        if parameter.requires_grad:
            row["trainable"] += parameter.numel()
        if name in gradient_receiving_names:
            row["gradient_receiving"] += parameter.numel()
    return modules


def _training_source_fingerprint(table: pd.DataFrame, split: Split) -> str:
    digest = hashlib.sha256()
    stable_table = table.copy()
    for column in stable_table.select_dtypes(include=["object", "str"]):
        stable_table[column] = stable_table[column].map(repr)
    digest.update(pd.util.hash_pandas_object(stable_table, index=True).values.tobytes())
    digest.update(np.asarray(split.train_mask, dtype=np.bool_).tobytes())
    digest.update(np.asarray(split.validation_mask, dtype=np.bool_).tobytes())
    digest.update(str(split.policy).encode("utf-8"))
    return digest.hexdigest()


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
    resume_checkpoint: str | None = None,
    resume_output: str | None = None,
) -> tuple[SetTransformerModel, pd.DataFrame, TrainingDiagnostics]:
    opts = opts or default_options()
    world_model_opts = world_model_opts or (
        initial_model.world_model_opts if initial_model is not None else WorldModelOptions()
    )
    if resume_checkpoint is not None and initial_model is not None:
        raise ValueError("resume_checkpoint cannot be combined with initial_model warm-starting.")
    runtime_config = _runtime_config(opts)
    seed_everything(
        opts.random_seed, deterministic_algorithms=runtime_config.deterministic_algorithms
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
    clamp_loss_log_vars(model, opts)
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
    sidecar_loaders = _prepare_sidecar_loaders(
        sidecar_tables,
        opts,
        generator=generator,
        device=device,
        world_model_opts=world_model_opts,
    )
    train_eval_loader = _make_loader(dataset, train_rows, opts, shuffle=False, device=device)
    validation_loader = _make_loader(dataset, validation_rows, opts, shuffle=False, device=device)
    target_names = [mapping.target_name for mapping in opts.target_map]
    if all(name in table.columns for name in target_names):
        raw_targets = table.iloc[train_rows][target_names].apply(pd.to_numeric, errors="coerce")
        target_mu_np = raw_targets.mean(axis=0).to_numpy(float, copy=True)
        target_sigma_np = raw_targets.std(axis=0, ddof=0).to_numpy(float, copy=True)
        target_sigma_np[~np.isfinite(target_sigma_np) | (target_sigma_np == 0)] = 1.0
    else:
        target_mu_np = np.zeros(len(target_names), dtype=float)
        target_sigma_np = np.ones(len(target_names), dtype=float)
    target_mu = torch.as_tensor(target_mu_np, dtype=torch.float32, device=device)
    target_sigma = torch.as_tensor(target_sigma_np, dtype=torch.float32, device=device)
    if opts.study_arm != "none":
        if len(target_mu) != 2:
            raise ValueError("Physics study requires exactly red/blue direct score targets.")

        def physical_stats(columns: list[str]) -> tuple[Tensor, Tensor]:
            values = table.iloc[train_rows][columns].apply(pd.to_numeric, errors="coerce")
            mu = values.mean(axis=0).to_numpy(float, copy=True)
            sigma = values.std(axis=0, ddof=0).to_numpy(float, copy=True)
            sigma[~np.isfinite(sigma) | (sigma == 0)] = 1.0
            if not np.isfinite(mu).all():
                raise ValueError("Physics training target statistics contain non-finite means.")
            return (
                torch.as_tensor(mu, dtype=torch.float32, device=device),
                torch.as_tensor(sigma, dtype=torch.float32, device=device),
            )

        atomic_mu, atomic_sigma = (
            physical_stats(ordered_atomic_columns())
            if model.physics_atomic_mu.numel()
            else (model.physics_atomic_mu, model.physics_atomic_sigma)
        )
        foul_mu, foul_sigma = (
            physical_stats(ordered_foul_columns())
            if model.physics_foul_mu.numel()
            else (model.physics_foul_mu, model.physics_foul_sigma)
        )
        score_values = table.iloc[train_rows][["red_total_score", "blue_total_score"]]
        score_values = score_values.apply(pd.to_numeric, errors="coerce").to_numpy(float).ravel()
        score_scale = float(np.nanstd(score_values))
        if not math.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("Physics score standard deviation is invalid.")
        with torch.no_grad():
            model.physics_cont_mu[:2].copy_(target_mu)
            model.physics_cont_sigma[:2].copy_(target_sigma)
            if model.physics_atomic_mu.numel():
                model.physics_atomic_mu.copy_(atomic_mu)
                model.physics_atomic_sigma.copy_(atomic_sigma)
            if model.physics_foul_mu.numel():
                model.physics_foul_mu.copy_(foul_mu)
                model.physics_foul_sigma.copy_(foul_sigma)
            model.physics_score_sigma.fill_(score_scale)
            if resume_checkpoint is None:

                def inverse_softplus(values: Tensor) -> Tensor:
                    values = values.clamp_min(1e-3)
                    return torch.where(values > 20, values, torch.log(torch.expm1(values)))

                if model.atomic_head is not None and atomic_mu.numel():
                    alliance_atomic_mu = (atomic_mu[:7] + atomic_mu[7:]) / 2
                    model.atomic_head.bias.copy_(inverse_softplus(alliance_atomic_mu))
                if model.foul_head is not None and foul_mu.numel():
                    alliance_foul_mu = (foul_mu[:2] + foul_mu[2:]) / 2
                    model.foul_head.bias.copy_(inverse_softplus(alliance_foul_mu))
    score_residual = 0.0
    if opts.score_target_mode == "phase-core":
        eligible = ~(
            table.iloc[train_rows].get("red_has_dq", False).astype(bool).to_numpy()
            | table.iloc[train_rows].get("blue_has_dq", False).astype(bool).to_numpy()
        )
        source = table.iloc[train_rows]
        red = source["red_total_score"].to_numpy(float) - (
            source["red_auto_pts"].to_numpy(float) + source["red_teleop_pts"].to_numpy(float)
        )
        blue = source["blue_total_score"].to_numpy(float) - (
            source["blue_auto_pts"].to_numpy(float) + source["blue_teleop_pts"].to_numpy(float)
        )
        score_residual = float(np.nanmean(np.concatenate([red[eligible], blue[eligible]])))

    scheduler = build_lr_scheduler(
        optimizer,
        runtime_config,
        epochs=opts.epochs,
        microbatches_per_epoch=len(train_loader),
    )
    controller = OptimizationController(
        model,
        optimizer,
        runtime_config,
        scheduler=scheduler,
        scaler=scaler,
        tensorboard_writer=tensorboard_writer,
        tensorboard_prefix="Optimization/Season",
        after_step=lambda: clamp_loss_log_vars(model, opts),
    )
    resolved_config = {
        "options": opts.model_dump(mode="json"),
        "world_model": world_model_opts.model_dump(mode="json"),
        "venue_mode": venue_mode,
        "freeze_team_embeddings": freeze_team_embeddings,
        "sidecar_tables": sorted(sidecar_loaders),
    }
    source_fingerprint = _training_source_fingerprint(table, split)

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
    start_epoch = 1
    initial_z_base: Tensor | None = None
    if resume_checkpoint is not None:
        payload = load_resume_checkpoint(resume_checkpoint)
        validate_resume_checkpoint(
            payload,
            trainer="season",
            phase="venue" if venue_mode else "fit",
            config=resolved_config,
            source_fingerprint=source_fingerprint,
        )
        load_training_state(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=generator,
        )
        extra_state = payload.get("extra_state", {})
        best_model = extra_state.get("best_model_state_dict")
        best_epoch = extra_state.get("best_epoch")
        best_validation = float(extra_state.get("best_validation", float("inf")))
        patience_counter = int(extra_state.get("patience_counter", 0))
        rows = list(payload.get("history", []))
        iteration = int(payload.get("optimizer_step", 0))
        controller.optimizer_step = iteration
        start_epoch = int(payload["completed_epoch"]) + 1
        initial_loss = float(extra_state.get("initial_loss", initial_loss))
        stored_initial_z_base = extra_state.get("initial_z_base")
        if torch.is_tensor(stored_initial_z_base):
            initial_z_base = stored_initial_z_base.detach().cpu().clone()

    active_base_indices = torch.cat(
        [dataset.red_team_idx[train_rows], dataset.blue_team_idx[train_rows]], dim=1
    )
    if initial_z_base is None:
        initial_z_base = model.Z_base.weight.detach().cpu().clone()
    nominal_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    gradient_receiving_names: set[str] = set()
    last_gradients: dict[str, Tensor] = {}
    gradient_hooks = []
    collect_observability = tensorboard_writer is not None or opts.state_model == "static-z-base"
    if collect_observability:
        for parameter_name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            def record_gradient(gradient: Tensor, *, name: str = parameter_name) -> Tensor:
                if torch.isfinite(gradient).any() and torch.count_nonzero(gradient).item() > 0:
                    gradient_receiving_names.add(name)
                    if name.startswith(("sab.", "cross.", "pma.", "cont_head.", "bin_head.")):
                        values = gradient.detach().flatten().cpu()
                        if values.numel() > 50_000:
                            indices = torch.linspace(
                                0, values.numel() - 1, 50_000, dtype=torch.long
                            )
                            values = values[indices]
                        last_gradients[name] = values
                return gradient

            gradient_hooks.append(parameter.register_hook(record_gradient))
    train_events = (
        int(table.iloc[train_rows]["event_key"].nunique()) if "event_key" in table.columns else 0
    )
    validation_events = (
        int(table.iloc[validation_rows]["event_key"].nunique())
        if len(validation_rows) and "event_key" in table.columns
        else 0
    )
    if tensorboard_writer is not None and not opts.core_objective:
        _write_parameter_histograms(
            tensorboard_writer,
            model,
            active_base_indices,
            initial_z_base,
            step=0,
        )

    for epoch in range(start_epoch, opts.epochs + 1):
        gradient_receiving_names.clear()
        epoch_started = time.perf_counter()
        epoch_step_start = len(controller.step_results)
        model.train()
        forward_model.train()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        task_sums, task_weights = _empty_epoch_accumulators()
        sidecar_iters = {name: cycle(loader) for name, loader in sidecar_loaders.items()}
        epoch_sample_count = 0
        for batch_index, batch in enumerate(train_loader):
            batch = batch_to_device(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                loss, metrics, _ = _loss_from_batch(
                    model, batch, opts, positive_weights, forward_model
                )
                batch_rows = int(batch[0].shape[0])
                core_weights = (
                    _core_batch_task_weights(batch, opts) if opts.core_objective else None
                )
                for task_name, loss_value in _metrics_to_losses(metrics).items():
                    weight = (
                        core_weights.get(task_name, 0.0) if core_weights is not None else batch_rows
                    )
                    _accumulate_loss(task_sums, task_weights, task_name, loss_value, weight)
                for name, iterator in sidecar_iters.items():
                    sidecar_batch = batch_to_device(next(iterator), device)
                    if name == "rank":
                        raw_loss = _rank_loss_from_batch(model, sidecar_batch, opts)
                    elif name == "playoff":
                        raw_loss = _playoff_loss_from_batch(model, sidecar_batch, opts)
                    elif name == "award_probe":
                        raw_loss = _award_probe_loss_from_batch(model, sidecar_batch)
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
            controller.backward(
                loss,
                microbatch_index=batch_index,
                microbatch_count=len(train_loader),
            )
            epoch_sample_count += batch_rows
            iteration = controller.optimizer_step

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
        validation_task_means: dict[str, float] = {name: math.nan for name in TASK_NAMES}
        validation_task_counts: dict[str, float] = {name: 0.0 for name in TASK_NAMES}
        if len(validation_rows) and collect_observability:
            validation_task_means, validation_task_counts = evaluate_task_means(
                model,
                validation_loader,
                opts,
                positive_weights,
                device,
                forward_model=forward_model,
                amp_enabled=amp_enabled,
            )
        train_forecast_metrics: dict[str, float] = {}
        validation_forecast_metrics: dict[str, float] = {}
        if collect_observability:
            train_forecast_metrics = evaluate_forecast_metrics(
                model,
                train_eval_loader,
                device,
                target_mu,
                target_sigma,
                opts,
                score_residual=score_residual,
                forward_model=forward_model,
                amp_enabled=amp_enabled,
            )
            validation_forecast_metrics = {name: math.nan for name in train_forecast_metrics}
            if len(validation_rows):
                validation_forecast_metrics = evaluate_forecast_metrics(
                    model,
                    validation_loader,
                    device,
                    target_mu,
                    target_sigma,
                    opts,
                    score_residual=score_residual,
                    forward_model=forward_model,
                    amp_enabled=amp_enabled,
                )
        task_means = _epoch_task_means(task_sums, task_weights)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": epoch_learning_rate,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_events": train_events,
            "validation_events": validation_events,
            "nominal_parameter_count": nominal_parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "gradient_receiving_parameter_count": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name in gradient_receiving_names
            ),
        }
        for group_index, group in enumerate(optimizer.param_groups):
            row[f"learning_rate_group_{group_index}"] = float(group["lr"])
        row.update(_z_base_statistics(model, active_base_indices, initial_z_base))
        for module, utilization in _parameter_utilization_by_module(
            model, gradient_receiving_names
        ).items():
            for kind, count in utilization.items():
                row[f"module_{module}_{kind}_parameter_count"] = count
        row.update({f"train_{name}": value for name, value in train_forecast_metrics.items()})
        row.update(
            {f"validation_{name}": value for name, value in validation_forecast_metrics.items()}
        )
        row.update(
            epoch_runtime_metrics(
                started_at=epoch_started,
                sample_count=epoch_sample_count,
                optimizer_summary=controller.epoch_summary(start_index=epoch_step_start),
            )
        )
        for task_name, value in task_means.items():
            row[f"{task_name}_loss"] = value
            row[f"{task_name}_active_count"] = task_weights[task_name]
            row[f"validation_{task_name}_loss"] = validation_task_means[task_name]
            row[f"validation_{task_name}_active_count"] = validation_task_counts[task_name]
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
                task_counts=task_weights,
                validation_task_means=validation_task_means,
                validation_task_counts=validation_task_counts,
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
                    opts.use_early_stopping and patience_counter >= opts.early_stopping_patience
                )
        if (
            resume_output is not None
            and runtime_config.checkpoint_every_epochs > 0
            and epoch % runtime_config.checkpoint_every_epochs == 0
        ):
            payload = resume_payload(
                trainer="season",
                phase="venue" if venue_mode else "fit",
                completed_epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_step=controller.optimizer_step,
                history=rows,
                config=resolved_config,
                source_fingerprint=source_fingerprint,
                scaler=scaler,
                loader_generator=generator,
                extra_state={
                    "best_model_state_dict": best_model,
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "patience_counter": patience_counter,
                    "initial_loss": initial_loss,
                    "initial_z_base": initial_z_base,
                },
            )
            save_resume_checkpoint(resume_output, payload)
            if epoch in set(opts.checkpoint_milestone_epochs):
                milestone = Path(resume_output).parent / "milestones" / f"epoch_{epoch:03d}.ckpt"
                milestone.parent.mkdir(parents=True, exist_ok=True)
                save_resume_checkpoint(milestone, payload)
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
    if tensorboard_writer is not None and not opts.core_objective:
        _write_parameter_histograms(
            tensorboard_writer,
            model,
            active_base_indices,
            initial_z_base,
            step=int(rows[-1]["epoch"]) if rows else 0,
            gradients=last_gradients,
        )
    for hook in gradient_hooks:
        hook.remove()
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
