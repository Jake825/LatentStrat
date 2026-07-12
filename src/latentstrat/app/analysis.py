"""Lightweight post-training analysis and diagnostic season inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from latentstrat.app.contracts import EmbeddingSpaceRef
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.experimental.frozen_targets import WorldModelOptions
from latentstrat.season.model import SetTransformerModel, init_model


@dataclass(frozen=True)
class PCAResult:
    coordinates: pd.DataFrame
    explained_variance_ratio: tuple[float, ...]


@dataclass(frozen=True)
class MatchDiagnostic:
    match_key: str
    event_key: str
    red_teams: tuple[str, str, str]
    blue_teams: tuple[str, str, str]
    unknown_teams: tuple[str, ...]
    predicted_red_score: float | None
    predicted_blue_score: float | None
    red_win_probability: float
    red_pma_weights: tuple[float, ...]
    blue_pma_weights: tuple[float, ...]
    swap_probability_gap: float
    zero_slot: pd.DataFrame


@dataclass(frozen=True)
class CheckpointCompatibility:
    compatible: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _finite_matrix(frame: pd.DataFrame, vector_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame[vector_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    return values[valid], valid


def compute_pca(
    space: EmbeddingSpaceRef,
    frame: pd.DataFrame,
    *,
    components: int = 3,
) -> PCAResult:
    columns = list(space.vector_columns)
    if not columns:
        columns = [str(column) for column in frame.columns if str(column).startswith("z_")]
    if not columns:
        raise ValueError(f"No vector columns were found for {space.label}.")
    matrix, valid = _finite_matrix(frame, columns)
    n_components = min(int(components), matrix.shape[0], matrix.shape[1])
    if n_components < 1:
        raise ValueError("PCA requires at least one finite embedding row and dimension.")
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(matrix)
    output = frame.loc[valid].copy().reset_index(drop=True)
    for index in range(n_components):
        output[f"PC{index + 1}"] = scores[:, index]
    return PCAResult(
        coordinates=output,
        explained_variance_ratio=tuple(float(value) for value in pca.explained_variance_ratio_),
    )


def cosine_neighbors(
    space: EmbeddingSpaceRef,
    frame: pd.DataFrame,
    query_index: int,
    *,
    limit: int = 10,
) -> pd.DataFrame:
    columns = list(space.vector_columns)
    if not columns:
        raise ValueError("Embedding space has no declared vector columns.")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not 0 <= query_index < len(values):
        raise IndexError(query_index)
    finite = np.isfinite(values).all(axis=1)
    norms = np.linalg.norm(values, axis=1)
    valid = finite & (norms > 0)
    if not valid[query_index]:
        raise ValueError("Selected embedding is zero or nonfinite.")
    normalized = np.zeros_like(values)
    normalized[valid] = values[valid] / norms[valid, None]
    similarities = normalized @ normalized[query_index]
    similarities[~valid] = -np.inf
    similarities[query_index] = -np.inf
    order = np.argsort(-similarities)[: max(int(limit), 0)]
    result = frame.iloc[order].copy()
    result.insert(0, "cosine_similarity", similarities[order])
    result.insert(0, "row_index", order)
    return result.reset_index(drop=True)


def _world_options(payload: dict[str, Any]) -> WorldModelOptions:
    if payload.get("checkpoint_schema_version") in {6, 7}:
        return WorldModelOptions.model_validate(payload.get("world_model") or {})
    return WorldModelOptions(enabled=False)


def build_season_model(payload: dict[str, Any]) -> SetTransformerModel:
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or "Z_base.weight" not in state:
        raise ValueError("Checkpoint does not contain a supported season model state.")
    opts = LatentStratOptions.model_validate(
        payload.get("options", default_options().model_dump(mode="json"))
    )
    base_weight = state["Z_base.weight"]
    event_weight = state.get("Z_event.weight")
    model = init_model(
        int(base_weight.shape[0]),
        int(base_weight.shape[1]),
        len(opts.target_map),
        len(opts.binary_targets),
        opts,
        num_event_teams=int(event_weight.shape[0]) if event_weight is not None else 1,
        num_endgame_classes=len(opts.endgame_class_order),
        num_awards=len(opts.award_targets),
        world_model_opts=_world_options(payload),
    )
    model.load_state_dict(state, strict=payload.get("checkpoint_schema_version") in {6, 7})
    model.eval()
    return model


def checkpoint_feature_compatibility(
    payload: dict[str, Any], table: pd.DataFrame
) -> CheckpointCompatibility:
    reasons: list[str] = []
    warnings: list[str] = []
    required = {
        "event_key",
        "match_key",
        *{f"{color}_team_{slot}_key" for color in ("red", "blue") for slot in range(1, 4)},
    }
    missing = sorted(required - set(table.columns))
    if missing:
        reasons.append(f"feature table is missing: {', '.join(missing)}")

    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        reasons.append("checkpoint has no model_state_dict")
        state = {}
    base = state.get("Z_base.weight")
    event = state.get("Z_event.weight")
    if not torch.is_tensor(base) or base.ndim != 2:
        reasons.append("checkpoint has no two-dimensional Z_base table")
    state_model = (payload.get("options") or {}).get("state_model", "base-plus-event")
    if state_model != "static-z-base" and (not torch.is_tensor(event) or event.ndim != 2):
        reasons.append("checkpoint has no two-dimensional Z_event table")
    if torch.is_tensor(base) and torch.is_tensor(event) and base.shape[1] != event.shape[1]:
        reasons.append("base and event embedding dimensions disagree")

    base_map = payload.get("team_base_index_map") or {}
    event_map = payload.get("team_event_index_map") or {}
    if torch.is_tensor(base) and base_map:
        indices = [int(value) for value in base_map.values()]
        if min(indices) < 0 or max(indices) >= base.shape[0]:
            reasons.append("team_base_index_map exceeds the base embedding table")
    if torch.is_tensor(event) and event_map:
        indices = [int(value) for value in event_map.values()]
        if min(indices) < 0 or max(indices) >= event.shape[0]:
            reasons.append("team_event_index_map exceeds the event embedding table")

    configured_season = (payload.get("options") or {}).get("season")
    if configured_season is not None and "season" in table:
        table_seasons = set(pd.to_numeric(table["season"], errors="coerce").dropna().astype(int))
        if table_seasons and table_seasons != {int(configured_season)}:
            reasons.append("checkpoint season does not match the selected feature-table season")

    team_columns = sorted(
        column for column in required if column.endswith("_team_1_key") or "_team_" in column
    )
    if team_columns and base_map and not missing:
        teams = pd.unique(table[team_columns].astype(str).to_numpy().reshape(-1))
        unknown_count = sum(team not in base_map for team in teams)
        if unknown_count:
            warnings.append(
                f"{unknown_count} feature-table teams are absent from the checkpoint vocabulary "
                "and will use the ghost row"
            )
    return CheckpointCompatibility(
        compatible=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def season_embedding_frame(
    payload: dict[str, Any],
    *,
    event_key: str | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    model = build_season_model(payload)
    team_map = {
        str(key): int(value) for key, value in (payload.get("team_base_index_map") or {}).items()
    }
    inverse = sorted(team_map.items(), key=lambda pair: pair[1])
    if not inverse:
        inverse = [(f"team_{index}", index) for index in range(1, model.Z_base.num_embeddings)]
    base_indices = torch.tensor([index for _, index in inverse], dtype=torch.long)
    event_indices = torch.zeros_like(base_indices)
    basis_suffix = "base"
    if event_key:
        event_map = {
            str(key): int(value)
            for key, value in (payload.get("team_event_index_map") or {}).items()
        }
        event_indices = torch.tensor(
            [event_map.get(f"{event_key}::{team}", 0) for team, _ in inverse], dtype=torch.long
        )
        basis_suffix = f"effective:{event_key}"
    with torch.inference_mode():
        values = model.team_latent(base_indices, event_indices).detach().cpu().numpy()
    columns = tuple(f"z_{index:03d}" for index in range(values.shape[1]))
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "team_index", base_indices.numpy())
    frame.insert(0, "team_key", [team for team, _ in inverse])
    if event_key:
        frame.insert(1, "event_key", event_key)
    return frame, columns, basis_suffix


def season_event_trajectory_frame(
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if (payload.get("options") or {}).get("state_model") == "static-z-base":
        raise ValueError("Static Z_base checkpoints have no event states or temporal trajectories.")
    model = build_season_model(payload)
    base_map = {
        str(key): int(value) for key, value in (payload.get("team_base_index_map") or {}).items()
    }
    event_map = {
        str(key): int(value)
        for key, value in (payload.get("team_event_index_map") or {}).items()
        if "::" in str(key)
    }
    rows = []
    for key, event_index in sorted(event_map.items()):
        event_key, team_key = key.split("::", 1)
        rows.append((event_key, team_key, base_map.get(team_key, 0), event_index))
    if not rows:
        raise ValueError("Checkpoint contains no team-event states.")
    base_indices = torch.tensor([row[2] for row in rows], dtype=torch.long)
    event_indices = torch.tensor([row[3] for row in rows], dtype=torch.long)
    with torch.inference_mode():
        values = model.team_latent(base_indices, event_indices).detach().cpu().numpy()
    columns = tuple(f"z_{index:03d}" for index in range(values.shape[1]))
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "event_key", [row[0] for row in rows])
    frame.insert(0, "team_key", [row[1] for row in rows])
    return frame, columns


def prior_embedding_frame(payload: dict[str, Any]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    table = payload.get("embedding_table")
    if not torch.is_tensor(table) or table.ndim != 2:
        raise ValueError("Prior checkpoint has no two-dimensional embedding table.")
    values = table.detach().cpu().numpy()
    columns = tuple(f"z_{index:03d}" for index in range(values.shape[1]))
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "team_number", np.arange(len(frame), dtype=int))
    frame.insert(
        0,
        "team_key",
        ["ghost" if index == 0 else f"frc{index}" for index in range(len(frame))],
    )
    return frame, columns


def _target_stats(payload: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    stats = payload.get("target_stats") or {}
    names = [str(value) for value in stats.get("target_names") or []]
    mu = np.asarray(stats.get("mu") or [], dtype=float)
    sigma = np.asarray(stats.get("sigma") or [], dtype=float)
    return names, mu, sigma


def _scores(payload: dict[str, Any], continuous: torch.Tensor) -> tuple[float | None, float | None]:
    names, mu, sigma = _target_stats(payload)
    values = continuous.detach().cpu().numpy()[0]
    if len(values) == len(mu) == len(sigma):
        values = values * sigma + mu
    by_name = {name: float(values[index]) for index, name in enumerate(names[: len(values)])}
    red_names = [
        name
        for name in by_name
        if name.startswith("red_") and name.endswith(("auto_pts", "teleop_pts"))
    ]
    blue_names = [
        name
        for name in by_name
        if name.startswith("blue_") and name.endswith(("auto_pts", "teleop_pts"))
    ]
    if red_names and blue_names:
        return sum(by_name[name] for name in red_names), sum(by_name[name] for name in blue_names)
    if len(values) >= 4:
        return float(values[0] + values[1]), float(values[2] + values[3])
    return None, None


def _pma_tuple(value: torch.Tensor) -> tuple[float, ...]:
    array = value.detach().cpu().numpy()[0]
    while array.ndim > 1:
        array = array.mean(axis=0)
    return tuple(float(item) for item in array)


def diagnose_match(payload: dict[str, Any], row: pd.Series) -> MatchDiagnostic:
    model = build_season_model(payload)
    base_map = {
        str(key): int(value) for key, value in (payload.get("team_base_index_map") or {}).items()
    }
    event_map = {
        str(key): int(value) for key, value in (payload.get("team_event_index_map") or {}).items()
    }
    event_key = str(row["event_key"])
    red_teams = tuple(str(row[f"red_team_{slot}_key"]) for slot in range(1, 4))
    blue_teams = tuple(str(row[f"blue_team_{slot}_key"]) for slot in range(1, 4))
    unknown = tuple(team for team in (*red_teams, *blue_teams) if team not in base_map)

    def tensors(teams: tuple[str, str, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.tensor([[base_map.get(team, 0) for team in teams]], dtype=torch.long)
        event = torch.tensor(
            [[event_map.get(f"{event_key}::{team}", 0) for team in teams]], dtype=torch.long
        )
        missing = torch.tensor([[team not in base_map for team in teams]], dtype=torch.bool)
        return base, event, missing

    red_base, red_event, red_missing = tensors(red_teams)
    blue_base, blue_event, blue_missing = tensors(blue_teams)

    def forward(red_zero_slot: int = 0, blue_zero_slot: int = 0, swap: bool = False):
        if swap:
            return model(
                blue_base,
                red_base,
                red_event_idx=blue_event,
                blue_event_idx=red_event,
                red_missing_mask=blue_missing,
                blue_missing_mask=red_missing,
            )
        return model(
            red_base,
            blue_base,
            red_event_idx=red_event,
            blue_event_idx=blue_event,
            red_missing_mask=red_missing,
            blue_missing_mask=blue_missing,
            red_zero_slot=red_zero_slot,
            blue_zero_slot=blue_zero_slot,
        )

    with torch.inference_mode():
        output = forward()
        probability = float(torch.sigmoid(output.bin_logits).reshape(-1)[0])
        red_score, blue_score = _scores(payload, output.cont_z)
        swapped_probability = float(torch.sigmoid(forward(swap=True).bin_logits).reshape(-1)[0])
        zero_rows = []
        for color in ("red", "blue"):
            teams = red_teams if color == "red" else blue_teams
            for slot, team in enumerate(teams, start=1):
                changed = forward(
                    red_zero_slot=slot if color == "red" else 0,
                    blue_zero_slot=slot if color == "blue" else 0,
                )
                changed_probability = float(torch.sigmoid(changed.bin_logits).reshape(-1)[0])
                changed_red, changed_blue = _scores(payload, changed.cont_z)
                zero_rows.append(
                    {
                        "alliance": color,
                        "slot": slot,
                        "team_key": team,
                        "red_win_probability_delta": changed_probability - probability,
                        "red_score_delta": (
                            changed_red - red_score
                            if changed_red is not None and red_score is not None
                            else np.nan
                        ),
                        "blue_score_delta": (
                            changed_blue - blue_score
                            if changed_blue is not None and blue_score is not None
                            else np.nan
                        ),
                    }
                )
    return MatchDiagnostic(
        match_key=str(row["match_key"]),
        event_key=event_key,
        red_teams=red_teams,
        blue_teams=blue_teams,
        unknown_teams=unknown,
        predicted_red_score=red_score,
        predicted_blue_score=blue_score,
        red_win_probability=probability,
        red_pma_weights=_pma_tuple(output.red_pma_weights),
        blue_pma_weights=_pma_tuple(output.blue_pma_weights),
        swap_probability_gap=abs((1.0 - probability) - swapped_probability),
        zero_slot=pd.DataFrame(zero_rows),
    )


__all__ = [
    "CheckpointCompatibility",
    "MatchDiagnostic",
    "PCAResult",
    "build_season_model",
    "checkpoint_feature_compatibility",
    "compute_pca",
    "cosine_neighbors",
    "diagnose_match",
    "prior_embedding_frame",
    "season_embedding_frame",
    "season_event_trajectory_frame",
]
