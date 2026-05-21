"""Post-training embedding inspection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from latentstrat.baselines import Baselines
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.model import SetTransformerModel


@dataclass
class Inspection:
    team_embedding_table: pd.DataFrame
    sanity_checks: pd.DataFrame
    pca_model: dict[str, np.ndarray]
    nearest_neighbors: pd.DataFrame
    archetype_similarity: pd.DataFrame
    alliance_attention_table: pd.DataFrame


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    out = values / np.maximum(norms, np.finfo(float).eps)
    out[norms[:, 0] <= np.finfo(float).eps] = 0
    return out


def compute_pca(embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mean = np.nanmean(embeddings, axis=0)
    centered = embeddings - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coeff = vt.T
    raw_score = centered @ coeff
    latent = singular_values**2 / max(len(embeddings) - 1, 1)
    explained = 100 * latent / max(np.sum(latent), np.finfo(float).eps)
    score = np.zeros((len(embeddings), 3))
    score[:, : min(3, raw_score.shape[1])] = raw_score[:, : min(3, raw_score.shape[1])]
    return score, {"coefficients": coeff, "latent": latent, "explained": explained, "mean": mean}


def _team_slots(table: pd.DataFrame) -> np.ndarray:
    columns = [
        "red_team_1_idx",
        "red_team_2_idx",
        "red_team_3_idx",
        "blue_team_1_idx",
        "blue_team_2_idx",
        "blue_team_3_idx",
    ]
    return table[columns].to_numpy(dtype=int)


def _role_average(red_slots, blue_slots, red_values, blue_values, num_teams):
    slots = np.concatenate([red_slots, blue_slots], axis=1).reshape(-1)
    values = np.concatenate(
        [
            np.repeat(np.asarray(red_values)[:, None], 3, axis=1),
            np.repeat(np.asarray(blue_values)[:, None], 3, axis=1),
        ],
        axis=1,
    ).reshape(-1)
    sums = np.bincount(slots, weights=values, minlength=num_teams)
    counts = np.bincount(slots, minlength=num_teams)
    out = sums / np.maximum(counts, 1)
    out[counts == 0] = np.nan
    return out


def build_team_embedding_table(
    embeddings: np.ndarray,
    team_index_map: dict[str, int],
    pca_score: np.ndarray,
    table: pd.DataFrame,
    baselines: Baselines | None,
) -> pd.DataFrame:
    rows = sorted(team_index_map.items(), key=lambda item: item[1])
    keys = [key for key, _ in rows]
    indices = np.array([idx for _, idx in rows], dtype=int)
    ordered = embeddings[indices]
    result = pd.DataFrame(
        {
            "team_key": keys,
            "team_index": indices,
            "embedding": list(ordered),
            "embedding_norm": np.linalg.norm(ordered, axis=1),
            "pc1": pca_score[indices, 0],
            "pc2": pca_score[indices, 1],
            "pc3": pca_score[indices, 2],
        }
    )
    team_slots = _team_slots(table)
    num_teams = len(result)
    result["match_count"] = np.bincount(team_slots.reshape(-1), minlength=num_teams)[indices]
    result["event_count"] = [
        table.loc[np.any(team_slots == idx, axis=1), "event_key"].astype(str).nunique()
        for idx in indices
    ]
    red_slots = table[["red_team_1_idx", "red_team_2_idx", "red_team_3_idx"]].to_numpy(dtype=int)
    blue_slots = table[["blue_team_1_idx", "blue_team_2_idx", "blue_team_3_idx"]].to_numpy(
        dtype=int
    )
    result["avg_alliance_score"] = _role_average(
        red_slots, blue_slots, table["red_total_score"], table["blue_total_score"], num_teams
    )[indices]
    result["avg_point_differential"] = _role_average(
        red_slots, blue_slots, table["win_margin"], -table["win_margin"], num_teams
    )[indices]
    result["avg_fouls_drawn"] = _role_average(
        red_slots, blue_slots, table["fouls_drawn"], -table["fouls_drawn"], num_teams
    )[indices]
    result["rate_win"] = _role_average(
        red_slots,
        blue_slots,
        table["red_win"].astype(float),
        (~table["red_win"].astype(bool)).astype(float),
        num_teams,
    )[indices]
    if baselines is not None:
        coeff = np.asarray(baselines.ridge.coefficients)
        for target_idx, target in enumerate(baselines.target_names):
            if coeff.shape[0] >= 2 * num_teams:
                result[f"ridge_{target}"] = 0.5 * (
                    coeff[indices, target_idx] + coeff[indices + num_teams, target_idx]
                )
            else:
                result[f"ridge_{target}"] = coeff[indices, target_idx]
    return result


def sanity_checks(embeddings: np.ndarray, team_table: pd.DataFrame) -> pd.DataFrame:
    norms = team_table["embedding_norm"].to_numpy()
    variances = np.nanvar(embeddings, axis=0, ddof=1)
    dominant_share = np.max(variances) / max(np.sum(variances), np.finfo(float).eps)
    if (
        np.nanstd(norms) <= np.finfo(float).eps
        or np.nanstd(np.log1p(team_table["match_count"])) <= np.finfo(float).eps
    ):
        norm_corr = 0
    else:
        norm_corr = np.corrcoef(norms, np.log1p(team_table["match_count"]))[0, 1]
    rows = [
        (
            "all_finite",
            bool(np.isfinite(embeddings).all()),
            float(np.isfinite(embeddings).all()),
            1,
        ),
        (
            "no_zero_norms",
            bool(np.all(norms > np.finfo(float).eps)),
            float(np.nanmin(norms)),
            np.finfo(float).eps,
        ),
        (
            "no_collapsed_norm_distribution",
            bool(np.nanstd(norms) > np.finfo(float).eps),
            float(np.nanstd(norms)),
            np.finfo(float).eps,
        ),
        ("no_single_dominant_dimension", bool(dominant_share < 0.90), float(dominant_share), 0.90),
        ("norm_not_only_match_count", bool(abs(norm_corr) < 0.95), float(norm_corr), 0.95),
    ]
    return pd.DataFrame(rows, columns=["check", "passed", "value", "threshold"])


def nearest_neighbors(
    team_table: pd.DataFrame, normalized: np.ndarray, top_k: int = 10
) -> pd.DataFrame:
    rows = []
    top_k = min(top_k, len(team_table) - 1)
    for query_row in range(len(team_table)):
        similarities = normalized @ normalized[query_row]
        similarities[query_row] = -np.inf
        order = np.argsort(-similarities)[:top_k]
        for rank, neighbor_row in enumerate(order, start=1):
            rows.append(
                {
                    "query_team_key": team_table["team_key"].iloc[query_row],
                    "query_team_index": team_table["team_index"].iloc[query_row],
                    "neighbor_team_key": team_table["team_key"].iloc[neighbor_row],
                    "neighbor_team_index": team_table["team_index"].iloc[neighbor_row],
                    "rank": rank,
                    "cosine_similarity": similarities[neighbor_row],
                }
            )
    return pd.DataFrame(rows)


def archetype_similarity(
    team_table: pd.DataFrame, normalized: np.ndarray, top_n: int = 10
) -> pd.DataFrame:
    candidates = {
        "high_alliance_score": "avg_alliance_score",
        "high_point_differential": "avg_point_differential",
        "high_win_rate": "rate_win",
        "foul_differential": "avg_fouls_drawn",
    }
    rows = []
    for archetype, metric in candidates.items():
        values = team_table[metric].to_numpy(dtype=float)
        valid = np.flatnonzero(~np.isnan(values))
        if len(valid) == 0:
            continue
        selected = valid[np.argsort(-values[valid])[: min(top_n, len(valid))]]
        prototype = np.nanmean(normalized[selected], axis=0)
        prototype = prototype / max(np.linalg.norm(prototype), np.finfo(float).eps)
        similarities = normalized @ prototype
        for rank, row in enumerate(
            np.argsort(-similarities)[: min(top_n, len(team_table))], start=1
        ):
            rows.append(
                {
                    "archetype": archetype,
                    "metric_name": metric,
                    "team_key": team_table["team_key"].iloc[row],
                    "team_index": team_table["team_index"].iloc[row],
                    "rank": rank,
                    "metric_value": values[row],
                    "cosine_similarity": similarities[row],
                }
            )
    return pd.DataFrame(rows)


def inspect_embeddings(
    model: SetTransformerModel,
    table: pd.DataFrame,
    team_index_map: dict[str, int],
    opts: LatentStratOptions | None = None,
    baselines: Baselines | None = None,
    *,
    top_k: int = 10,
) -> Inspection:
    opts = opts or default_options()
    embeddings = model.team_embedding.weight.detach().cpu().numpy()
    pca_score, pca_model = compute_pca(embeddings)
    team_table = build_team_embedding_table(embeddings, team_index_map, pca_score, table, baselines)
    normalized = normalize_rows(embeddings)
    # Reuse evaluation attention output without split labels here.
    from latentstrat.data import Split
    from latentstrat.evaluation import evaluate_model

    dummy_split = Split(
        np.ones(len(table), dtype=bool),
        np.zeros(len(table), dtype=bool),
        np.zeros(len(table), dtype=bool),
        "inspection",
    )
    from latentstrat.data import TargetStats

    dummy_stats = TargetStats(
        [mapping.target_name for mapping in opts.target_map],
        np.zeros(len(opts.target_map)),
        np.ones(len(opts.target_map)),
        np.zeros(len(opts.target_map), dtype=bool),
    )
    attention = evaluate_model(
        model, table, dummy_split, dummy_stats, opts, baselines
    ).set_attention
    return Inspection(
        team_embedding_table=team_table,
        sanity_checks=sanity_checks(embeddings, team_table),
        pca_model=pca_model,
        nearest_neighbors=nearest_neighbors(team_table, normalized, top_k),
        archetype_similarity=archetype_similarity(team_table, normalized, top_k),
        alliance_attention_table=attention,
    )
