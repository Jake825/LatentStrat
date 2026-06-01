"""Post-training visualization report for historical match-breakdown embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from latentstrat.inspection import compute_pca
from latentstrat.world_model.match_breakdown.model import MatchBreakdownAutoencoder
from latentstrat.world_model.match_breakdown.parse import AllianceBreakdownRow
from latentstrat.world_model.match_breakdown.train import _season_tensors, _split_indices

PARQUET_ENGINE = "pyarrow"
LATENT_COLUMNS = tuple(f"z_{index:03d}" for index in range(16))
PERCENTILE_COLUMNS = (
    "score_percentile",
    "auto_percentile",
    "endgame_percentile",
    "fouls_percentile",
)
COMPONENT_ALIASES = {
    "auto": ("totalAutoPoints", "autoPoints", "auto_points"),
    "endgame": (
        "endgamePoints",
        "endGameTowerPoints",
        "endGameBargePoints",
        "endGameTotalStagePoints",
        "habClimbPoints",
    ),
    "fouls": ("foulPoints", "foul_points"),
}
EXPECTED_REPORT_FILES = (
    "component_metric_sources.json",
    "production_pca_coordinates.csv",
    "holdout_pca_coordinates.csv",
    "pca_explained_variance.csv",
    "pca_explained_variance.png",
    "production_pca_multiview.png",
    "holdout_pca_multiview.png",
    "production_tsne_coordinates.csv",
    "holdout_tsne_coordinates.csv",
    "production_tsne_multiview.png",
    "holdout_tsne_multiview.png",
    "latent_parallel_coordinates.png",
    "latent_archetype_profiles.csv",
    "cross_season_neighbor_nodes.csv",
    "cross_season_neighbor_edges.csv",
    "cross_season_neighbor_network.png",
)


@dataclass(frozen=True)
class MatchBreakdownInspectionOptions:
    seed: int = 2026
    tsne_max_rows: int = 11_000
    production_tsne_per_season: int = 1_000
    holdout_tsne_per_season: int = 500
    network_node_limit: int = 400
    neighbors_per_node: int = 2
    parallel_rows_per_slice: int = 50
    tsne_iterations: int = 1_000


@dataclass(frozen=True)
class MatchBreakdownInspectionResult:
    output_dir: Path
    manifest: dict[str, Any]


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_plot(plt, fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _validate_options(options: MatchBreakdownInspectionOptions) -> None:
    positive = {
        "tsne_max_rows": options.tsne_max_rows,
        "production_tsne_per_season": options.production_tsne_per_season,
        "holdout_tsne_per_season": options.holdout_tsne_per_season,
        "network_node_limit": options.network_node_limit,
        "neighbors_per_node": options.neighbors_per_node,
        "parallel_rows_per_slice": options.parallel_rows_per_slice,
        "tsne_iterations": options.tsne_iterations,
    }
    invalid = sorted(name for name, value in positive.items() if value <= 0)
    if invalid:
        raise ValueError(f"Inspection options must be positive: {', '.join(invalid)}.")


def load_match_breakdown_model(
    path: str | Path,
) -> tuple[MatchBreakdownAutoencoder, dict[str, Any]]:
    """Reload a frozen match-breakdown encoder checkpoint for inspection."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "v6-lite-match-breakdown-shared-bottleneck":
        raise ValueError(
            f"Unsupported match-breakdown checkpoint kind: {checkpoint.get('kind')!r}."
        )
    widths = {int(season): int(width) for season, width in checkpoint["season_widths"].items()}
    model = MatchBreakdownAutoencoder(widths, latent_dim=int(checkpoint["latent_dim"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _raw_rows(table: pd.DataFrame) -> list[AllianceBreakdownRow]:
    required = {
        "row_id",
        "season",
        "event_key",
        "match_key",
        "comp_level",
        "set_number",
        "match_number",
        "alliance",
        "alliance_score",
        "opponent_score",
        "winning_alliance",
        "flat_breakdown_json",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Alliance feature table is missing columns: {', '.join(missing)}.")
    rows = []
    for row in table.itertuples(index=False):
        flat_breakdown = json.loads(row.flat_breakdown_json)
        if not isinstance(flat_breakdown, dict):
            raise ValueError(f"Expected flattened breakdown object for {row.row_id!r}.")
        rows.append(
            AllianceBreakdownRow(
                row_id=str(row.row_id),
                season=int(row.season),
                event_key=str(row.event_key),
                match_key=str(row.match_key),
                comp_level=str(row.comp_level),
                set_number=int(row.set_number),
                match_number=int(row.match_number),
                alliance=str(row.alliance),
                alliance_score=float(row.alliance_score),
                opponent_score=float(row.opponent_score),
                winning_alliance=str(row.winning_alliance),
                flat_breakdown=flat_breakdown,
            )
        )
    return rows


def holdout_embeddings(
    alliance_features: pd.DataFrame, checkpoint_path: str | Path
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Recreate eval holdout rows and encode them with the holdout-only model."""

    model, checkpoint = load_match_breakdown_model(checkpoint_path)
    rows = _raw_rows(alliance_features)
    tensors = _season_tensors(rows)
    expected = {int(season): int(width) for season, width in checkpoint["season_widths"].items()}
    actual = {season: data.schema.width for season, data in tensors.items()}
    if actual != expected:
        raise ValueError(
            f"Checkpoint season widths do not match alliance features: {expected} != {actual}."
        )
    options = checkpoint["options"]
    _, validation = _split_indices(
        tensors,
        fraction=float(options["validation_fraction"]),
        seed=int(options["seed"]),
    )
    frames = []
    with torch.inference_mode():
        for season, indices in sorted(validation.items()):
            if not len(indices):
                continue
            data = tensors[season]
            _, latent = model(season, data.values[indices], data.masks[indices])
            metadata = pd.DataFrame(
                [
                    {
                        "row_id": data.rows[index].row_id,
                        "season": season,
                        "event_key": data.rows[index].event_key,
                        "match_key": data.rows[index].match_key,
                        "comp_level": data.rows[index].comp_level,
                        "set_number": data.rows[index].set_number,
                        "match_number": data.rows[index].match_number,
                        "alliance": data.rows[index].alliance,
                        "alliance_score": data.rows[index].alliance_score,
                        "opponent_score": data.rows[index].opponent_score,
                        "winning_alliance": data.rows[index].winning_alliance,
                    }
                    for index in indices
                ]
            )
            for latent_index, column in enumerate(LATENT_COLUMNS):
                metadata[column] = latent[:, latent_index].numpy()
            frames.append(metadata)
    if not frames:
        raise ValueError("The evaluation checkpoint has no holdout rows.")
    return pd.concat(frames, ignore_index=True), validation


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return np.nan
    number = float(value)
    return number if np.isfinite(number) else np.nan


def select_component_sources(raw_table: pd.DataFrame) -> dict[str, dict[str, str | None]]:
    """Choose one non-overlapping aggregate source field per season and component."""

    parsed = raw_table[["season", "flat_breakdown_json"]].copy()
    parsed["flat_breakdown"] = parsed["flat_breakdown_json"].map(json.loads)
    selected: dict[str, dict[str, str | None]] = {}
    for season, rows in parsed.groupby("season", sort=True):
        selected[str(int(season))] = {}
        breakdowns = rows["flat_breakdown"].tolist()
        for component, aliases in COMPONENT_ALIASES.items():
            selected[str(int(season))][component] = next(
                (
                    alias
                    for alias in aliases
                    if any(np.isfinite(_finite_number(flat.get(alias))) for flat in breakdowns)
                ),
                None,
            )
    return selected


def component_metric_source_payload(
    selected: dict[str, dict[str, str | None]]
) -> dict[str, Any]:
    return {
        "strategy": "first finite season-specific aggregate alias; never sum overlapping fields",
        "aliases": {name: list(values) for name, values in COMPONENT_ALIASES.items()},
        "selected_sources": selected,
        "missing_seasons": {
            component: [
                int(season)
                for season, sources in sorted(selected.items())
                if sources[component] is None
            ]
            for component in COMPONENT_ALIASES
        },
    }


def add_component_percentiles(
    geometry: pd.DataFrame,
    raw_table: pd.DataFrame,
    selected_sources: dict[str, dict[str, str | None]],
) -> pd.DataFrame:
    """Attach season-relative score and aggregate-component percentile labels."""

    raw = raw_table[["row_id", "flat_breakdown_json"]].copy()
    raw["flat_breakdown"] = raw["flat_breakdown_json"].map(json.loads)
    merged = geometry.merge(
        raw[["row_id", "flat_breakdown"]],
        on="row_id",
        how="left",
        validate="1:1",
    )
    if merged["flat_breakdown"].isna().any():
        raise ValueError("Geometry rows are missing flattened match-breakdown payloads.")
    merged["score_percentile"] = merged.groupby("season")["alliance_score"].rank(
        method="average", pct=True
    )
    for component in COMPONENT_ALIASES:
        raw_column = f"{component}_value"
        merged[raw_column] = [
            (
                _finite_number(flat.get(selected_sources[str(int(season))][component]))
                if selected_sources[str(int(season))][component] is not None
                else np.nan
            )
            for season, flat in zip(
                merged["season"], merged["flat_breakdown"], strict=True
            )
        ]
        merged[f"{component}_percentile"] = merged.groupby("season")[raw_column].rank(
            method="average", pct=True
        )
    return merged.drop(columns=["flat_breakdown"])


def shared_pca_coordinates(
    production: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Project production rows and orthogonally aligned eval rows into one PCA frame."""

    production_matrix = production[list(LATENT_COLUMNS)].to_numpy(dtype=float)
    holdout_matrix = holdout[list(LATENT_COLUMNS)].to_numpy(dtype=float)
    production_score, pca = compute_pca(production_matrix)
    reference = production.set_index("row_id").loc[holdout["row_id"], list(LATENT_COLUMNS)]
    reference_matrix = reference.to_numpy(dtype=float)
    holdout_mean = holdout_matrix.mean(axis=0)
    reference_mean = reference_matrix.mean(axis=0)
    rotation, _ = orthogonal_procrustes(
        holdout_matrix - holdout_mean,
        reference_matrix - reference_mean,
    )
    aligned = (holdout_matrix - holdout_mean) @ rotation + reference_mean
    raw_holdout_score = (aligned - pca["mean"]) @ pca["coefficients"]
    holdout_score = np.zeros((len(holdout), 3), dtype=float)
    holdout_score[:, : min(3, raw_holdout_score.shape[1])] = raw_holdout_score[
        :, : min(3, raw_holdout_score.shape[1])
    ]
    production_out = production.copy()
    holdout_out = holdout.copy()
    for index, name in enumerate(("pc1", "pc2", "pc3")):
        production_out[name] = production_score[:, index]
        holdout_out[name] = holdout_score[:, index]
    for index, column in enumerate(LATENT_COLUMNS):
        holdout_out[f"eval_{column}"] = holdout_matrix[:, index]
        holdout_out[f"aligned_{column}"] = aligned[:, index]
    explained = pd.DataFrame(
        {
            "component": np.arange(1, len(pca["explained"]) + 1, dtype=int),
            "explained_variance_percent": pca["explained"],
            "cumulative_explained_variance_percent": np.cumsum(pca["explained"]),
        }
    )
    alignment = {
        "kind": "orthogonal-procrustes",
        "purpose": (
            "Align independently trained eval latents to the production coordinate frame "
            "without changing eval-model pairwise Euclidean distances."
        ),
        "holdout_rows": len(holdout),
    }
    return production_out, holdout_out, explained, alignment


def stratified_projection_sample(
    table: pd.DataFrame,
    *,
    max_rows: int,
    per_season_limit: int,
    seed: int,
) -> pd.DataFrame:
    """Sample deterministically by season and score decile for local-neighborhood plots."""

    if table.empty:
        return table.copy()
    work = table.copy()
    work["score_decile"] = np.minimum(
        np.floor(work["score_percentile"].fillna(0).to_numpy(dtype=float) * 10).astype(int),
        9,
    )
    rng = np.random.default_rng(seed)
    queues = {}
    for key, rows in work.groupby(["season", "score_decile"], sort=True):
        values = rows.index.to_numpy(dtype=int)
        queues[key] = list(values[rng.permutation(len(values))])
    selected = []
    season_counts: dict[int, int] = {}
    active = list(sorted(queues))
    while active and len(selected) < max_rows:
        next_active = []
        for key in active:
            season = int(key[0])
            if season_counts.get(season, 0) >= per_season_limit or not queues[key]:
                continue
            selected.append(queues[key].pop())
            season_counts[season] = season_counts.get(season, 0) + 1
            if len(selected) >= max_rows:
                break
            if queues[key] and season_counts[season] < per_season_limit:
                next_active.append(key)
        active = next_active
    return work.loc[selected].reset_index(drop=True)


def tsne_coordinates(table: pd.DataFrame, *, seed: int, iterations: int) -> pd.DataFrame:
    """Fit one deterministic two-dimensional t-SNE projection for one geometry dataset."""

    if table.empty:
        return table.assign(tsne_x=pd.Series(dtype=float), tsne_y=pd.Series(dtype=float))
    matrix = table[list(LATENT_COLUMNS)].to_numpy(dtype=float)
    if len(matrix) < 3:
        coords = np.zeros((len(matrix), 2), dtype=float)
        if len(matrix) == 2:
            coords[:, 0] = [-0.5, 0.5]
    else:
        perplexity = min(30.0, max(1.0, float(len(matrix) - 1) / 3.0))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=iterations,
            random_state=seed,
        ).fit_transform(matrix)
    out = table.copy()
    out["tsne_x"] = coords[:, 0]
    out["tsne_y"] = coords[:, 1]
    return out


def _scatter_season(ax, table: pd.DataFrame, x: str, y: str) -> None:
    categories = pd.Categorical(table["season"].astype(str))
    scatter = ax.scatter(table[x], table[y], c=categories.codes, cmap="tab20", s=7, alpha=0.55)
    ax.set_title("Season")
    colorbar = ax.figure.colorbar(scatter, ax=ax, ticks=range(len(categories.categories)))
    colorbar.ax.set_yticklabels(categories.categories)


def _scatter_percentile(ax, table: pd.DataFrame, x: str, y: str, column: str, title: str) -> None:
    values = pd.to_numeric(table[column], errors="coerce")
    finite = values.notna()
    if (~finite).any():
        ax.scatter(table.loc[~finite, x], table.loc[~finite, y], c="#d1d5db", s=6, alpha=0.25)
    if finite.any():
        scatter = ax.scatter(
            table.loc[finite, x],
            table.loc[finite, y],
            c=values.loc[finite],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=7,
            alpha=0.58,
        )
        ax.figure.colorbar(scatter, ax=ax, label="within-season percentile")
    ax.set_title(title)


def _scatter_highlight(ax, table: pd.DataFrame, x: str, y: str, column: str, title: str) -> None:
    values = pd.to_numeric(table[column], errors="coerce")
    selected = values >= 0.95
    ax.scatter(table[x], table[y], c="#d1d5db", s=5, alpha=0.18)
    ax.scatter(table.loc[selected, x], table.loc[selected, y], c="#dc2626", s=8, alpha=0.68)
    ax.set_title(title)


def write_projection_multiview(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    x: str,
    y: str,
    axis_prefix: str,
    title: str,
) -> None:
    """Write one fixed-coordinate multi-view plot with season and archetype colorings."""

    plt = _plot_setup()
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    _scatter_season(axes[0, 0], table, x, y)
    _scatter_percentile(axes[0, 1], table, x, y, "score_percentile", "Score Percentile")
    _scatter_percentile(axes[0, 2], table, x, y, "auto_percentile", "Auto Percentile")
    _scatter_percentile(axes[0, 3], table, x, y, "endgame_percentile", "Endgame Percentile")
    _scatter_percentile(axes[1, 0], table, x, y, "fouls_percentile", "Foul Percentile")
    _scatter_highlight(axes[1, 1], table, x, y, "auto_percentile", "Top 5% Auto")
    _scatter_highlight(axes[1, 2], table, x, y, "endgame_percentile", "Top 5% Endgame")
    _scatter_highlight(axes[1, 3], table, x, y, "fouls_percentile", "Top 5% Fouls")
    for ax in axes.flat:
        ax.set_xlabel(f"{axis_prefix} 1")
        ax.set_ylabel(f"{axis_prefix} 2")
        ax.grid(True, alpha=0.12)
    fig.suptitle(title)
    _save_plot(plt, fig, Path(output_path))


def write_pca_explained_variance(table: pd.DataFrame, output_path: str | Path) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(table["component"], table["explained_variance_percent"], color="#0f766e")
    ax.plot(
        table["component"],
        table["cumulative_explained_variance_percent"],
        color="#dc2626",
        marker="o",
        linewidth=1.5,
        label="cumulative",
    )
    ax.set_title("Production Latent PCA Explained Variance")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(table["component"])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    _save_plot(plt, fig, Path(output_path))


def archetype_profiles(
    production: pd.DataFrame,
    *,
    rows_per_slice: int,
    seed: int,
) -> pd.DataFrame:
    """Select deterministic latent rows for parallel-coordinate interpretation."""

    rng = np.random.default_rng(seed)
    archetypes = {
        "random_reference": None,
        "top_5pct_score": "score_percentile",
        "top_5pct_auto": "auto_percentile",
        "top_5pct_endgame": "endgame_percentile",
        "top_5pct_fouls": "fouls_percentile",
    }
    latent = production[list(LATENT_COLUMNS)].to_numpy(dtype=float)
    mean = latent.mean(axis=0)
    sigma = latent.std(axis=0)
    sigma[sigma <= np.finfo(float).eps] = 1.0
    frames = []
    for archetype, percentile in archetypes.items():
        candidates = (
            production if percentile is None else production[production[percentile] >= 0.95]
        )
        if candidates.empty:
            continue
        indices = candidates.index.to_numpy(dtype=int)
        selected = indices[rng.permutation(len(indices))[: min(rows_per_slice, len(indices))]]
        frame = production.loc[selected].copy()
        frame.insert(0, "archetype", archetype)
        for index, column in enumerate(LATENT_COLUMNS):
            frame[f"standardized_{column}"] = (frame[column] - mean[index]) / sigma[index]
        frames.append(frame)
    if not frames:
        raise ValueError("No archetype rows were available for parallel coordinates.")
    return pd.concat(frames, ignore_index=True)


def write_parallel_coordinates(table: pd.DataFrame, output_path: str | Path) -> None:
    archetypes = list(table["archetype"].drop_duplicates())
    plt = _plot_setup()
    fig, axes = plt.subplots(len(archetypes), 1, figsize=(13, max(4, 2.6 * len(archetypes))))
    if len(archetypes) == 1:
        axes = [axes]
    x = np.arange(len(LATENT_COLUMNS))
    for ax, archetype in zip(axes, archetypes, strict=True):
        values = table.loc[
            table["archetype"] == archetype,
            [f"standardized_{column}" for column in LATENT_COLUMNS],
        ].to_numpy(dtype=float)
        for row in values:
            ax.plot(x, row, color="#64748b", alpha=0.16, linewidth=0.8)
        ax.plot(x, values.mean(axis=0), color="#dc2626", linewidth=2.4)
        ax.set_title(archetype.replace("_", " ").title())
        ax.set_ylabel("standardized activation")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xticks(x, LATENT_COLUMNS, rotation=45, ha="right")
    fig.suptitle("Production Latent Archetype Profiles")
    _save_plot(plt, fig, Path(output_path))


def extreme_network_seeds(
    production: pd.DataFrame,
    *,
    node_limit: int,
    seed: int,
) -> pd.DataFrame:
    """Select a balanced deterministic set of extreme production rows."""

    work = production.copy()
    choices = []
    for archetype, percentile in (
        ("score", "score_percentile"),
        ("auto", "auto_percentile"),
        ("endgame", "endgame_percentile"),
        ("fouls", "fouls_percentile"),
    ):
        rows = work[work[percentile] >= 0.95].copy()
        rows["network_archetype"] = archetype
        choices.append(rows)
    candidates = pd.concat(choices, ignore_index=True)
    if candidates.empty:
        raise ValueError("No extreme rows were available for the cross-season neighbor network.")
    rng = np.random.default_rng(seed)
    queues = {}
    for key, rows in candidates.groupby(["season", "network_archetype"], sort=True):
        values = rows.index.to_numpy(dtype=int)
        queues[key] = list(values[rng.permutation(len(values))])
    selected = []
    seen = set()
    active = list(sorted(queues))
    while active and len(selected) < node_limit:
        next_active = []
        for key in active:
            while queues[key] and candidates.loc[queues[key][-1], "row_id"] in seen:
                queues[key].pop()
            if not queues[key]:
                continue
            index = queues[key].pop()
            selected.append(index)
            seen.add(candidates.loc[index, "row_id"])
            if len(selected) >= node_limit:
                break
            if queues[key]:
                next_active.append(key)
        active = next_active
    return candidates.loc[selected].reset_index(drop=True)


def cross_season_neighbor_network(
    production: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    neighbors_per_node: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Connect each extreme seed to nearest production rows from other seasons."""

    matrix = production[list(LATENT_COLUMNS)].to_numpy(dtype=float)
    normalized = matrix / np.maximum(
        np.linalg.norm(matrix, axis=1, keepdims=True), np.finfo(float).eps
    )
    row_lookup = production.reset_index(drop=True)
    row_index_lookup = {
        str(row_id): int(index) for index, row_id in row_lookup["row_id"].items()
    }
    edges = []
    for season, season_seeds in seeds.groupby("season", sort=True):
        candidate_indices = row_lookup.index[row_lookup["season"] != season].to_numpy(dtype=int)
        if not len(candidate_indices):
            continue
        count = min(neighbors_per_node, len(candidate_indices))
        neighbors = NearestNeighbors(n_neighbors=count, metric="euclidean")
        neighbors.fit(matrix[candidate_indices])
        seed_matrix = season_seeds[list(LATENT_COLUMNS)].to_numpy(dtype=float)
        distances, local_indices = neighbors.kneighbors(seed_matrix)
        for seed_row, row_distances, row_indices in zip(
            season_seeds.itertuples(index=False), distances, local_indices, strict=True
        ):
            source_index = row_index_lookup[row_row_id(row=seed_row)]
            for rank, (distance, local_index) in enumerate(
                zip(row_distances, row_indices, strict=True), start=1
            ):
                target_index = int(candidate_indices[int(local_index)])
                edges.append(
                    {
                        "source_row_id": row_row_id(seed_row),
                        "target_row_id": row_lookup.loc[target_index, "row_id"],
                        "source_season": int(season),
                        "target_season": int(row_lookup.loc[target_index, "season"]),
                        "rank": rank,
                        "euclidean_distance": float(distance),
                        "cosine_similarity": float(
                            normalized[source_index] @ normalized[target_index]
                        ),
                    }
                )
    edges_frame = pd.DataFrame(edges)
    node_ids = set(seeds["row_id"])
    if not edges_frame.empty:
        node_ids.update(edges_frame["target_row_id"])
    nodes = row_lookup[row_lookup["row_id"].isin(node_ids)].copy()
    nodes["is_extreme_seed"] = nodes["row_id"].isin(set(seeds["row_id"]))
    archetype_lookup = seeds.drop_duplicates("row_id").set_index("row_id")["network_archetype"]
    nodes["network_archetype"] = nodes["row_id"].map(archetype_lookup)
    return nodes.reset_index(drop=True), edges_frame


def row_row_id(row: Any) -> str:
    """Read a row ID from a namedtuple without depending on tuple class names."""

    return str(row.row_id)


def write_neighbor_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_path: str | Path,
) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(12, 9))
    lookup = nodes.set_index("row_id")
    for edge in edges.itertuples(index=False):
        source = lookup.loc[edge.source_row_id]
        target = lookup.loc[edge.target_row_id]
        ax.plot(
            [source["pc1"], target["pc1"]],
            [source["pc2"], target["pc2"]],
            color="#94a3b8",
            linewidth=0.45,
            alpha=0.28,
            zorder=1,
        )
    categories = pd.Categorical(nodes["season"].astype(str))
    ax.scatter(
        nodes["pc1"],
        nodes["pc2"],
        c=categories.codes,
        cmap="tab20",
        s=np.where(nodes["is_extreme_seed"], 22, 10),
        alpha=0.65,
        edgecolors=np.where(nodes["is_extreme_seed"], "#0f172a", "none"),
        linewidths=0.35,
        zorder=2,
    )
    ax.set_title("Cross-Season Nearest-Neighbor Network In Production PCA Frame")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True, alpha=0.15)
    _save_plot(plt, fig, Path(output_path))


def _metadata_columns(table: pd.DataFrame) -> list[str]:
    preferred = [
        "row_id",
        "season",
        "event_key",
        "match_key",
        "comp_level",
        "set_number",
        "match_number",
        "alliance",
        "alliance_score",
        "opponent_score",
        "winning_alliance",
        *PERCENTILE_COLUMNS,
        "auto_value",
        "endgame_value",
        "fouls_value",
        *LATENT_COLUMNS,
        "pc1",
        "pc2",
        "pc3",
        "tsne_x",
        "tsne_y",
        "score_decile",
        "is_extreme_seed",
        "archetype",
        "network_archetype",
    ]
    preferred.extend(f"eval_{column}" for column in LATENT_COLUMNS)
    preferred.extend(f"aligned_{column}" for column in LATENT_COLUMNS)
    preferred.extend(f"standardized_{column}" for column in LATENT_COLUMNS)
    return [column for column in preferred if column in table.columns]


def write_match_breakdown_inspection(
    artifact_dir: str | Path,
    output_dir: str | Path,
    options: MatchBreakdownInspectionOptions | None = None,
) -> MatchBreakdownInspectionResult:
    """Write a deterministic static inspection report without mutating the trained bundle."""

    options = options or MatchBreakdownInspectionOptions()
    _validate_options(options)
    artifact = Path(artifact_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = artifact / "bundle.json"
    production_path = artifact / "embeddings.parquet"
    model_path = artifact / "model.pt"
    eval_model_path = artifact / "eval_model.pt"
    for path in (bundle_path, production_path, model_path, eval_model_path):
        if not path.exists():
            raise FileNotFoundError(f"Match-breakdown artifact file does not exist: {path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    alliance_features_path = Path(bundle["alliance_features"]["path"])
    if not alliance_features_path.exists():
        raise FileNotFoundError(
            f"Match-breakdown alliance feature file does not exist: {alliance_features_path}"
        )
    raw = pd.read_parquet(alliance_features_path, engine=PARQUET_ENGINE)
    production = pd.read_parquet(production_path, engine=PARQUET_ENGINE)
    if production["row_id"].duplicated().any() or raw["row_id"].duplicated().any():
        raise ValueError("Match-breakdown inspection requires unique alliance row IDs.")
    if set(production["row_id"]) != set(raw["row_id"]):
        raise ValueError(
            "Production embeddings and alliance features must contain identical row IDs."
        )
    _, production_checkpoint = load_match_breakdown_model(model_path)
    holdout, validation = holdout_embeddings(raw, eval_model_path)
    selected_sources = select_component_sources(raw)
    source_payload = component_metric_source_payload(selected_sources)
    _json_write(output / "component_metric_sources.json", source_payload)
    production = add_component_percentiles(production, raw, selected_sources)
    holdout = add_component_percentiles(holdout, raw, selected_sources)
    production, holdout, explained, alignment = shared_pca_coordinates(production, holdout)
    production[_metadata_columns(production)].to_csv(
        output / "production_pca_coordinates.csv", index=False
    )
    holdout[_metadata_columns(holdout)].to_csv(output / "holdout_pca_coordinates.csv", index=False)
    explained.to_csv(output / "pca_explained_variance.csv", index=False)
    write_pca_explained_variance(explained, output / "pca_explained_variance.png")
    write_projection_multiview(
        production,
        output / "production_pca_multiview.png",
        x="pc1",
        y="pc2",
        axis_prefix="PCA",
        title="Production Match-Breakdown Latents",
    )
    write_projection_multiview(
        holdout,
        output / "holdout_pca_multiview.png",
        x="pc1",
        y="pc2",
        axis_prefix="PCA",
        title="Eval-Model Holdout Latents Aligned To Production PCA",
    )
    production_sample = stratified_projection_sample(
        production,
        max_rows=options.tsne_max_rows,
        per_season_limit=options.production_tsne_per_season,
        seed=options.seed,
    )
    holdout_sample = stratified_projection_sample(
        holdout,
        max_rows=min(
            options.tsne_max_rows,
            options.holdout_tsne_per_season * holdout["season"].nunique(),
        ),
        per_season_limit=options.holdout_tsne_per_season,
        seed=options.seed + 1,
    )
    production_tsne = tsne_coordinates(
        production_sample, seed=options.seed, iterations=options.tsne_iterations
    )
    holdout_tsne = tsne_coordinates(
        holdout_sample, seed=options.seed + 1, iterations=options.tsne_iterations
    )
    production_tsne[_metadata_columns(production_tsne)].to_csv(
        output / "production_tsne_coordinates.csv", index=False
    )
    holdout_tsne[_metadata_columns(holdout_tsne)].to_csv(
        output / "holdout_tsne_coordinates.csv", index=False
    )
    write_projection_multiview(
        production_tsne,
        output / "production_tsne_multiview.png",
        x="tsne_x",
        y="tsne_y",
        axis_prefix="t-SNE",
        title="Production Match-Breakdown Local Neighborhoods",
    )
    write_projection_multiview(
        holdout_tsne,
        output / "holdout_tsne_multiview.png",
        x="tsne_x",
        y="tsne_y",
        axis_prefix="t-SNE",
        title="Eval-Model Holdout Local Neighborhoods",
    )
    profiles = archetype_profiles(
        production, rows_per_slice=options.parallel_rows_per_slice, seed=options.seed
    )
    profiles[_metadata_columns(profiles)].to_csv(
        output / "latent_archetype_profiles.csv", index=False
    )
    write_parallel_coordinates(profiles, output / "latent_parallel_coordinates.png")
    seeds = extreme_network_seeds(
        production, node_limit=options.network_node_limit, seed=options.seed
    )
    nodes, edges = cross_season_neighbor_network(
        production, seeds, neighbors_per_node=options.neighbors_per_node
    )
    nodes[_metadata_columns(nodes)].to_csv(output / "cross_season_neighbor_nodes.csv", index=False)
    edges.to_csv(output / "cross_season_neighbor_edges.csv", index=False)
    write_neighbor_network(nodes, edges, output / "cross_season_neighbor_network.png")
    generated = {
        name: _sha256_file(output / name)
        for name in EXPECTED_REPORT_FILES
        if (output / name).exists()
    }
    missing_outputs = sorted(set(EXPECTED_REPORT_FILES) - set(generated))
    if missing_outputs:
        raise RuntimeError(f"Inspection report did not write files: {', '.join(missing_outputs)}.")
    manifest = {
        "schema_version": 1,
        "kind": "v6-lite-match-breakdown-inspection",
        "note": (
            "Projection plots are interpretation aids, not standalone promotion evidence. "
            "Game-specific season structure may remain visible despite the shared bottleneck."
        ),
        "artifact_dir": str(artifact),
        "options": asdict(options),
        "inputs": {
            "bundle.json": _sha256_file(bundle_path),
            "embeddings.parquet": _sha256_file(production_path),
            "model.pt": _sha256_file(model_path),
            "eval_model.pt": _sha256_file(eval_model_path),
            str(alliance_features_path): _sha256_file(alliance_features_path),
        },
        "geometry": {
            "production": {
                "source": "embeddings.parquet exported from all-data model.pt",
                "rows": len(production),
                "checkpoint_provenance": production_checkpoint["provenance"],
            },
            "holdout": {
                "source": "eval_model.pt deterministic match-grouped holdout rows",
                "rows": len(holdout),
                "match_keys": int(holdout["match_key"].nunique()),
                "split_rows_by_season": {
                    str(season): int(len(indices)) for season, indices in sorted(validation.items())
                },
                "display_alignment": alignment,
            },
        },
        "generated_files": generated,
    }
    _json_write(output / "inspection_manifest.json", manifest)
    return MatchBreakdownInspectionResult(output_dir=output, manifest=manifest)


__all__ = [
    "COMPONENT_ALIASES",
    "EXPECTED_REPORT_FILES",
    "LATENT_COLUMNS",
    "MatchBreakdownInspectionOptions",
    "MatchBreakdownInspectionResult",
    "add_component_percentiles",
    "component_metric_source_payload",
    "cross_season_neighbor_network",
    "extreme_network_seeds",
    "holdout_embeddings",
    "load_match_breakdown_model",
    "select_component_sources",
    "shared_pca_coordinates",
    "stratified_projection_sample",
    "tsne_coordinates",
    "write_match_breakdown_inspection",
]
