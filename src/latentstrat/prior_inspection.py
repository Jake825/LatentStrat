"""Inspection and visualization utilities for V5.6 prior checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from latentstrat.config import PriorOpts
from latentstrat.inspection import compute_pca, normalize_rows
from latentstrat.pretrain_features import read_prior_feature_table
from latentstrat.pretrain_loop import PriorTensorDataset
from latentstrat.prior_model import TeamPriorDistiller


@dataclass
class PriorInspection:
    latent_table: pd.DataFrame
    nearest_neighbors: pd.DataFrame
    training_history: pd.DataFrame
    sanity_checks: pd.DataFrame
    pca_model: dict[str, np.ndarray]


def _checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "embedding_table" not in checkpoint:
        raise KeyError("Prior checkpoint is missing embedding_table.")
    return checkpoint


def _prior_opts(checkpoint: dict[str, Any]) -> PriorOpts:
    if checkpoint.get("prior_opts"):
        return PriorOpts.model_validate(checkpoint["prior_opts"])
    table = torch.as_tensor(checkpoint["embedding_table"], dtype=torch.float32)
    return PriorOpts(max_team_number=int(table.shape[0]) - 1, latent_dim=int(table.shape[1]))


def _embedding_table(checkpoint: dict[str, Any], opts: PriorOpts) -> torch.Tensor:
    table = torch.as_tensor(checkpoint["embedding_table"], dtype=torch.float32).detach().cpu()
    expected_shape = (opts.max_team_number + 1, opts.latent_dim)
    if tuple(table.shape) != expected_shape:
        raise ValueError(
            f"Prior embedding_table has shape {tuple(table.shape)}, "
            f"expected {expected_shape}."
        )
    return table


def _reconstruction_mse(
    checkpoint: dict[str, Any],
    features: pd.DataFrame | None,
    opts: PriorOpts,
) -> dict[int, float]:
    if features is None or "model_state_dict" not in checkpoint:
        return {}
    dataset = PriorTensorDataset(features, opts)
    model = TeamPriorDistiller(opts)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        reconstruction, _, _ = model(dataset.team_numbers)
        mse = F.mse_loss(reconstruction, dataset.vectors, reduction="none").mean(dim=1)
    return {
        int(team_number): float(mse[idx].detach().cpu())
        for idx, team_number in enumerate(dataset.team_numbers.tolist())
    }


def _nearest_neighbors(
    team_table: pd.DataFrame, normalized: np.ndarray, top_k: int
) -> pd.DataFrame:
    if len(team_table) <= 1:
        return pd.DataFrame(
            columns=[
                "query_team_key",
                "neighbor_team_key",
                "rank",
                "cosine_similarity",
            ]
        )
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
                    "neighbor_team_key": team_table["team_key"].iloc[neighbor_row],
                    "rank": rank,
                    "cosine_similarity": float(similarities[neighbor_row]),
                }
            )
    return pd.DataFrame(rows)


def _sanity_checks(vectors: np.ndarray, latent_table: pd.DataFrame) -> pd.DataFrame:
    norms = latent_table["embedding_norm"].to_numpy(dtype=float)
    reconstruction = latent_table["reconstruction_mse"]
    known_reconstruction = reconstruction[reconstruction.notna()]
    reconstruction_ok = (
        True if known_reconstruction.empty else bool(np.isfinite(known_reconstruction).all())
    )
    reconstruction_value = (
        np.nan if known_reconstruction.empty else float(np.nanmax(known_reconstruction))
    )
    variances = (
        np.nanvar(vectors, axis=0, ddof=1)
        if len(vectors) > 1
        else np.zeros(vectors.shape[1])
    )
    dominant_share = np.max(variances) / max(np.sum(variances), np.finfo(float).eps)
    rows = [
        ("all_finite", bool(np.isfinite(vectors).all()), float(np.isfinite(vectors).all()), 1.0),
        (
            "no_zero_norms",
            bool(np.all(norms > np.finfo(float).eps)),
            float(np.nanmin(norms)) if len(norms) else np.nan,
            float(np.finfo(float).eps),
        ),
        (
            "no_collapsed_norm_distribution",
            bool(len(norms) <= 1 or np.nanstd(norms) > np.finfo(float).eps),
            float(np.nanstd(norms)) if len(norms) else np.nan,
            float(np.finfo(float).eps),
        ),
        (
            "no_single_dominant_dimension",
            bool(dominant_share < 0.90),
            float(dominant_share),
            0.90,
        ),
        (
            "finite_reconstruction_mse",
            reconstruction_ok,
            reconstruction_value,
            np.inf,
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "passed", "value", "threshold"])


def _feature_lookup(features: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    if features is None:
        return lookup
    for _, row in features.iterrows():
        number = int(row["team_number"])
        lookup[number] = {
            "team_key": str(row.get("team_key", f"frc{number}")),
            "archetype": str(row.get("archetype", "")),
            "narrative_hash": str(row.get("narrative_hash", "")),
            "raw_epa": row.get("raw_epa", np.nan),
            "target_epa": row.get("target_epa", np.nan),
            "epa_source_year": row.get("epa_source_year", np.nan),
            "epa_is_imputed": row.get("epa_is_imputed", np.nan),
        }
    return lookup


def inspect_prior_checkpoint(
    checkpoint_path: str | Path,
    features_path: str | Path | None = None,
    *,
    top_k: int = 10,
) -> PriorInspection:
    checkpoint = _checkpoint(checkpoint_path)
    opts = _prior_opts(checkpoint)
    features = read_prior_feature_table(features_path) if features_path is not None else None
    embedding_table = _embedding_table(checkpoint, opts)
    reconstruction = _reconstruction_mse(checkpoint, features, opts)
    lookup = _feature_lookup(features)
    rows = []
    for team_number in range(0, opts.max_team_number + 1):
        vector = embedding_table[team_number].numpy()
        feature = lookup.get(team_number, {})
        rows.append(
            {
                "team_number": team_number,
                "team_key": feature.get("team_key", f"frc{team_number}"),
                "archetype": feature.get("archetype", ""),
                "narrative_hash": feature.get("narrative_hash", ""),
                "raw_epa": feature.get("raw_epa", np.nan),
                "target_epa": feature.get("target_epa", np.nan),
                "epa_source_year": feature.get("epa_source_year", np.nan),
                "epa_is_imputed": feature.get("epa_is_imputed", np.nan),
                "embedding": vector.tolist(),
                "embedding_norm": float(np.linalg.norm(vector)),
                "reconstruction_mse": reconstruction.get(team_number, np.nan),
            }
        )
    latent_table = pd.DataFrame(rows)
    vector_matrix = np.stack(latent_table["embedding"].map(np.asarray).to_list(), axis=0)
    pca_score, pca_model = compute_pca(vector_matrix)
    latent_table["pc1"] = pca_score[:, 0]
    latent_table["pc2"] = pca_score[:, 1]
    latent_table["pc3"] = pca_score[:, 2]
    normalized = normalize_rows(vector_matrix)
    history = pd.DataFrame(checkpoint.get("history", []))
    return PriorInspection(
        latent_table=latent_table,
        nearest_neighbors=_nearest_neighbors(latent_table, normalized, top_k),
        training_history=history,
        sanity_checks=_sanity_checks(vector_matrix, latent_table),
        pca_model=pca_model,
    )


def _label_teams(latent_table: pd.DataFrame) -> set[str]:
    labels = {"frc2290"}
    top_norms = latent_table.sort_values("embedding_norm", ascending=False).head(10)
    labels.update(top_norms["team_key"].astype(str).tolist())
    return labels


def _plot_pca(
    latent_table: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = _label_teams(latent_table)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        latent_table[x_column],
        latent_table[y_column],
        s=18,
        alpha=0.72,
        c=latent_table["embedding_norm"],
        cmap="viridis",
    )
    for _, row in latent_table.iterrows():
        if str(row["team_key"]) in labels:
            ax.annotate(
                str(row["team_key"]),
                (row[x_column], row[y_column]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel(x_column.upper())
    ax.set_ylabel(y_column.upper())
    ax.set_title(f"V5.6 Prior Latent Space: {x_column.upper()} vs {y_column.upper()}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_norm_hist(latent_table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latent_table["embedding_norm"], bins=30, color="#3b82f6", alpha=0.85)
    ax.set_xlabel("Embedding L2 norm")
    ax.set_ylabel("Team count")
    ax.set_title("V5.6 Prior Embedding Norms")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_pca_epa(latent_table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = pd.to_numeric(latent_table.get("target_epa"), errors="coerce")
    finite = values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    colors = values if finite.any() else latent_table["embedding_norm"]
    label = "Normalized EPA target" if finite.any() else "Embedding L2 norm"
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        latent_table["pc1"],
        latent_table["pc2"],
        s=18,
        alpha=0.72,
        c=colors,
        cmap="coolwarm" if finite.any() else "viridis",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("V5.6.1 Prior Latent Space Colored By EPA")
    fig.colorbar(scatter, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_training_loss(history: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    if {"epoch", "train_loss"}.issubset(history.columns) and not history.empty:
        ax.plot(history["epoch"], history["train_loss"], color="#0f766e", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("V5.6 Prior Distiller Training Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_prior_inspection_artifacts(
    inspection: PriorInspection,
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inspection.latent_table.to_csv(out / "prior_latent_table.csv", index=False)
    inspection.nearest_neighbors.to_csv(out / "prior_nearest_neighbors.csv", index=False)
    inspection.training_history.to_csv(out / "prior_training_history.csv", index=False)
    inspection.sanity_checks.to_csv(out / "prior_sanity_checks.csv", index=False)
    _plot_pca(inspection.latent_table, "pc1", "pc2", out / "prior_pca_pc1_pc2.png")
    _plot_pca(inspection.latent_table, "pc1", "pc3", out / "prior_pca_pc1_pc3.png")
    _plot_pca_epa(inspection.latent_table, out / "prior_pca_epa_pc1_pc2.png")
    _plot_norm_hist(inspection.latent_table, out / "prior_embedding_norm_hist.png")
    _plot_training_loss(inspection.training_history, out / "prior_training_loss.png")
    return out
