"""Inspection and visualization utilities for V5.5 prior checkpoints."""

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
from latentstrat.prior_model import UnifiedPriorAutoencoder


@dataclass
class PriorInspection:
    latent_table: pd.DataFrame
    nearest_neighbors: pd.DataFrame
    training_history: pd.DataFrame
    sanity_checks: pd.DataFrame
    pca_model: dict[str, np.ndarray]


def _checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "team_vectors" not in checkpoint:
        raise KeyError("Prior checkpoint is missing team_vectors.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Prior checkpoint is missing model_state_dict.")
    return checkpoint


def _prior_opts(checkpoint: dict[str, Any]) -> PriorOpts:
    return PriorOpts.model_validate(checkpoint.get("prior_opts", {}))


def _team_vectors(checkpoint: dict[str, Any], opts: PriorOpts) -> dict[str, torch.Tensor]:
    vectors = {}
    for team_key, value in checkpoint["team_vectors"].items():
        vector = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
        if int(vector.numel()) != opts.latent_dim:
            raise ValueError(
                f"Prior vector for {team_key} has width {vector.numel()}, "
                f"expected {opts.latent_dim}."
            )
        vectors[str(team_key)] = vector
    return vectors


def _reconstruction_mse(
    checkpoint: dict[str, Any],
    features: pd.DataFrame,
    opts: PriorOpts,
) -> dict[str, float]:
    dataset = PriorTensorDataset(features, opts)
    model = UnifiedPriorAutoencoder(opts)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        reconstruction = model.decoder(model.encoder(dataset.vectors))
        mse = F.mse_loss(reconstruction, dataset.vectors, reduction="none").mean(dim=1)
    return {
        team_key: float(mse[idx].detach().cpu())
        for idx, team_key in enumerate(dataset.team_keys)
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
            bool(np.isfinite(latent_table["reconstruction_mse"]).all()),
            float(np.nanmax(latent_table["reconstruction_mse"])),
            np.inf,
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "passed", "value", "threshold"])


def inspect_prior_checkpoint(
    checkpoint_path: str | Path,
    features_path: str | Path,
    *,
    top_k: int = 10,
) -> PriorInspection:
    checkpoint = _checkpoint(checkpoint_path)
    opts = _prior_opts(checkpoint)
    features = read_prior_feature_table(features_path)
    vectors = _team_vectors(checkpoint, opts)
    reconstruction = _reconstruction_mse(checkpoint, features, opts)
    rows = []
    for team_key in sorted(vectors):
        vector = vectors[team_key].numpy()
        rows.append(
            {
                "team_key": team_key,
                "embedding": vector.tolist(),
                "embedding_norm": float(np.linalg.norm(vector)),
                "reconstruction_mse": reconstruction.get(team_key, np.nan),
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
    ax.set_title(f"V5.5 Prior Latent Space: {x_column.upper()} vs {y_column.upper()}")
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
    ax.set_title("V5.5 Prior Embedding Norms")
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
    ax.set_title("V5.5 Prior Autoencoder Training Loss")
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
    _plot_norm_hist(inspection.latent_table, out / "prior_embedding_norm_hist.png")
    _plot_training_loss(inspection.training_history, out / "prior_training_loss.png")
    return out
