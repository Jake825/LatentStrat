"""Static visualization helpers for LatentStrat phase artifacts."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from latentstrat.inspection import compute_pca
from latentstrat.pretraining.prior.train import load_prior_embedding_table


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_plot(plt, fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _coerce_vector(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list | tuple):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=float)
    return np.asarray(value, dtype=float)


def embedding_matrix_from_table(table: pd.DataFrame) -> np.ndarray:
    if table.empty or "embedding" not in table.columns:
        return np.empty((0, 0), dtype=float)
    return np.stack([_coerce_vector(value) for value in table["embedding"]], axis=0)


def _finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]


def tsne_coordinates(
    latent_table: pd.DataFrame,
    *,
    random_seed: int,
) -> pd.DataFrame:
    matrix = embedding_matrix_from_table(latent_table)
    columns = [
        "team_number",
        "team_key",
        "archetype",
        "embedding_norm",
        "norm_epa_t_minus_1",
        "target_epa",
        "raw_epa",
        "raw_rookie_year_delta",
        "tsne_x",
        "tsne_y",
    ]
    if matrix.size == 0:
        return pd.DataFrame(columns=columns)
    if len(matrix) < 3:
        coords = np.zeros((len(matrix), 2), dtype=float)
        if len(matrix) == 2:
            coords[:, 0] = [-0.5, 0.5]
    else:
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(1.0, float(len(matrix) - 1) / 3.0))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=random_seed,
        ).fit_transform(matrix)
    out = latent_table[
        [column for column in columns if column in latent_table.columns and column != "tsne_x"]
    ].copy()
    out["tsne_x"] = coords[:, 0]
    out["tsne_y"] = coords[:, 1]
    return out


def latent_stat_correlations(latent_table: pd.DataFrame) -> pd.DataFrame:
    matrix = embedding_matrix_from_table(latent_table)
    columns = ["stat", "latent_dim", "correlation", "count"]
    if matrix.size == 0:
        return pd.DataFrame(columns=columns)
    stat_columns = [
        column
        for column in latent_table.columns
        if column.startswith(("raw_", "norm_epa_")) or column in {"target_epa", "raw_epa"}
    ]
    rows = []
    for stat in stat_columns:
        values = pd.to_numeric(latent_table[stat], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if np.sum(finite) < 3 or np.nanstd(values[finite]) <= np.finfo(float).eps:
            continue
        for dim in range(matrix.shape[1]):
            latent_values = matrix[:, dim]
            mask = finite & np.isfinite(latent_values)
            if np.sum(mask) < 3 or np.nanstd(latent_values[mask]) <= np.finfo(float).eps:
                correlation = np.nan
            else:
                correlation = float(np.corrcoef(latent_values[mask], values[mask])[0, 1])
            rows.append(
                {
                    "stat": stat,
                    "latent_dim": f"z{dim:02d}",
                    "correlation": correlation,
                    "count": int(np.sum(mask)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _scatter_continuous(ax, frame: pd.DataFrame, column: str, title: str):
    source = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
    values = pd.to_numeric(source, errors="coerce")
    finite = values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    colors = values if finite.any() else frame["embedding_norm"]
    label = column if finite.any() else "embedding_norm"
    scatter = ax.scatter(
        frame["tsne_x"],
        frame["tsne_y"],
        c=colors,
        cmap="coolwarm" if finite.any() else "viridis",
        s=16,
        alpha=0.72,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    return scatter, label


def write_prior_team_galaxy_plot(coords: pd.DataFrame, output_path: Path) -> None:
    if coords.empty:
        return
    plt = _plot_setup()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scatter, label = _scatter_continuous(
        axes[0], coords, "norm_epa_t_minus_1", "Latest EPA"
    )
    fig.colorbar(scatter, ax=axes[0], label=label)
    scatter, label = _scatter_continuous(
        axes[1], coords, "raw_rookie_year_delta", "Rookie-Year Delta"
    )
    fig.colorbar(scatter, ax=axes[1], label=label)
    archetype = pd.Categorical(coords.get("archetype", pd.Series([""] * len(coords))).fillna(""))
    scatter = axes[2].scatter(
        coords["tsne_x"],
        coords["tsne_y"],
        c=archetype.codes,
        cmap="tab20",
        s=16,
        alpha=0.72,
    )
    axes[2].set_title("Archetype")
    axes[2].set_xlabel("t-SNE 1")
    axes[2].set_ylabel("t-SNE 2")
    categories = list(archetype.categories)
    if categories:
        colorbar = fig.colorbar(scatter, ax=axes[2], ticks=range(len(categories)))
        colorbar.ax.set_yticklabels(categories)
    _save_plot(plt, fig, output_path)


def write_correlation_heatmap(correlations: pd.DataFrame, output_path: Path) -> None:
    finite = correlations.dropna(subset=["correlation"]) if not correlations.empty else correlations
    if finite.empty:
        return
    matrix = finite.pivot(index="stat", columns="latent_dim", values="correlation")
    plt = _plot_setup()
    fig_height = max(5, 0.35 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Prior Latent Dimension Correlations")
    fig.colorbar(image, ax=ax, label="Pearson r")
    _save_plot(plt, fig, output_path)


def write_prior_phase_visuals(
    latent_table: pd.DataFrame,
    output_dir: str | Path,
    *,
    random_seed: int,
) -> None:
    out = Path(output_dir)
    coords = tsne_coordinates(latent_table, random_seed=random_seed)
    coords.to_csv(out / "prior_tsne_coordinates.csv", index=False)
    write_prior_team_galaxy_plot(coords, out / "prior_tsne_team_galaxy.png")
    correlations = latent_stat_correlations(latent_table)
    correlations.to_csv(out / "prior_latent_stat_correlations.csv", index=False)
    write_correlation_heatmap(correlations, out / "prior_latent_stat_correlation_heatmap.png")


def _prior_inspection_candidates(prior_checkpoint: str | Path) -> list[Path]:
    checkpoint = Path(prior_checkpoint)
    return [
        checkpoint.parent / "inspection" / "prior_latent_table.csv",
        checkpoint.parent / "prior_latent_table.csv",
    ]


def prior_lookup_table(prior_checkpoint: str | Path | None) -> pd.DataFrame:
    columns = [
        "team_number",
        "team_key",
        "prior_embedding_norm",
        "norm_epa_t_minus_1",
        "target_epa",
        "raw_epa",
        "archetype",
    ]
    if prior_checkpoint is None:
        return pd.DataFrame(columns=columns)
    prior = load_prior_embedding_table(prior_checkpoint).detach().cpu().numpy()
    lookup = pd.DataFrame(
        {
            "team_number": np.arange(prior.shape[0], dtype=int),
            "team_key": [f"frc{idx}" for idx in range(prior.shape[0])],
            "prior_embedding_norm": np.linalg.norm(prior, axis=1),
        }
    )
    for candidate in _prior_inspection_candidates(prior_checkpoint):
        if not candidate.exists():
            continue
        latent = pd.read_csv(candidate)
        keep = [column for column in columns if column in latent.columns]
        if "team_number" not in keep:
            continue
        lookup = lookup.merge(
            latent[keep].drop_duplicates(subset=["team_number"]),
            on="team_number",
            how="left",
            suffixes=("", "_latent"),
        )
        if "team_key_latent" in lookup.columns:
            lookup["team_key"] = lookup["team_key_latent"].fillna(lookup["team_key"])
            lookup = lookup.drop(columns=["team_key_latent"])
        break
    return lookup


def _team_number_from_key(team_key: str) -> int:
    try:
        return int(str(team_key).removeprefix("frc"))
    except ValueError:
        return 0


def team_attention_summary(attention: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "team_key",
        "team_number",
        "appearance_count",
        "validation_appearance_count",
        "event_count",
        "avg_attention",
        "median_attention",
        "max_attention_rate",
    ]
    if attention.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in attention.iterrows():
        max_team = str(row.get("max_attention_team_key", ""))
        for slot in (1, 2, 3):
            team_key = str(row.get(f"team_{slot}_key", ""))
            if not team_key or team_key.lower() == "nan" or bool(row.get(f"team_{slot}_missing")):
                continue
            rows.append(
                {
                    "team_key": team_key,
                    "team_number": _team_number_from_key(team_key),
                    "event_key": str(row.get("event_key", "")),
                    "split": str(row.get("split", "")),
                    "attention": float(row.get(f"team_{slot}_attention", np.nan)),
                    "is_max_attention": team_key == max_team,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    long = pd.DataFrame(rows)
    grouped = (
        long.groupby(["team_key", "team_number"], as_index=False)
        .agg(
            appearance_count=("attention", "size"),
            event_count=("event_key", "nunique"),
            avg_attention=("attention", "mean"),
            median_attention=("attention", "median"),
            max_attention_rate=("is_max_attention", "mean"),
        )
        .sort_values("avg_attention", ascending=False)
    )
    validation = (
        long[long["split"] == "validation"]
        .groupby("team_key")
        .size()
        .rename("validation_appearance_count")
    )
    grouped = grouped.merge(validation, on="team_key", how="left")
    grouped["validation_appearance_count"] = (
        grouped["validation_appearance_count"].fillna(0).astype(int)
    )
    return grouped[columns]


def _merge_prior_metric(frame: pd.DataFrame, prior_lookup: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or prior_lookup.empty:
        return pd.DataFrame()
    return frame.merge(prior_lookup, on="team_number", how="left", suffixes=("", "_prior"))


def _prior_x_column(frame: pd.DataFrame) -> tuple[str, str] | None:
    candidates = [
        ("norm_epa_t_minus_1", "Latest normalized EPA"),
        ("target_epa", "Target EPA"),
        ("raw_epa", "Raw EPA"),
        ("prior_embedding_norm", "Prior embedding norm"),
    ]
    for column, label in candidates:
        if column not in frame.columns:
            continue
        values = _finite_numeric(frame[column])
        if not values.empty:
            return column, label
    return None


def write_pma_carry_support_plot(
    attention_summary: pd.DataFrame,
    prior_lookup: pd.DataFrame,
    output_path: Path,
) -> None:
    merged = _merge_prior_metric(attention_summary, prior_lookup)
    selected = _prior_x_column(merged)
    if merged.empty or selected is None:
        return
    x_column, x_label = selected
    plot = merged.dropna(subset=[x_column, "avg_attention"])
    if plot.empty:
        return
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        plot[x_column],
        plot["avg_attention"],
        c=plot["appearance_count"],
        cmap="viridis",
        s=28,
        alpha=0.76,
    )
    ax.axhline(1 / 3, linestyle="--", color="#666666", linewidth=1, label="equal share")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Average PMA attention")
    ax.set_title("PMA Carry vs Support")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Alliance appearances")
    _save_plot(plt, fig, output_path)


def write_task_precision_plot(history: pd.DataFrame, output_path: Path) -> None:
    precision_columns = [column for column in history.columns if column.endswith("_precision")]
    if history.empty or "epoch" not in history.columns or not precision_columns:
        return
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    for column in precision_columns:
        values = pd.to_numeric(history[column], errors="coerce")
        if values.notna().any():
            ax.plot(history["epoch"], values, label=column.removesuffix("_precision"))
    ax.set_title("Task Precision Evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("exp(-log_var)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    _save_plot(plt, fig, output_path)


def write_zero_out_war_plot(
    sensitivity: pd.DataFrame,
    prior_lookup: pd.DataFrame,
    output_path: Path,
) -> None:
    if sensitivity.empty:
        return
    target = (
        "alliance_phase_score"
        if "alliance_phase_score" in set(sensitivity["target"])
        else str(sensitivity["target"].iloc[0])
    )
    plot = sensitivity[sensitivity["target"] == target].copy()
    plot["team_number"] = plot["team_base_idx"].astype(int)
    merged = _merge_prior_metric(plot, prior_lookup)
    selected = _prior_x_column(merged)
    if merged.empty or selected is None:
        return
    x_column, x_label = selected
    merged = merged.dropna(subset=[x_column, "delta_rmse"])
    if merged.empty:
        return
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        merged[x_column],
        merged["delta_rmse"],
        c=merged["appearance_count"],
        cmap="magma",
        s=32,
        alpha=0.78,
    )
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Zero-out delta RMSE")
    ax.set_title("Zero-Out Wins Above Replacement")
    fig.colorbar(scatter, ax=ax, label="Validation appearances")
    _save_plot(plt, fig, output_path)


def _event_week_lookup(prepared: pd.DataFrame) -> dict[str, float]:
    if "event_key" not in prepared.columns or "event_week" not in prepared.columns:
        return {}
    weeks = (
        prepared[["event_key", "event_week"]]
        .dropna(subset=["event_key"])
        .drop_duplicates(subset=["event_key"])
    )
    return {str(row["event_key"]): row["event_week"] for _, row in weeks.iterrows()}


def event_delta_summary(
    model,
    team_event_index_map: dict[str, int],
    prepared: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["event_key", "team_key", "team_event_idx", "event_week", "event_delta_norm"]
    if not team_event_index_map:
        return pd.DataFrame(columns=columns)
    event_weight = model.Z_event.weight.detach().cpu().numpy()
    week_lookup = _event_week_lookup(prepared)
    rows = []
    for pair_key, event_idx in sorted(team_event_index_map.items(), key=lambda item: item[1]):
        if "::" not in pair_key or int(event_idx) >= len(event_weight):
            continue
        event_key, team_key = str(pair_key).split("::", 1)
        rows.append(
            {
                "event_key": event_key,
                "team_key": team_key,
                "team_event_idx": int(event_idx),
                "event_week": week_lookup.get(event_key, np.nan),
                "event_delta_norm": float(np.linalg.norm(event_weight[int(event_idx)])),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_event_delta_by_week_plot(delta_summary: pd.DataFrame, output_path: Path) -> None:
    if delta_summary.empty or "event_week" not in delta_summary.columns:
        return
    frame = delta_summary.dropna(subset=["event_week", "event_delta_norm"]).copy()
    if frame.empty:
        return
    frame["event_week"] = pd.to_numeric(frame["event_week"], errors="coerce")
    groups = [
        values["event_delta_norm"].to_numpy(dtype=float)
        for _, values in frame.groupby("event_week", sort=True)
    ]
    labels = [str(week) for week in sorted(frame["event_week"].dropna().unique())]
    if not groups:
        return
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_title("Event Delta Magnitude By Week")
    ax.set_xlabel("Event week")
    ax.set_ylabel("||Z_event||2")
    ax.grid(True, axis="y", alpha=0.25)
    _save_plot(plt, fig, output_path)


def embedding_drift_table(
    model,
    team_index_map: dict[str, int],
    team_event_index_map: dict[str, int],
    prepared: pd.DataFrame,
    prior_checkpoint: str | Path | None,
) -> pd.DataFrame:
    columns = [
        "event_key",
        "team_key",
        "team_base_idx",
        "team_event_idx",
        "event_week",
        "prior_norm",
        "final_norm",
        "event_delta_norm",
        "drift_norm",
        "pca_start_x",
        "pca_start_y",
        "pca_final_x",
        "pca_final_y",
    ]
    if prior_checkpoint is None or not team_event_index_map:
        return pd.DataFrame(columns=columns)
    prior = load_prior_embedding_table(prior_checkpoint).detach().cpu().numpy()
    base = model.Z_base.weight.detach().cpu().numpy()
    event = model.Z_event.weight.detach().cpu().numpy()
    week_lookup = _event_week_lookup(prepared)
    rows = []
    prior_vectors = []
    final_vectors = []
    for pair_key, event_idx in sorted(team_event_index_map.items(), key=lambda item: item[1]):
        if "::" not in pair_key:
            continue
        event_key, team_key = str(pair_key).split("::", 1)
        base_idx = int(team_index_map.get(team_key, _team_number_from_key(team_key)))
        event_idx = int(event_idx)
        if base_idx >= len(prior) or base_idx >= len(base) or event_idx >= len(event):
            continue
        prior_vector = prior[base_idx]
        final_vector = base[base_idx] + event[event_idx]
        rows.append(
            {
                "event_key": event_key,
                "team_key": team_key,
                "team_base_idx": base_idx,
                "team_event_idx": event_idx,
                "event_week": week_lookup.get(event_key, np.nan),
                "prior_norm": float(np.linalg.norm(prior_vector)),
                "final_norm": float(np.linalg.norm(final_vector)),
                "event_delta_norm": float(np.linalg.norm(event[event_idx])),
                "drift_norm": float(np.linalg.norm(final_vector - prior_vector)),
            }
        )
        prior_vectors.append(prior_vector)
        final_vectors.append(final_vector)
    if not rows:
        return pd.DataFrame(columns=columns)
    vectors = np.vstack([prior_vectors, final_vectors])
    score, _ = compute_pca(vectors)
    count = len(rows)
    for idx, row in enumerate(rows):
        row["pca_start_x"] = float(score[idx, 0])
        row["pca_start_y"] = float(score[idx, 1])
        row["pca_final_x"] = float(score[idx + count, 0])
        row["pca_final_y"] = float(score[idx + count, 1])
    return pd.DataFrame(rows, columns=columns)


def write_embedding_drift_plot(drift: pd.DataFrame, output_path: Path) -> None:
    if drift.empty:
        return
    plot = drift.sort_values("drift_norm", ascending=False)
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.quiver(
        plot["pca_start_x"],
        plot["pca_start_y"],
        plot["pca_final_x"] - plot["pca_start_x"],
        plot["pca_final_y"] - plot["pca_start_y"],
        plot["drift_norm"],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=1,
        alpha=0.45,
        width=0.0025,
    )
    for _, row in plot.head(10).iterrows():
        ax.annotate(
            str(row["team_key"]),
            (row["pca_final_x"], row["pca_final_y"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Embedding Drift: Prior To Final Team-Event Vector")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True, alpha=0.2)
    _save_plot(plt, fig, output_path)


def selection_value_table(model, selections: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "event_key",
        "event_week",
        "alliance_number",
        "pick_order",
        "actual_pick_number",
        "captain_team_key",
        "pick_team_key",
        "passed_over_team_key",
        "pick_base_idx",
        "pick_event_idx",
        "team_value",
    ]
    if selections is None or selections.empty:
        return pd.DataFrame(columns=columns)
    required = {"pick_base_idx", "pick_event_idx", "pick_team_key", "event_key", "alliance_number"}
    if not required.issubset(selections.columns):
        return pd.DataFrame(columns=columns)
    frame = selections.copy()
    if "pick_order" not in frame.columns:
        frame["pick_order"] = 1
    alliance_counts = frame.groupby("event_key")["alliance_number"].max().to_dict()

    def pick_number(row) -> int:
        alliance_count = max(int(alliance_counts.get(row["event_key"], 0)), 1)
        alliance = int(row["alliance_number"])
        pick_order = int(row["pick_order"])
        round_offset = (pick_order - 1) * alliance_count
        if pick_order % 2 == 1:
            return round_offset + alliance
        return round_offset + alliance_count - alliance + 1

    frame["actual_pick_number"] = [pick_number(row) for _, row in frame.iterrows()]
    device = next(model.parameters()).device
    base_idx = torch.as_tensor(
        frame["pick_base_idx"].to_numpy(dtype=int, copy=True), device=device
    )
    event_idx = torch.as_tensor(
        frame["pick_event_idx"].to_numpy(dtype=int, copy=True), device=device
    )
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        values = model.team_value(base_idx, event_idx).detach().cpu().numpy()
    if was_training:
        model.train()
    frame["team_value"] = values
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns].sort_values(["event_key", "actual_pick_number"])


def write_selection_value_plot(selection_values: pd.DataFrame, output_path: Path) -> None:
    if selection_values.empty:
        return
    plot = selection_values.dropna(subset=["actual_pick_number", "team_value"])
    if plot.empty:
        return
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        plot["actual_pick_number"],
        plot["team_value"],
        c=plot["pick_order"],
        cmap="viridis",
        s=36,
        alpha=0.78,
    )
    ax.set_xlabel("Actual draft pick number")
    ax.set_ylabel("TeamValueHead score")
    ax.set_title("Team Value vs Actual Draft Pick")
    fig.colorbar(scatter, ax=ax, label="Pick order")
    ax.grid(True, alpha=0.25)
    _save_plot(plt, fig, output_path)


def write_feature_phase_visuals(
    *,
    model,
    prepared: pd.DataFrame,
    history: pd.DataFrame,
    report,
    team_index_map: dict[str, int],
    team_event_index_map: dict[str, int],
    prior_checkpoint: str | Path | None,
    sidecar_tables: dict[str, pd.DataFrame] | None,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    attention_summary = team_attention_summary(report.set_attention)
    if not attention_summary.empty:
        attention_summary.to_csv(out / "feature_team_attention_summary.csv", index=False)
    prior_lookup = prior_lookup_table(prior_checkpoint)
    write_pma_carry_support_plot(
        attention_summary, prior_lookup, out / "feature_pma_carry_support.png"
    )
    sensitivity = getattr(report, "team_zero_out_sensitivity", pd.DataFrame())
    if not sensitivity.empty:
        sensitivity.to_csv(out / "feature_team_zero_out_sensitivity.csv", index=False)
    write_zero_out_war_plot(sensitivity, prior_lookup, out / "feature_zero_out_war.png")
    write_task_precision_plot(history, out / "feature_task_precision_evolution.png")
    drift = embedding_drift_table(
        model, team_index_map, team_event_index_map, prepared, prior_checkpoint
    )
    if not drift.empty:
        drift.to_csv(out / "feature_embedding_drift.csv", index=False)
        write_embedding_drift_plot(drift, out / "feature_embedding_drift_quiver.png")
    deltas = event_delta_summary(model, team_event_index_map, prepared)
    if not deltas.empty:
        deltas.to_csv(out / "feature_event_delta_summary.csv", index=False)
        write_event_delta_by_week_plot(deltas, out / "feature_event_delta_by_week.png")
    selections = selection_value_table(
        model, sidecar_tables.get("selections") if sidecar_tables else None
    )
    if not selections.empty:
        selections.to_csv(out / "feature_selection_value.csv", index=False)
        write_selection_value_plot(selections, out / "feature_team_value_vs_draft_pick.png")
