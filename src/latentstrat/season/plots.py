"""Static plots written with season-model training artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _plot_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_empty(ax, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_plot(plt, fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_training_loss_plot(history: pd.DataFrame, output_path: Path) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    if history.empty or "epoch" not in history.columns:
        _plot_empty(ax, "Feature Training Loss")
    else:
        if "train_loss" in history.columns:
            ax.plot(history["epoch"], history["train_loss"], label="train", linewidth=1.8)
        if "validation_loss" in history.columns:
            ax.plot(history["epoch"], history["validation_loss"], label="validation", linewidth=1.8)
        ax.set_title("Feature Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.25)
    _save_plot(plt, fig, output_path)


def _write_task_loss_plot(history: pd.DataFrame, output_path: Path) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    excluded = {"train_loss", "validation_loss"}
    loss_columns = [
        column for column in history.columns if column.endswith("_loss") and column not in excluded
    ]
    if history.empty or "epoch" not in history.columns or not loss_columns:
        _plot_empty(ax, "Raw Task Losses")
    else:
        for column in loss_columns:
            values = pd.to_numeric(history[column], errors="coerce")
            if values.notna().any():
                ax.plot(history["epoch"], values, label=column.removesuffix("_loss"))
        ax.set_title("Raw Task Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, alpha=0.25)
    _save_plot(plt, fig, output_path)


def _write_calibration_plot(calibration: pd.DataFrame, output_path: Path) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(6, 6))
    if calibration.empty:
        _plot_empty(ax, "Win Calibration")
    else:
        ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
        for target, rows in calibration.groupby("target", sort=True):
            ordered = rows.sort_values("bin_low")
            centers = 0.5 * (
                ordered["bin_low"].to_numpy(dtype=float) + ordered["bin_high"].to_numpy(dtype=float)
            )
            mean_probability = pd.to_numeric(ordered["mean_probability"], errors="coerce").to_numpy(
                dtype=float
            )
            observed_rate = pd.to_numeric(ordered["observed_rate"], errors="coerce").to_numpy(
                dtype=float
            )
            count = pd.to_numeric(ordered["count"], errors="coerce").fillna(0)
            x = np.where(np.isfinite(mean_probability), mean_probability, centers)
            ax.scatter(x, observed_rate, s=25 + 3 * count.to_numpy(dtype=float), label=target)
            ax.plot(x, observed_rate, linewidth=1)
        ax.set_title("Win Calibration")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.25)
    _save_plot(plt, fig, output_path)


def _write_attention_entropy_plot(attention: pd.DataFrame, output_path: Path) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    if attention.empty or "attention_entropy" not in attention.columns:
        _plot_empty(ax, "PMA Attention Entropy")
    else:
        for name, rows in attention.groupby(["split", "alliance_color"], sort=True):
            values = pd.to_numeric(rows["attention_entropy"], errors="coerce").dropna()
            if not values.empty:
                ax.hist(values, bins=15, alpha=0.45, label=f"{name[0]} {name[1]}")
        ax.set_title("PMA Attention Entropy")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Alliance rows")
        if ax.has_data():
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    _save_plot(plt, fig, output_path)


def _write_zero_out_plot(zero_out: pd.DataFrame, output_path: Path) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(9, 5))
    if zero_out.empty or "delta_rmse" not in zero_out.columns:
        _plot_empty(ax, "Zero-Out Delta RMSE")
    else:
        summary = (
            zero_out.assign(delta_rmse=pd.to_numeric(zero_out["delta_rmse"], errors="coerce"))
            .dropna(subset=["delta_rmse"])
            .groupby("scenario", sort=True)["delta_rmse"]
            .mean()
        )
        if summary.empty:
            _plot_empty(ax, "Zero-Out Delta RMSE")
        else:
            summary.sort_values().plot.barh(ax=ax, color="#0f766e")
            ax.set_title("Zero-Out Delta RMSE")
            ax.set_xlabel("Mean validation delta RMSE")
            ax.set_ylabel("Scenario")
            ax.grid(True, axis="x", alpha=0.25)
    _save_plot(plt, fig, output_path)


def write_feature_artifact_plots(result: Any, out: Path) -> None:
    _write_training_loss_plot(result.history, out / "feature_training_loss.png")
    _write_task_loss_plot(result.history, out / "feature_task_losses.png")
    _write_calibration_plot(result.report.calibration, out / "feature_win_calibration.png")
    _write_attention_entropy_plot(
        result.report.set_attention, out / "feature_attention_entropy.png"
    )
    _write_zero_out_plot(
        result.report.zero_out_diagnostics, out / "feature_zero_out_delta_rmse.png"
    )
