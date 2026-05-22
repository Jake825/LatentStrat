"""Offline training loop for V5.6 transductive prior distillation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from latentstrat.config import PriorOpts
from latentstrat.pretrain_features import read_prior_feature_table
from latentstrat.prior_model import TeamPriorDistiller, prior_distillation_losses


@dataclass
class PriorTrainingResult:
    model: TeamPriorDistiller
    history: pd.DataFrame
    embedding_table: Tensor
    checkpoint: dict[str, Any]
    output_path: Path | None = None
    tensorboard_logdir: Path | None = None


class PriorTensorDataset(Dataset):
    """Tensor-backed V5.6 distillation dataset."""

    def __init__(self, table: pd.DataFrame, opts: PriorOpts | None = None) -> None:
        opts = opts or PriorOpts()
        if table.empty:
            raise ValueError("Prior feature table is empty.")
        if "team_number" not in table.columns:
            raise KeyError("Prior feature table is missing team_number.")
        if "openai_narrative_vector" not in table.columns:
            raise KeyError("Prior feature table is missing openai_narrative_vector.")
        if "target_epa" not in table.columns:
            raise KeyError("Prior feature table is missing target_epa; rebuild prior features.")

        team_numbers = [int(value) for value in table["team_number"].tolist()]
        invalid = [value for value in team_numbers if value < 0 or value > opts.max_team_number]
        if invalid:
            sample = ", ".join(str(value) for value in sorted(set(invalid))[:5])
            raise ValueError(
                "Prior feature table contains team numbers outside "
                f"0..{opts.max_team_number}: {sample}."
            )

        vectors = [
            np.asarray(value, dtype=np.float32)
            for value in table["openai_narrative_vector"].tolist()
        ]
        bad_widths = sorted(
            {int(vector.shape[0]) for vector in vectors if vector.shape[0] != opts.llm_dim}
        )
        if bad_widths:
            raise ValueError(
                f"Prior vector width mismatch; expected {opts.llm_dim}, saw {bad_widths}."
            )

        self.team_numbers = torch.as_tensor(team_numbers, dtype=torch.long)
        self.team_keys = (
            [str(value) for value in table["team_key"].tolist()]
            if "team_key" in table.columns
            else [f"frc{value}" for value in team_numbers]
        )
        self.vectors = torch.as_tensor(np.stack(vectors, axis=0), dtype=torch.float32)
        target_epa = pd.to_numeric(table["target_epa"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(target_epa).all():
            raise ValueError("Prior feature table target_epa must be finite.")
        self.target_epa = torch.as_tensor(target_epa[:, None], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.team_numbers[index], self.vectors[index], self.target_epa[index]


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _target_season(table: pd.DataFrame) -> int | None:
    if "target_season" not in table.columns or table["target_season"].dropna().empty:
        return None
    seasons = sorted({int(value) for value in table["target_season"].dropna().tolist()})
    if len(seasons) > 1:
        raise ValueError(f"Prior feature table contains multiple target seasons: {seasons}.")
    return seasons[0]


def default_tensorboard_run_name(table: pd.DataFrame, *, timestamp: int | None = None) -> str:
    target_season = _target_season(table)
    season_text = str(target_season) if target_season is not None else "unknown"
    timestamp = int(time.time()) if timestamp is None else int(timestamp)
    return f"v561_prior_{season_text}_{timestamp}"


def create_tensorboard_writer(log_dir: str | Path) -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled, but tensorboard is not installed. "
            'Run `pip install -e ".[dev]"` or pass `--no-tensorboard`.'
        ) from exc
    return SummaryWriter(str(log_dir))


def _tensorboard_context(
    *,
    features_path: str | Path | None,
    output_path: str | Path | None,
    table: pd.DataFrame,
    opts: PriorOpts,
) -> str:
    metadata = {
        "features_path": str(features_path) if features_path is not None else "",
        "output_path": str(output_path) if output_path is not None else "",
        "target_season": _target_season(table),
        "latent_dim": opts.latent_dim,
        "max_team_number": opts.max_team_number,
        "llm_dim": opts.llm_dim,
        "epochs": opts.epochs,
        "batch_size": opts.batch_size,
        "learning_rate": opts.learning_rate,
        "epa_source_year": (
            int(table["epa_source_year"].dropna().iloc[0])
            if "epa_source_year" in table.columns and not table["epa_source_year"].dropna().empty
            else None
        ),
    }
    return "\n".join(f"- {key}: {value}" for key, value in metadata.items())


def _write_tensorboard_epoch(writer: Any, row: dict[str, float | int]) -> None:
    epoch = int(row["epoch"])
    log_var_openai = float(row["log_var_openai"])
    log_var_epa = float(row["log_var_epa"])
    writer.add_scalar("Loss/Total", float(row["train_loss"]), epoch)
    writer.add_scalar("Loss/OpenAI_MSE", float(row["openai_mse"]), epoch)
    writer.add_scalar("Loss/EPA_MSE", float(row["epa_mse"]), epoch)
    writer.add_scalar("LogVar/OpenAI", log_var_openai, epoch)
    writer.add_scalar("LogVar/EPA", log_var_epa, epoch)
    writer.add_scalar("Weights/OpenAI_Precision", math.exp(-log_var_openai), epoch)
    writer.add_scalar("Weights/EPA_Precision", math.exp(-log_var_epa), epoch)


def train_prior_model(
    table: pd.DataFrame,
    opts: PriorOpts | None = None,
    *,
    verbose: bool = True,
    tensorboard_writer: Any | None = None,
) -> tuple[TeamPriorDistiller, pd.DataFrame, PriorTensorDataset]:
    opts = opts or PriorOpts()
    torch.manual_seed(opts.random_seed)
    dataset = PriorTensorDataset(table, opts)
    device = _resolve_device(opts.device)
    model = TeamPriorDistiller(opts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.learning_rate)
    generator = torch.Generator()
    generator.manual_seed(opts.random_seed)
    loader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        generator=generator,
    )
    rows = []
    for epoch in range(1, opts.epochs + 1):
        model.train()
        total_loss = 0.0
        total_openai_mse = 0.0
        total_epa_mse = 0.0
        total_rows = 0
        for team_numbers, vectors, target_epa in loader:
            team_numbers = team_numbers.to(device)
            vectors = vectors.to(device)
            target_epa = target_epa.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, openai_mse, epa_mse = prior_distillation_losses(
                model, team_numbers, vectors, target_epa
            )
            loss.backward()
            optimizer.step()
            batch_rows = int(vectors.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_rows
            total_openai_mse += float(openai_mse.detach().cpu()) * batch_rows
            total_epa_mse += float(epa_mse.detach().cpu()) * batch_rows
            total_rows += batch_rows
        mean_loss = total_loss / max(total_rows, 1)
        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "openai_mse": total_openai_mse / max(total_rows, 1),
            "epa_mse": total_epa_mse / max(total_rows, 1),
            "log_var_openai": float(model.log_var_openai.detach().cpu()),
            "log_var_epa": float(model.log_var_epa.detach().cpu()),
        }
        rows.append(row)
        if tensorboard_writer is not None:
            _write_tensorboard_epoch(tensorboard_writer, row)
        if verbose and (epoch == 1 or epoch == opts.epochs or epoch % 100 == 0):
            print(f"Prior epoch {epoch}/{opts.epochs}: train loss {mean_loss:.6f}")
    return model, pd.DataFrame(rows), dataset


def build_prior_checkpoint(
    model: TeamPriorDistiller,
    history: pd.DataFrame,
    table: pd.DataFrame,
    opts: PriorOpts,
    *,
    source_feature_path: str | Path | None = None,
) -> dict[str, Any]:
    embedding_table = model.embedding_table().cpu()
    return {
        "embedding_table": embedding_table,
        "max_team_number": opts.max_team_number,
        "target_season": _target_season(table),
        "prior_opts": opts.model_dump(mode="json"),
        "source_feature_path": str(source_feature_path) if source_feature_path else None,
        "feature_metadata": {
            "rows": len(table),
            "team_count": max(int(embedding_table.shape[0]) - 1, 0),
            "has_ghost_token": True,
            "has_epa_target": "target_epa" in table.columns,
            "epa_source_year": (
                int(table["epa_source_year"].dropna().iloc[0])
                if "epa_source_year" in table.columns
                and not table["epa_source_year"].dropna().empty
                else None
            ),
            "epa_mean": (
                float(table["epa_mean"].dropna().iloc[0])
                if "epa_mean" in table.columns and not table["epa_mean"].dropna().empty
                else None
            ),
            "epa_std": (
                float(table["epa_std"].dropna().iloc[0])
                if "epa_std" in table.columns and not table["epa_std"].dropna().empty
                else None
            ),
            "max_team_number": opts.max_team_number,
            "embedding_model": opts.embedding_model,
            "llm_dim": opts.llm_dim,
            "latent_dim": opts.latent_dim,
            "history_rows": len(history),
        },
        "history": history.to_dict(orient="records"),
    }


def resolve_prior_checkpoint_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.suffix.lower() in {".pt", ".pth"}:
        return output
    return output / "checkpoint.pt"


def train_prior_file(
    features_path: str | Path,
    output_path: str | Path,
    opts: PriorOpts | None = None,
    *,
    verbose: bool = True,
    tensorboard_logdir: str | Path | None = None,
    tensorboard_run_name: str | None = None,
) -> PriorTrainingResult:
    opts = opts or PriorOpts()
    table = read_prior_feature_table(features_path)
    output = resolve_prior_checkpoint_output(output_path)
    writer = None
    run_dir = None
    if tensorboard_logdir is not None:
        run_name = tensorboard_run_name or default_tensorboard_run_name(table)
        run_dir = Path(tensorboard_logdir) / run_name
        writer = create_tensorboard_writer(run_dir)
        writer.add_text(
            "Run/Context",
            _tensorboard_context(
                features_path=features_path,
                output_path=output,
                table=table,
                opts=opts,
            ),
            0,
        )
    try:
        model, history, _ = train_prior_model(
            table,
            opts,
            verbose=verbose,
            tensorboard_writer=writer,
        )
        checkpoint = build_prior_checkpoint(
            model,
            history,
            table,
            opts,
            source_feature_path=features_path,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output)
        return PriorTrainingResult(
            model=model,
            history=history,
            embedding_table=checkpoint["embedding_table"],
            checkpoint=checkpoint,
            output_path=output,
            tensorboard_logdir=run_dir,
        )
    finally:
        if writer is not None:
            writer.close()


def load_prior_embedding_table(prior_checkpoint: str | Path) -> Tensor:
    checkpoint = torch.load(prior_checkpoint, map_location="cpu", weights_only=False)
    table = checkpoint.get("embedding_table")
    if table is None:
        raise KeyError("Prior checkpoint is missing embedding_table.")
    table = torch.as_tensor(table, dtype=torch.float32).detach().cpu()
    if table.ndim != 2:
        raise ValueError("Prior embedding_table must be a 2D tensor.")
    if table.shape[0] < 2:
        raise ValueError("Prior embedding_table must include null row 0 and at least one team.")
    return table


def apply_prior_checkpoint_to_model(
    model: Any,
    prior_checkpoint: str | Path,
    *,
    latent_dim: int,
) -> int:
    embedding_table = load_prior_embedding_table(prior_checkpoint)
    if int(embedding_table.shape[1]) != int(latent_dim):
        raise ValueError(
            f"Prior embedding width {embedding_table.shape[1]} does not match {latent_dim}."
        )
    if int(model.Z_base.weight.shape[1]) != int(latent_dim):
        raise ValueError(
            f"Model Z_base width {model.Z_base.weight.shape[1]} does not match {latent_dim}."
        )
    if int(model.Z_base.weight.shape[0]) < int(embedding_table.shape[0]):
        raise ValueError(
            "Model Z_base is smaller than the V5.6 prior dictionary; initialize the model "
            f"with at least {embedding_table.shape[0]} base rows."
        )
    with torch.no_grad():
        model.Z_base.weight[: embedding_table.shape[0]].copy_(
            embedding_table.to(
                dtype=model.Z_base.weight.dtype,
                device=model.Z_base.weight.device,
            )
        )
    return max(int(embedding_table.shape[0]) - 1, 0)
