"""Offline training loop for V5.5 text-prior pretraining."""

from __future__ import annotations

from collections.abc import Mapping
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
from latentstrat.prior_model import UnifiedPriorAutoencoder, reconstruction_loss


@dataclass
class PriorTrainingResult:
    model: UnifiedPriorAutoencoder
    history: pd.DataFrame
    team_vectors: dict[str, Tensor]
    checkpoint: dict[str, Any]


class PriorTensorDataset(Dataset):
    def __init__(self, table: pd.DataFrame, opts: PriorOpts | None = None) -> None:
        opts = opts or PriorOpts()
        if table.empty:
            raise ValueError("Prior feature table is empty.")
        if "team_key" not in table.columns:
            raise KeyError("Prior feature table is missing team_key.")
        if "openai_narrative_vector" not in table.columns:
            raise KeyError("Prior feature table is missing openai_narrative_vector.")
        self.team_keys = [str(value) for value in table["team_key"].tolist()]
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
        self.vectors = torch.as_tensor(np.stack(vectors, axis=0), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        return self.team_keys[index], self.vectors[index]


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


def train_prior_model(
    table: pd.DataFrame,
    opts: PriorOpts | None = None,
    *,
    verbose: bool = True,
) -> tuple[UnifiedPriorAutoencoder, pd.DataFrame, PriorTensorDataset]:
    opts = opts or PriorOpts()
    torch.manual_seed(opts.random_seed)
    dataset = PriorTensorDataset(table, opts)
    device = _resolve_device(opts.device)
    model = UnifiedPriorAutoencoder(opts).to(device)
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
        total_rows = 0
        for _, vectors in loader:
            vectors = vectors.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model, vectors)
            loss.backward()
            optimizer.step()
            batch_rows = int(vectors.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_rows
            total_rows += batch_rows
        mean_loss = total_loss / max(total_rows, 1)
        rows.append({"epoch": epoch, "train_loss": mean_loss})
        if verbose and (epoch == 1 or epoch == opts.epochs or epoch % 100 == 0):
            print(f"Prior epoch {epoch}/{opts.epochs}: train loss {mean_loss:.6f}")
    return model, pd.DataFrame(rows), dataset


def encode_prior_vectors(
    model: UnifiedPriorAutoencoder,
    dataset: PriorTensorDataset,
) -> dict[str, Tensor]:
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        encoded = model.encode(dataset.vectors.to(device)).detach().cpu()
    return {
        team_key: encoded[idx].clone()
        for idx, team_key in enumerate(dataset.team_keys)
    }


def build_prior_checkpoint(
    model: UnifiedPriorAutoencoder,
    history: pd.DataFrame,
    dataset: PriorTensorDataset,
    table: pd.DataFrame,
    opts: PriorOpts,
    *,
    source_feature_path: str | Path | None = None,
) -> dict[str, Any]:
    team_vectors = encode_prior_vectors(model, dataset)
    return {
        "team_vectors": team_vectors,
        "target_season": _target_season(table),
        "prior_opts": opts.model_dump(mode="json"),
        "model_state_dict": model.state_dict(),
        "source_feature_path": str(source_feature_path) if source_feature_path else None,
        "feature_metadata": {
            "rows": len(table),
            "team_count": len(team_vectors),
            "embedding_model": opts.embedding_model,
            "llm_dim": opts.llm_dim,
            "latent_dim": opts.latent_dim,
            "history_rows": len(history),
        },
        "history": history.to_dict(orient="records"),
    }


def train_prior_file(
    features_path: str | Path,
    output_path: str | Path,
    opts: PriorOpts | None = None,
    *,
    verbose: bool = True,
) -> PriorTrainingResult:
    opts = opts or PriorOpts()
    table = read_prior_feature_table(features_path)
    model, history, dataset = train_prior_model(table, opts, verbose=verbose)
    checkpoint = build_prior_checkpoint(
        model,
        history,
        dataset,
        table,
        opts,
        source_feature_path=features_path,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return PriorTrainingResult(
        model=model,
        history=history,
        team_vectors=checkpoint["team_vectors"],
        checkpoint=checkpoint,
    )


def load_prior_vectors(prior_checkpoint: str | Path) -> dict[str, Tensor]:
    checkpoint = torch.load(prior_checkpoint, map_location="cpu", weights_only=False)
    vectors = checkpoint.get("team_vectors")
    if not isinstance(vectors, dict):
        raise KeyError("Prior checkpoint is missing team_vectors.")
    return {
        str(team_key): torch.as_tensor(vector, dtype=torch.float32).detach().cpu()
        for team_key, vector in vectors.items()
    }


def apply_prior_checkpoint_to_model(
    model: Any,
    prior_checkpoint: str | Path,
    team_base_index_map: Mapping[str, int],
    *,
    latent_dim: int,
) -> int:
    vectors = load_prior_vectors(prior_checkpoint)
    applied = 0
    with torch.no_grad():
        for team_key, row_idx in team_base_index_map.items():
            if int(row_idx) <= 0:
                continue
            vector = vectors.get(str(team_key))
            if vector is None:
                continue
            if int(vector.numel()) != latent_dim:
                raise ValueError(
                    f"Prior vector for {team_key} has width {vector.numel()}, "
                    f"expected {latent_dim}."
                )
            if int(row_idx) >= int(model.Z_base.weight.shape[0]):
                raise IndexError(f"team_base_idx {row_idx} is outside Z_base.")
            model.Z_base.weight[int(row_idx)].copy_(
                vector.to(dtype=model.Z_base.weight.dtype, device=model.Z_base.weight.device)
            )
            applied += 1
    return applied
