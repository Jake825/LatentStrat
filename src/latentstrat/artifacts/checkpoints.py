"""Season checkpoint detection and strict schema-aware loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from latentstrat.config import LatentStratOptions, default_options
from latentstrat.experimental.frozen_targets import WorldModelOptions
from latentstrat.season.model import SetTransformerModel, init_model


def load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def _init_from_state(
    state: dict[str, Any],
    opts: LatentStratOptions,
    *,
    world_model_opts: WorldModelOptions,
) -> SetTransformerModel:
    base_weight = state["Z_base.weight"]
    event_weight = state.get("Z_event.weight")
    return init_model(
        base_weight.shape[0],
        base_weight.shape[1],
        len(opts.target_map),
        len(opts.binary_targets),
        opts,
        num_event_teams=event_weight.shape[0] if event_weight is not None else 1,
        num_endgame_classes=len(opts.endgame_class_order),
        num_awards=len(opts.award_targets),
        world_model_opts=world_model_opts,
    )


def load_v5_checkpoint_model(checkpoint_path: str | Path) -> SetTransformerModel:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    opts = LatentStratOptions.model_validate(
        checkpoint.get("options", default_options().model_dump(mode="json"))
    )
    model = _init_from_state(
        checkpoint["model_state_dict"],
        opts,
        world_model_opts=WorldModelOptions(enabled=False),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model


def load_v6_checkpoint_model(checkpoint_path: str | Path) -> SetTransformerModel:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    if checkpoint.get("checkpoint_schema_version") not in {6, 7}:
        raise ValueError("Expected a schema-6 or schema-7 seasonal checkpoint.")
    opts = LatentStratOptions.model_validate(checkpoint["options"])
    world_model_opts = WorldModelOptions.model_validate(checkpoint["world_model"])
    model = _init_from_state(
        checkpoint["model_state_dict"],
        opts,
        world_model_opts=world_model_opts,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def load_season_checkpoint_model(checkpoint_path: str | Path) -> SetTransformerModel:
    checkpoint = load_checkpoint_payload(checkpoint_path)
    if checkpoint.get("checkpoint_schema_version") in {6, 7}:
        return load_v6_checkpoint_model(checkpoint_path)
    return load_v5_checkpoint_model(checkpoint_path)


def load_match_breakdown_checkpoint_model(
    checkpoint_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Strictly load a V1 or V2 historical match-breakdown checkpoint."""

    from latentstrat.pretraining.match_breakdown.model import MatchBreakdownAutoencoder

    checkpoint = load_checkpoint_payload(checkpoint_path)
    if checkpoint.get("kind") != "v6-lite-match-breakdown-shared-bottleneck":
        raise ValueError(
            f"Unsupported match-breakdown checkpoint kind: {checkpoint.get('kind')!r}."
        )
    widths = {int(season): int(width) for season, width in checkpoint["season_widths"].items()}
    schema_version = int(checkpoint.get("schema_version", 1))
    if schema_version not in {1, 2}:
        raise ValueError(
            f"Unsupported match-breakdown checkpoint schema version: {schema_version}."
        )
    model = MatchBreakdownAutoencoder(
        widths,
        latent_dim=int(checkpoint["latent_dim"]),
        quality_head_enabled=schema_version == 2,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
