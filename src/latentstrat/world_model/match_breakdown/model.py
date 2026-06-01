"""Shared-bottleneck autoencoder with season-specific front and back ends."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SeasonEncoder(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feature_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.net(torch.cat([values, mask], dim=-1))


class SharedBottleneck(nn.Module):
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SeasonDecoder(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, feature_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent)


class MatchBreakdownAutoencoder(nn.Module):
    """Route each season through its own schema while sharing the exported latent."""

    def __init__(self, season_widths: dict[int, int], latent_dim: int = 16) -> None:
        super().__init__()
        if not season_widths:
            raise ValueError("At least one season schema is required.")
        self.season_widths = dict(sorted(season_widths.items()))
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleDict(
            {str(season): SeasonEncoder(width) for season, width in self.season_widths.items()}
        )
        self.bottleneck = SharedBottleneck(latent_dim)
        self.decoders = nn.ModuleDict(
            {
                str(season): SeasonDecoder(width, latent_dim)
                for season, width in self.season_widths.items()
            }
        )

    def forward(self, season: int, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        key = str(int(season))
        if key not in self.encoders:
            raise ValueError(f"No match-breakdown schema is registered for season {season}.")
        expected = self.season_widths[int(season)]
        if values.shape[-1] != expected or mask.shape[-1] != expected:
            raise ValueError(
                f"Season {season} expects width {expected}, got values={values.shape[-1]} "
                f"mask={mask.shape[-1]}."
            )
        latent = self.bottleneck(self.encoders[key](values, mask))
        return self.decoders[key](latent), latent
