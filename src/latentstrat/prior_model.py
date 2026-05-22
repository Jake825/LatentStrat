"""Denoising autoencoder for V5.5 text-only priors."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from latentstrat.config import PriorOpts


class UnifiedPriorAutoencoder(nn.Module):
    """Compress 256-D OpenAI text priors into the V5 base-embedding bottleneck."""

    def __init__(self, opts: PriorOpts | None = None) -> None:
        super().__init__()
        opts = opts or PriorOpts()
        self.opts = opts
        self.corruption = nn.Dropout(p=opts.dropout_rate)
        self.encoder = nn.Sequential(
            nn.Linear(opts.llm_dim, 128),
            nn.GELU(),
            nn.Linear(128, opts.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(opts.latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, opts.llm_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        corrupted = self.corruption(x)
        return self.decoder(self.encoder(corrupted))


def reconstruction_loss(model: UnifiedPriorAutoencoder, x: Tensor) -> Tensor:
    return F.mse_loss(model(x), x)
