"""Day Zero prior pretraining workflow."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "build_prior_feature_table": "latentstrat.pretraining.prior.features",
    "inspect_prior_checkpoint": "latentstrat.pretraining.prior.inspection",
    "run_prior_grid": "latentstrat.pretraining.prior.grid",
    "train_prior_file": "latentstrat.pretraining.prior.train",
    "write_prior_feature_table": "latentstrat.pretraining.prior.features",
    "write_prior_inspection_artifacts": "latentstrat.pretraining.prior.inspection",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load workflow functions lazily to keep model imports acyclic."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)
