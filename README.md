# LatentStrat

LatentStrat is a Python-first FRC machine learning project for learning team and
alliance representations from The Blue Alliance and Statbotics data. It uses a
Set Transformer-style PyTorch model to encode unordered alliances, model
cross-alliance interactions, and produce match predictions plus diagnostics.

## Install

```bash
pip install -e ".[dev]"
```

The runtime stack is Python 3.11+, `tbapy` for The Blue Alliance API v3,
`statbotics==3.0.0` for EPA/Statbotics data, Pandas/NumPy/SciPy for tables,
scikit-learn for linear baselines, and PyTorch for the Set Transformer model.

## Commands

```bash
latentstrat smoke-test
latentstrat full-season-offline --season 2026
latentstrat inspect-embeddings
latentstrat build-evidence-packet
```

Set `TBA_API_KEY` before running live TBA ingestion. Python builds fresh
SQLite/disk caches for provider calls.

The current codebase treats the Python implementation as the source of truth.
