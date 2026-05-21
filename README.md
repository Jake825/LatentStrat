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
`statbotics==3.0.0` for EPA/Statbotics data, Pandas/NumPy/SciPy/PyArrow for
tables and Parquet features, scikit-learn for linear baselines, and PyTorch for
the Set Transformer model.

## Secrets

Live TBA ingestion reads `TBA_API_KEY` from the environment. For local
development, create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
notepad .env
```

The CLI loads `.env` automatically. `.env` is ignored by Git.

## Commands

```bash
latentstrat api-smoke
latentstrat init-scouting-db --path data/scouting.db
latentstrat build-features --event-key 2026ilch --output data/features_2026ilch.parquet
latentstrat train-features data/features_2026ilch.parquet --output artifacts/features_run
latentstrat smoke-test
latentstrat full-season-offline --season 2026
latentstrat inspect-embeddings
latentstrat build-evidence-packet
```

The preferred ML loop is to run `build-features` after data changes, then run
`train-features` repeatedly from the local Parquet file. Python also keeps
SQLite/disk caches for raw provider calls.

Optional scouting data lives in `data/scouting.db`. `build-features` merges it
into Parquet when the database exists; training still reads only the Parquet
feature file.

The current codebase treats the Python implementation as the source of truth.
