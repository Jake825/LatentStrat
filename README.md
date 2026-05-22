# LatentStrat

LatentStrat is a Python-first FRC machine learning project for learning durable
team base embeddings and event-specific deltas from The Blue Alliance,
Statbotics, and scouting data. The default V5 model is a masked Set Transformer
that predicts match outcomes, endgame state, and post-event judged-award
auxiliary labels.

## Install

```bash
pip install -e ".[dev]"
```

The runtime stack is Python 3.11+, `tbapy` for The Blue Alliance API v3,
`statbotics==3.0.0` for EPA/Statbotics data, `openai>=1` for optional V5.5
text-prior pretraining, Pandas/NumPy/SciPy/PyArrow for tables and Parquet
features, scikit-learn for linear baselines, and PyTorch for the Set
Transformer model.

## Secrets

Live TBA ingestion reads `TBA_API_KEY` from the environment. V5.5 prior
pretraining also needs `OPENAI_API_KEY` when generating uncached text
embeddings. For local development, create `.env` from `.env.example`:

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
latentstrat build-prior-features --target-season 2026 --teams-from data/features_2026.parquet --output data/prior_features_2026.parquet
latentstrat train-prior --features data/prior_features_2026.parquet --output data/pretrained_prior_2026.pt
latentstrat inspect-prior --checkpoint data/pretrained_prior_2026.pt --features data/prior_features_2026.parquet --output artifacts/prior_2026
latentstrat train-features data/features_2026.parquet --prior-checkpoint data/pretrained_prior_2026.pt
latentstrat train-features data/features_2026ilch.parquet --output artifacts/features_run
latentstrat train-features data/features_2026ilch.parquet --venue-mode --event-key 2026ilch --checkpoint artifacts/features_run/v5_checkpoint.pt
latentstrat consolidate-event artifacts/features_run/v5_checkpoint.pt --event-key 2026ilch
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

## Docs

- [Feature pipeline](docs/feature-pipeline.md): TBA match spine, scouting joins,
  Parquet boundaries, V5 team/event indexing, awards, and tensor-ready data.
- [Training and validation](docs/training-and-validation.md): splits,
  normalization, baselines, controls, and training workflow.
- [Model structure](docs/model-structure.md): V5 latent trunk, masked Set
  Transformer architecture, task heads, and tensor contract.
- [Model evaluation](docs/model-evaluation.md): calibration, baselines,
  controls, evidence packets, and embeddings.
- [Embedding inspection](docs/embedding-inspection.md): PCA, cosine neighbors,
  archetypes, PMA attention, and zero-out diagnostics.
- [Scouting data layer](docs/scouting-data-layer.md): SQLite scouting schema and
  merge behavior.

The current codebase treats the Python implementation as the source of truth.
