# LatentStrat

LatentStrat is a Python-first FRC machine-learning toolkit. It learns durable team identities,
event-specific adjustments, and supervised match predictions from The Blue Alliance, Statbotics,
and optional scouting data. V6.1 also provides offline pretraining workflows for Day Zero priors
and historical match-breakdown embeddings.

## Install

```powershell
pip install -e ".[dev]"
```

Create `.env` from the example file:

```powershell
Copy-Item .env.example .env
notepad .env
```

Live TBA ingestion requires `TBA_API_KEY`. Prior pretraining also requires `OPENAI_API_KEY` when
the text embedding cache does not already contain the requested vectors.

## Supported Workflow

```powershell
latentstrat dev api-smoke
latentstrat scouting init

latentstrat pretrain prior build --target-season 2026
latentstrat pretrain prior train `
  --features data/features/pretraining/prior/prior_features_2026.parquet

latentstrat season build-features --season 2026
latentstrat season train data/features/season/features_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_run/checkpoint.pt

latentstrat season validate `
  --features data/features/season/features_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_run/checkpoint.pt
```

Historical match-breakdown pretraining is a separate offline workflow:

```powershell
latentstrat pretrain match-breakdown sync --start-season 2015 --end-season 2026
latentstrat pretrain match-breakdown train --start-season 2015 --end-season 2026
latentstrat pretrain match-breakdown inspect `
  --artifact-dir artifacts/pretraining/match-breakdown/v1_2015_2026
```

Flat command aliases remain available for one release and print V6.2 removal warnings.

## Repository Layout

```text
.agents/        Local Codex skills and workflow guidance for this repo.
baselines/      Checked-in baseline metadata used by tests and comparisons.
configs/        YAML configuration for season training and experimental workflows.
data/           Small checked-in catalogs plus ignored local caches, databases, and features.
docs/           Active project documentation, with historical notes under docs/archive/.
scripts/        Maintenance scripts for local storage organization.
src/frc/        FRC data models, providers, scouting helpers, and importers.
src/latentstrat/ Core package: CLI, features, training, pretraining, artifacts, and evaluation.
tests/          Pytest coverage for data handling, features, training, models, and workflows.
```

Generated experiment outputs live outside the tracked source tree in ignored folders such as
`artifacts/`, `runs/`, `data/features/`, `data/pretraining/`, `data/scouting/`, and
`statbotics_offline_cache/`.

## Storage

Generated local files are ignored by Git:

```text
data/cache/
data/pretraining/
data/features/
data/scouting/
data/sidecars/
artifacts/pretraining/
artifacts/season/
artifacts/walk-forward/
artifacts/experimental/
artifacts/archive/
runs/
```

Preview legacy local-output migration with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/organize_local_outputs.ps1
```

Apply only after reviewing the preview:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/organize_local_outputs.ps1 -Apply
```

## Documentation

- [V6.1 overview](docs/V6.1.md)
- [Architecture](docs/architecture.md)
- [CLI reference](docs/cli-reference.md)
- [Rebuild from scratch](docs/rebuild-from-scratch.md)
- [Evaluation and artifacts](docs/evaluation-and-artifacts.md)
- [Documentation hub](docs/index.md)
