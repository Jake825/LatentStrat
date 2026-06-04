# Rebuild From Scratch

Run these commands from the repository root.

## Install

```powershell
pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env
latentstrat dev api-smoke
latentstrat scouting init
```

Set `TBA_API_KEY`. Set `OPENAI_API_KEY` if prior text embeddings are not already cached.

## Prior

```powershell
latentstrat pretrain prior build --target-season 2026
latentstrat pretrain prior train `
  --features data/features/pretraining/prior/prior_features_2026.parquet `
  --output artifacts/pretraining/prior/prior_run
```

## Season

```powershell
latentstrat season build-features --season 2026
latentstrat season train data/features/season/features_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_run/checkpoint.pt `
  --output artifacts/season/features_run
latentstrat season validate `
  --features data/features/season/features_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_run/checkpoint.pt `
  --output artifacts/walk-forward/v58_walk_forward_2026
```

## Historical Match Breakdowns

```powershell
latentstrat pretrain match-breakdown sync --start-season 2015 --end-season 2026
latentstrat pretrain match-breakdown train --start-season 2015 --end-season 2026
latentstrat pretrain match-breakdown inspect `
  --artifact-dir artifacts/pretraining/match-breakdown/v1_2015_2026
```

The all-years artifact is for offline representation inspection. It is not leakage-safe
walk-forward evidence.

## Validate The Checkout

```powershell
python -m pytest -q
python -m ruff check .
git diff --check
latentstrat --help
```
