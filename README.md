# LatentStrat

LatentStrat is a Python port of the original MATLAB LatentStrat V4 project and
its FRC/TBA support library. The port keeps the MATLAB model math as the parity
target while moving ingestion, training, evaluation, diagnostics, and evidence
packet generation into a Python package.

## Install

```bash
pip install -e ".[dev]"
```

The intended runtime stack is Python 3.11+, `tbapy` for The Blue Alliance API
v3, `statbotics==3.0.0` for EPA/Statbotics data, Pandas/NumPy/SciPy for tables
and baselines, and PyTorch for the V4 Set Transformer.

## Commands

```bash
latentstrat smoke-test
latentstrat full-season-offline --season 2026
latentstrat inspect-embeddings
latentstrat build-evidence-packet
```

Set `TBA_API_KEY` before running live TBA ingestion. Python builds fresh
SQLite/disk caches and does not migrate the legacy MATLAB `.mat` cache files.

The `LatentStrat MATLAB/` and `The Blue Alliance API Library MATLAB/` folders
remain as reference sources and golden-fixture generators during the migration.
