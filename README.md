# LatentStrat

LatentStrat is a Python-first FRC machine learning project for learning durable
team base embeddings and event-specific deltas from The Blue Alliance,
Statbotics, and scouting data. The default V5 model is a cross-alliance Set
Transformer with a learned ghost robot for missing slots, predicting match
outcomes, endgame state, and post-event judged-award auxiliary labels.

## Install

```bash
pip install -e ".[dev]"
```

The runtime stack is Python 3.11+, `tbapy` for The Blue Alliance API v3,
`statbotics==3.0.0` for EPA/Statbotics data, `openai>=1` for optional V5.6.4
text-plus-normalized-EPA prior distillation, Pandas/NumPy/SciPy/PyArrow for tables and Parquet
features, scikit-learn for linear baselines, and PyTorch for the Set
Transformer model.

## Secrets

Live TBA ingestion reads `TBA_API_KEY` from the environment. V5.6.4 prior
distillation also needs `OPENAI_API_KEY` when generating uncached text
embeddings. For local development, create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
notepad .env
```

The CLI loads `.env` automatically. `.env` is ignored by Git.

## Commands

For a complete command-by-command reference, see
[CLI reference](docs/cli-reference.md).

```bash
latentstrat api-smoke
latentstrat init-scouting-db --path data/scouting/scouting.db
latentstrat build-features --event-key 2026ilch --output data/features/event/features_2026ilch.parquet
latentstrat build-features --season 2026 --output data/features/season/features_2026.parquet --sidecar-output-dir data/sidecars/v58_2026
latentstrat build-prior-features --target-season 2026 --output data/features/prior/prior_features_2026.parquet
latentstrat train-prior --features data/features/prior/prior_features_2026.parquet --output artifacts/prior/prior_v564_latent16 --epochs 1000 --latent-dim 16 --tensorboard
tensorboard --logdir=runs
latentstrat inspect-prior --checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt --output artifacts/prior/prior_v564_latent16/inspection
latentstrat run-prior-grid --features data/features/prior/prior_features_2026.parquet --output artifacts/prior-grid/prior_elbow_grid --epochs 500 --latent-dims 2,4,8,16,32,64,128,256
latentstrat train-features data/features/season/features_2026.parquet --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt
latentstrat train-features data/features/season/features_2026.parquet --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet
latentstrat train-features data/features/season/features_2026.parquet --epochs 100 --no-early-stopping --restore-best
tensorboard --logdir=runs
latentstrat validate-walk-forward --features data/features/season/features_2026.parquet --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet --output artifacts/walk-forward/v58_walk_forward_2026 --epochs 5 --latent-dim 16
latentstrat train-features data/features/event/features_2026ilch.parquet --output artifacts/season/features_run
latentstrat train-features data/features/event/features_2026ilch.parquet --venue-mode --event-key 2026ilch --checkpoint artifacts/season/features_run/v5_checkpoint.pt
latentstrat consolidate-event artifacts/season/features_run/v5_checkpoint.pt --event-key 2026ilch
latentstrat smoke-test
latentstrat full-season-offline --season 2026
latentstrat inspect-embeddings
latentstrat build-evidence-packet
```

The preferred ML loop is to run `build-features` after data changes, then run
`train-features` repeatedly from the local Parquet file. Provider and embedding
caches use local SQLite files under `data/cache/`.

All LatentStrat training CLI entrypoints should support local TensorBoard logs
under `runs/` by default, with `--no-tensorboard` available for quiet batch or
test runs. Current prior and feature training commands follow this convention.
The `runs/` directory is ignored by Git. See
[Training and validation](docs/training-and-validation.md) for the logging
policy expected of future training scripts.

Optional scouting data lives in `data/scouting/scouting.db`. `build-features` merges it
into Parquet when the database exists; training still reads only the Parquet
feature file.

## Local Storage Layout

Generated local files are organized by purpose. Provider caches live under `data/cache/`: TBA uses `data/cache/tba.sqlite`, Statbotics uses `data/cache/statbotics.sqlite`, and OpenAI embeddings use `data/cache/openai_embeddings.sqlite`. Scouting uses `data/scouting/scouting.db`, durable consolidated embeddings use `data/embeddings/latentstrat_embeddings.sqlite`, Parquet feature tables live under `data/features/`, sidecars live under `data/sidecars/`, model artifacts live under grouped `artifacts/` subdirectories, and TensorBoard runs stay under `runs/`. Legacy explicit paths still work, but new commands and docs should prefer the canonical layout.

Use `scripts/organize_local_outputs.ps1` to preview local generated-file moves into this layout. The default mode is a dry run; pass `-Apply` only after reviewing the planned moves.

V5.7 match features include generic score-breakdown targets for atomic scoring
counts, committed fouls, bonus ranking-point thresholds, and special penalties.
Optional rankings, alliance selections, and playoff sidecars can be written with
`--sidecar-output-dir` and supplied to `train-features` as auxiliary
learning-to-rank labels. V5.7.2 clamps homoscedastic log variances, uses cosine
learning-rate decay, and restores the best validation epoch even for long
`--no-early-stopping` monitoring runs. V5.8 adds walk-forward validation over
canonical season weeks, with TBA Week 0 bundled into model-facing Week 1.

## Docs

- [Documentation hub](docs/index.md): guided entrypoint for students and
  developers.
- [Student primer](docs/student-primer.md): plain-language explanation of
  LatentStrat for FRC students.
- [Current state](docs/current-state.md): canonical snapshot of the current
  implementation, artifacts, caveats, and validation status.
- [Changelog](docs/changelog.md): semantic version timeline tied to Git commits
  and local artifact eras.
- [Rebuild from scratch](docs/rebuild-from-scratch.md): end-to-end setup,
  feature building, prior training, season training, and walk-forward commands.
- [CLI reference](docs/cli-reference.md): current command surface and examples.
- [Prior training](docs/prior-training.md): V5.6.4 Day Zero prior,
  narratives, EPA trajectory, culture targets, and checkpoint handoff.
- [Season training](docs/season-training.md): V5.7/V5.8 training, sidecars,
  stability, TensorBoard, and walk-forward validation.
- [Data sources](docs/data-sources.md): TBA, Statbotics, OpenAI, scouting
  SQLite, sidecars, and timing boundaries.
- [Metrics and artifacts](docs/metrics-and-artifacts.md): Brier score, log
  loss, score MSE, TensorBoard, and artifact interpretation.
- [Schemas and artifacts reference](docs/schemas-and-artifacts-reference.md):
  generated Parquet and CSV contracts.
- [Project history](docs/project-history.md): experiment ledger and design
  decisions through V5.8.
- [Experiment ledger](docs/experiment-ledger.md): run-by-run local artifact
  evidence and design lessons.
- [Feature pipeline](docs/feature-pipeline.md): TBA match spine, scouting joins,
  Parquet boundaries, V5 team/event indexing, awards, and tensor-ready data.
- [Training and validation](docs/training-and-validation.md): splits,
  normalization, baselines, controls, and training workflow.
- [Model structure](docs/model-structure.md): V5 latent trunk, Set Transformer
  architecture, task heads, ghost-token behavior, and tensor contract.
- [Model evaluation](docs/model-evaluation.md): calibration, baselines,
  controls, evidence packets, and embeddings.
- [Embedding inspection](docs/embedding-inspection.md): PCA, cosine neighbors,
  archetypes, PMA attention, and zero-out diagnostics.
- [Scouting data layer](docs/scouting-data-layer.md): SQLite scouting schema and
  merge behavior.

The current codebase treats the Python implementation as the source of truth.
