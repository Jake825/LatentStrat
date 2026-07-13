# CLI Reference

V6.1 groups commands by responsibility. Flat aliases remain available with V6.2 removal warnings.

## Pretraining

```text
latentstrat pretrain prior build|train|inspect|grid
latentstrat pretrain match-breakdown sync|train|inspect
```

Use prior commands for Day Zero `Z_base` initialization. Use match-breakdown commands for the
offline historical alliance-result bottleneck.

Training commands share `--gradient-accumulation-steps`, `--max-grad-norm`, `--scheduler`, `--lr-eta-min`, `--checkpoint-every-epochs`, deterministic-algorithm, TensorBoard, and resume controls where applicable. `cosine`, clipping at `1.0`, and CPU execution are the defaults.

## Season

```text
latentstrat season build-features
latentstrat season build-statbotics-baseline
latentstrat season train
latentstrat season validate
latentstrat season validate-reference-2026
latentstrat season run-static-reliability-study
latentstrat season run-static-championship-diagnostic
latentstrat season run-static-multitask-study
```

`season build-features` writes reusable Parquet. `season train` writes a strict schema-`6`
`checkpoint.pt`. `season validate` runs temporal walk-forward evaluation. `season
build-statbotics-baseline` writes pre-match Statbotics score and win-probability predictions; it
does not use post-match EPA or train a model.

`season validate-reference-2026` runs the schema-`7` static-`Z_base` study. It requires official
event metadata and a leakage-safe pre-2026 prior, defaults to a one-fold, at-most-two-epoch smoke
gate, writes isolated TensorBoard logs under `runs/reference-2026-static/`, and refuses a full
campaign whose timing estimate exceeds the CPU budget. Use `--full-study` only after reviewing the
[reference-study contract](reference-2026-static-study.md).

`season run-static-reliability-study` is the follow-up three-seed architecture experiment. It
freezes clipping and duration on development week 4, treats weeks 6 and 8 as primary evidence,
keeps week 10 as a stress test, and writes isolated TensorBoard logs under
`runs/reference-2026-static-reliability/`. It defaults to `--smoke`; inspect its help and pass
`--full-study` only after the smoke dashboards and runtime gate pass.

`season run-static-championship-diagnostic` runs the fixed-loss 2026 static Championship study.
It defaults to a two-epoch smoke, freezes exact match-key splits, masks DQs and winner ties, and
compares phase-derived with direct official-total supervision across additive, teammate-set, and
full-match architectures. Formal execution requires exactly 100 epochs, a clean Git commit, and a
passing 360-minute timing gate. See the [study contract](2026-static-championship-core-study.md).

`season run-static-multitask-study` runs the four-arm physics-consistent follow-up across Additive
and Full-match. It defaults to a one-seed, two-epoch smoke. `--prepare-only --full-study` writes the
hashed physics targets, exact split, corrected award candidates, and pre-training contract without
training. Formal execution requires `--full-study`, a clean commit, and a passing 720-minute timing
gate; `--resume` permits only exact contract-preserving continuation. See the
[multitask study contract](2026-static-multitask-physics-study.md).

The old `full-season-offline` alias is a tombstone. Use `season build-features` followed by
`season train`.

`season train --resume-checkpoint <path>` performs exact epoch-boundary continuation. `season validate` uses fold-local resume checkpoints under its output directory automatically.

## Artifacts

```text
latentstrat artifacts baseline-manifest
latentstrat artifacts compare-predictions
latentstrat artifacts evaluate-predictions
latentstrat artifacts finalize-static-reliability
```

Use `evaluate-predictions` for the supported metric-only comparison between saved LatentStrat and
Statbotics prediction Parquet files. It writes paired rows, score and probability metrics,
calibration, event-cluster bootstrap results, coverage, and a non-promotion manifest. The older
`compare-predictions` command remains the V6-Lite non-inferiority compatibility surface.

`finalize-static-reliability` attaches a verified saved Statbotics pre-match artifact to an
existing reliability study and recomputes only metrics, calibration, bootstrap evidence, coverage,
and manifests. It never loads or trains a PyTorch model.

## Scouting

```text
latentstrat scouting init
```

The default database is `data/scouting/scouting.db`.

## Developer Commands

```text
latentstrat dev api-smoke
latentstrat dev smoke-test
latentstrat dev clear-cache
latentstrat dev migrate-layout
latentstrat dev diagnostics embeddings
latentstrat dev diagnostics evidence
```

`dev migrate-layout` previews local-output moves by default. Pass `--apply` after review.

## Experimental

```text
latentstrat experimental frozen-targets build
latentstrat experimental venue train
latentstrat experimental venue consolidate
```

These are explicit research surfaces. Frozen award, ranking, and pick targets are not promoted
season-model inputs.

## Discover Options

```powershell
latentstrat --help
latentstrat pretrain match-breakdown train --help
latentstrat season train --help
```
