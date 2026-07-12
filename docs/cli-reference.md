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
```

`season build-features` writes reusable Parquet. `season train` writes a strict schema-`6`
`checkpoint.pt`. `season validate` runs temporal walk-forward evaluation. `season
build-statbotics-baseline` writes pre-match Statbotics score and win-probability predictions; it
does not use post-match EPA or train a model.

The old `full-season-offline` alias is a tombstone. Use `season build-features` followed by
`season train`.

`season train --resume-checkpoint <path>` performs exact epoch-boundary continuation. `season validate` uses fold-local resume checkpoints under its output directory automatically.

## Artifacts

```text
latentstrat artifacts baseline-manifest
latentstrat artifacts compare-predictions
latentstrat artifacts evaluate-predictions
```

Use `evaluate-predictions` for the supported metric-only comparison between saved LatentStrat and
Statbotics prediction Parquet files. It writes paired rows, score and probability metrics,
calibration, event-cluster bootstrap results, coverage, and a non-promotion manifest. The older
`compare-predictions` command remains the V6-Lite non-inferiority compatibility surface.

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
