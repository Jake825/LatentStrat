# Evaluation And Artifacts

LatentStrat separates generated evidence from tracked source files. Generated outputs are ignored
locally, while tracked manifests preserve promoted baseline hashes where required.

## Metrics

Use walk-forward paired predictions for season-model promotion:

- Total score MSE and phase score MSE.
- Winner accuracy.
- Brier score.
- Log loss.
- Fold-level paired bootstrap comparisons.

Calibration matters independently from winner accuracy. A model can rank winners more accurately
while producing worse probabilities.

## Manifest Contract

New artifact directories write `manifest.json` containing:

- Workflow and artifact version.
- UTC generation timestamp and Git SHA.
- Resolved configuration.
- Source hashes.
- Output hashes and byte sizes.
- Promotion eligibility and migration provenance.

The manifest excludes itself from its output hash table.

## Paths

```text
artifacts/pretraining/prior/
artifacts/pretraining/prior-grid/
artifacts/pretraining/match-breakdown/
artifacts/season/
artifacts/walk-forward/
artifacts/experimental/frozen-targets/
artifacts/archive/
```

## Interpretation

Embedding PCA, t-SNE, parallel-coordinate, and cross-season neighbor plots are diagnostics. They can
reveal gradients, clustering, and suspicious calendar-year separation. They are not standalone
promotion evidence.

The V5.8 tagged replay remains the season-model comparison boundary:

```text
baselines/v5.8-baseline.json
artifacts/baselines/v5.8/
```
