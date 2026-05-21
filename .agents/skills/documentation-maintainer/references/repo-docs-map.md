# LatentStrat Documentation Map

Use this reference to place documentation in the right file.

## README.md

Use for:

- Project purpose.
- Installation.
- Secret setup.
- Common commands.
- High-level workflow.

Keep it short. Link deeper details to `docs/`.

## docs/training-and-validation.md

Use for:

- High-level feature-building workflow.
- Training from Parquet.
- Splits, target normalization, and validation boundaries.
- Baselines, controls, and evaluation context.

Coordinate with `$frc-time-aware-analysis` for leakage and timing language.

## docs/feature-pipeline.md

Use for:

- TBA match-table spine.
- Scouting merge prefixes and join grain.
- PyArrow Parquet boundaries.
- Team indexing and tensor conversion.
- Missing-data expectations for feature construction.

Coordinate with `$latentstrat-feature-pipeline`.

## docs/model-structure.md

Use for:

- Neural model architecture.
- Tensor shapes.
- Prediction heads.
- Model internals needed for maintainers.

## docs/model-selection.md

Use for:

- Selecting model variants.
- Comparing baselines and controls.
- Choosing evaluation criteria.

## docs/model-evaluation.md

Use for:

- Metric interpretation.
- Calibration, Brier score, and log loss.
- Current baselines and controls.
- Evidence packet outputs.
- Reading embedding diagnostics alongside metrics.

Coordinate with `$latentstrat-model-evaluation`.

## docs/embedding-inspection.md

Use for:

- Team embedding inspection.
- Diagnostics and interpretation.
- Evidence packet context.

## docs/scouting-data-layer.md

Use for:

- Scouting database purpose.
- SQLModel table overview.
- Feature merge behavior.
- Leakage discipline for scouting fields.

Coordinate with `$latentstrat-scouting-db`.

## docs/scouting-data-ingestion.md

Use for:

- Importer examples.
- CSV/spreadsheet ingestion.
- `session.merge` patterns.
- Scouting key normalization.

Coordinate with `$scouting-adapter-writer` and `$scouting-data-normalization`.
