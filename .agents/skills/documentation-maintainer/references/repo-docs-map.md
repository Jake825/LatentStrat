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

## docs/current-state.md

Use for:

- The canonical current implementation snapshot.
- Promoted prior and walk-forward artifacts.
- Current caveats and local validation status.

Update this whenever model defaults, promoted artifacts, or current caveats
change.

## docs/changelog.md

Use for:

- Mapping semantic LatentStrat versions to Git commits.
- Recording unreleased or working-tree version states.
- Explaining how to inspect older code snapshots.

Update this whenever a named version, public CLI, model behavior, training
default, or promoted artifact recommendation changes.

## docs/experiment-ledger.md

Use for:

- Run-by-run local experiment evidence.
- Artifact paths, inputs, epochs/folds, outcomes, and design lessons.
- Comparing prior and walk-forward runs without crowding narrative docs.

## docs/cli-reference.md

Use for:

- Current `latentstrat` command surface.
- Command examples and recommended modern paths.
- Explaining underused diagnostic, evidence, venue, and cache commands.

## docs/schemas-and-artifacts-reference.md

Use for:

- Generated Parquet schemas.
- Sidecar schemas.
- Artifact CSV contracts and local artifact row counts.

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

## docs/model-architecture-reference.md

Use for:

- Exact tensor shapes.
- Layer dimensions and parameter counts.
- Loss equations and metric formulas.

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

## docs/student-primer.md

Use for:

- High-school-student-friendly explanations.
- Plain-language definitions of LatentStrat concepts.
- First-click path before technical references.

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

## Maintenance Rule

When model or training behavior changes, update the narrowest specific doc and
also check whether these source-of-truth docs need edits:

- `docs/current-state.md`
- `docs/changelog.md`
- `docs/model-architecture-reference.md`
- `docs/experiment-ledger.md`

If the exact commit SHA is not known yet, mark changelog rows as `Unreleased`
or `Pending commit` and replace them with a SHA after the work is committed.
