# LatentStrat Documentation Map

Use `docs/index.md` as the active-document index. Pages under `docs/archive/` describe superseded designs and commands.

## Entry Points

- `README.md`: purpose, installation, supported workflow, repository layout, storage, and links.
- `docs/index.md`: active documentation navigation.
- `docs/V6.1.md`: supported V6.1 contract and compatibility window.
- `docs/rebuild-from-scratch.md`: end-to-end runnable rebuild path.
- `docs/cli-reference.md`: grouped command surface and discovery guidance.

## Current Behavior

- `docs/architecture.md`: active boundaries, season model, and checkpoint flow.
- `docs/current-state.md`: recommended stack, canonical paths, evidence, and caveats.
- `docs/data-sources.md`: TBA, Statbotics, OpenAI, scouting, sidecars, secrets, and caches.
- `docs/season-training.md`: season feature, training, and validation workflow.
- `docs/prior-pretraining.md`: Day Zero prior inputs, commands, and artifact contract.
- `docs/match-breakdown-pretraining.md`: historical match-breakdown corpus, training, and inspection.

## Evaluation and Operations

- `docs/evaluation-and-artifacts.md`: metrics, Statbotics comparison, manifests, artifact paths, and interpretation limits.
- `docs/experiment-ledger.md`: local run evidence and open follow-ups.
- `docs/roadmap.md`: future milestones; never present these as implemented behavior.
- `docs/streamlit-app.md`: read-only research workbench behavior and validation.
- `docs/changelog.md`: version and commit timeline, including clearly marked unreleased work.

## Scouting and Teaching

- `docs/scouting-data-layer.md`: SQLModel schema, feature merge, and leakage discipline.
- `docs/scouting-data-ingestion.md`: normalization and importer examples.
- `docs/student-primer.md`: accessible explanations for students and new contributors.

## Maintenance Rule

Update the narrowest page that owns the behavior. Also inspect `docs/current-state.md`, `docs/changelog.md`, `docs/cli-reference.md`, and `docs/index.md` when a public command, supported workflow, artifact contract, or promoted recommendation changes.
