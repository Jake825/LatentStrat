---
name: statbotics
description: Work directly with Statbotics, its Python package, LatentStrat's StatboticsProvider, EPA fields, or pre-match Statbotics baseline artifacts. Use for Statbotics retrieval, pagination, caching, field semantics, and saved baseline predictions. Do not use for TBA-only data, generic feature joins, model evaluation without Statbotics, or broad temporal audits.
---

# Statbotics

Use `statbotics==3.0.0` through `frc.providers.statbotics_provider.StatboticsProvider` when repository caching and stable provider behavior matter.

## Core Rules

- Treat EPA and predictions as model outputs, not ground truth.
- Distinguish pre-match `pred` fields from post-match `result` fields.
- Inspect returned keys before hard-coding nested paths.
- Prefer paginated bulk endpoints for historical workflows.
- Normalize shared event, match, and team identifiers before joins.
- For baseline artifacts, preserve exactly one row per match key and validate finite scores and probabilities in `[0, 1]`.

## References

- Read `references/data-access.md` before provider or API code.
- Read `references/epa-model.md` before interpreting EPA or prediction fields.
- Read `references/codebase-structure.md` only when working on the upstream Statbotics platform rather than LatentStrat.

Current LatentStrat baseline commands are `latentstrat season build-statbotics-baseline`, `latentstrat artifacts evaluate-predictions`, and `latentstrat artifacts compare-predictions`.
