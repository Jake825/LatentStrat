---
tags:
  - latentstrat
  - changelog
  - project-history
aliases:
  - "LatentStrat Changelog"
  - "Version Timeline"
related:
  - "[[current-state]]"
  - "[[experiment-ledger]]"
  - "[[project-history]]"
---

# Changelog

This changelog connects LatentStrat's research version names, Git commits, and local artifact eras. It is meant to help an FRC student understand the project timeline and help an agent find the closest code snapshot for an older design.

## How To Use This Changelog

- **Semantic versions** such as V5.6.4 or V5.8 are research milestones. They describe modeling ideas and experiment phases, not packaged software releases.
- **Git commits** are code-state anchors. If a row has a commit hash, an agent can inspect or check out that exact code state.
- **Artifacts** are local evidence for experiment outcomes. They may not exist in a fresh clone unless they are rebuilt.
- **Unreleased** or **Working tree** rows describe the current local state after the latest commit. They cannot be checked out by hash until committed.

`v5.8-baseline` is the annotated historical boundary before V6-Lite. Tags mark source history;
tracked baseline manifests and ignored local archives preserve empirical comparison inputs and
outputs.

To inspect a committed code state without disturbing the working tree:

```bash
git show --stat <commit>
git worktree add ../latentstrat-<commit> <commit>
```

Use a detached checkout only when you intentionally want to inspect history:

```bash
git switch --detach <commit>
```

## Current Unreleased State

| Field | Value |
|---|---|
| Status | `Unreleased / V6-Lite implementation after v5.8-baseline` |
| Current stack | V5.6.4 prior, tagged V5.8 comparison baseline, V6-Lite offline historical score artifact |
| Current recommended prior | `artifacts/prior_v564_latent16/checkpoint.pt` local historical path; new default equivalent is `artifacts/prior/prior_v564_latent16/checkpoint.pt` |
| Current recommended walk-forward comparison | `artifacts/baselines/v5.8/walk-forward/` with tracked manifest `baselines/v5.8-baseline.json` |
| Baseline Git boundary | annotated tag `v5.8-baseline` at `b8e9a6c` |
| State note | V6-Lite V1 builds an offline historical score encoder; runtime score attachment and walk-forward promotion remain deferred |

Read [Current State](current-state.md) for the current recommended artifacts, caveats, and validation status.

## Commit Timeline

| Version / milestone | Date | Git commit | Commit subject | What changed | Main docs/artifacts to inspect | How to inspect code |
|---|---|---|---|---|---|---|
| Initial repository | 2026-05-20 | `5d0d669` | Initial commit | Started the repository with a README. | `README.md` at commit | `git show --stat 5d0d669` |
| Python port | 2026-05-20 | `8d9f0c1` | Port LatentStrat core to Python | Added the first Python implementation, providers, baseline modeling, Set Transformer modules, evaluation, experiments, and tests. | Python source under `src/`, initial tests | `git show --stat 8d9f0c1` |
| Native PyTorch refactor | 2026-05-20 | `0803648` | Refactor LatentStrat for native PyTorch | Refined the PyTorch training/model path and removed older MATLAB fixture workflow. | `src/latentstrat/model.py`, `training.py`, `evaluation.py` | `git show --stat 0803648` |
| Batched evaluation and acceleration | 2026-05-20 | `b2bd344` | Add batched evaluation and PyTorch acceleration options | Added batched evaluation and performance-oriented training/evaluation options. | evaluation and training tests | `git show --stat b2bd344` |
| Secrets and Parquet workflow | 2026-05-20 | `050a3fa` | Add dotenv secrets and Parquet feature workflow | Added `.env` loading, reusable Parquet feature workflow, and feature table tests. | [Feature Pipeline](feature-pipeline.md) | `git show --stat 050a3fa` |
| Scouting database layer | 2026-05-21 | `7b40629` | Add SQLModel scouting layer and feature merge | Added SQLModel scouting database schema and feature merge behavior. | [Scouting Data Layer](scouting-data-layer.md) | `git show --stat 7b40629` |
| Scouting ingestion docs | 2026-05-21 | `9c7e771` | Add scouting data ingestion guide | Added detailed scouting ingestion guidance. | [Scouting Data Ingestion](scouting-data-ingestion.md) | `git show --stat 9c7e771` |
| Repo skills and ML docs | 2026-05-21 | `434fd37` | Add repo Codex skills and ML docs | Added local agent skills and ML documentation references. | `.agents/skills/`, early docs | `git show --stat 434fd37` |
| V5.5 pretraining | 2026-05-22 | `ff5d3da` | Implement LatentStrat V5.5 pretraining | Added text/narrative prior pretraining, prior feature generation, prior inspection, embedding store, and richer docs/tests. | [Prior Training](prior-training.md), `src/latentstrat/pretrain_*` | `git show --stat ff5d3da` |
| Obsidian vault | 2026-05-22 | `5d587f8` | Add obsidian | Added Obsidian vault metadata under `docs/.obsidian/`. | `docs/.obsidian/` | `git show --stat 5d587f8` |
| V5.7 sensor fusion | 2026-05-22 | `680bbb5` | Implement V5.7 sensor fusion training | Added V5.7 match-spine targets, sidecars, prior grid, richer prior features, heterogeneous training, and expanded tests. | [V5.7 Notes](V5.7.md), [Feature Pipeline](feature-pipeline.md) | `git show --stat 680bbb5` |
| TensorBoard policy | 2026-05-23 | `cc58c31` | Document TensorBoard training policy | Documented TensorBoard as the expected observability path for training commands. | [Training And Validation](training-and-validation.md) | `git show --stat cc58c31` |
| Current docs and V5.8 state | 2026-05-24 | `Unreleased` | Working tree after `cc58c31` | Documents V5.6.4 prior, V5.7.2 stability, V5.8 walk-forward, current artifacts, model architecture math, and experiment ledger. | [Current State](current-state.md), [Experiment Ledger](experiment-ledger.md), [Model Architecture Reference](model-architecture-reference.md) | commit pending |
| V5.8 frozen comparison baseline | 2026-06-01 | `b8e9a6c` | Freeze V5.8 diagnostics baseline | Adds prediction/config/hash exports for walk-forward replay and marks the historical source boundary with `v5.8-baseline`. | `artifacts/baselines/v5.8/`, `baselines/v5.8-baseline.json` | `git show --stat v5.8-baseline` |
| V6-Lite historical score artifact | 2026-06-01 | `Unreleased` | Working tree after `v5.8-baseline` | Adds durable historical TBA match-breakdown storage, per-season raw schemas, a shared 16D bottleneck, and offline artifact exports. Keeps score runtime integration disabled while retaining later award, ranking, and pick scaffolds. | [V6-Lite](V6-Lite.md), `src/latentstrat/world_model/match_breakdown/` | commit pending |

## Semantic Version Map

| Semantic version | Closest Git anchor | State type | What it means | Notes |
|---|---|---|---|---|
| V5.5 | `ff5d3da` | Committed | Text/narrative pretraining era. | This is the cleanest committed anchor for prior pretraining before V5.6 transductive refinements. |
| V5.6 | `680bbb5` plus local artifacts | Mixed | Transductive team-number dictionary and stripped checkpoint handoff. | The committed V5.7 work contains many V5.6-era prior pieces; exact V5.6 experiment states are best understood from local artifacts. |
| V5.6.1 | `680bbb5` plus `artifacts/prior_v561_latent16/` | Local artifact era | Multi-task prior with OpenAI plus EPA target. | No separate clean commit anchor; use artifact evidence and docs. |
| V5.6.2 | current working tree plus `artifacts/prior_v562_latent16/` | Local artifact era | Cultural prior attempt with broken normalized EPA masks. | Important negative result, not a promoted code state. |
| V5.6.3 | current working tree plus `artifacts/prior_v563_latent16/` | Local artifact era | Corrected normalized EPA extraction and grouped EPA trajectory loss. | Used as the prior feature schema for V5.6.4 training. |
| V5.6.4 | current working tree plus `artifacts/prior_v564_latent16/` | Current local state | Feature-summed OpenAI loss with 16D prior. | Current recommended prior checkpoint until superseded. |
| V5.7 | `680bbb5` | Committed | Match-spine expansion, sidecars, heterogeneous training, and V5.7 target heads. | This is the main committed sensor-fusion anchor. |
| V5.7.2 | current working tree plus V5.7.2 artifacts | Current local state | Stability patch: log-var clamp, best restoration, cosine LR, AdamW decay. | Not separately tagged or committed as a named version yet. |
| V5.8 | tag `v5.8-baseline` at `b8e9a6c` plus archived artifacts | Tagged historical baseline | Walk-forward validation, common metrics, canonical week handling, prediction export. | Comparison boundary for V6 promotion. |
| V6-Lite | current working tree after `v5.8-baseline` | Unreleased active architecture | Offline historical score representations plus reserved frozen-target scaffolds. | Score attachment is deferred; no V6 phase is promoted yet. |

## Maintenance Rule

When a commit changes a named version, public CLI, model behavior, training default, promoted artifact, or experiment conclusion:

1. Add or update a row in this changelog.
2. Update [Current State](current-state.md) if the recommended stack or caveats changed.
3. Update [Experiment Ledger](experiment-ledger.md) if a run result or artifact changed.
4. Use `Unreleased` or `Pending commit` until the work has a real commit hash, then replace it with the hash.
