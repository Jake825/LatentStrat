---
name: documentation-maintainer
description: Use when adding, reviewing, or maintaining Markdown documentation for LatentStrat, especially README.md and docs/*.md. Covers keeping docs aligned with CLI behavior, project configuration, domain skills, command examples, validation instructions, and concise repo documentation style without introducing generated docs tooling or Python docstring policy.
---

# Documentation Maintainer

Use this skill when a task changes or reviews LatentStrat Markdown documentation.

## Core Directives

1. Keep docs factual and tied to repo behavior. Inspect code, CLI help, configs, or tests before documenting commands or APIs.
2. Prefer concise Markdown with runnable examples.
3. Update the narrowest relevant document instead of duplicating the same explanation across many files.
4. Keep README high-level and route deeper details to `docs/*.md`.
5. Do not introduce mkdocs, Sphinx, generated API docs, or Python docstring policy unless the user explicitly asks.

## Routing

- Read `references/markdown-docs-workflow.md` before editing or reviewing Markdown docs.
- Read `references/repo-docs-map.md` to choose the right LatentStrat document.
- Read `references/command-docs-checklist.md` before documenting CLI commands, env vars, dependencies, or validation commands.
- Read `references/docs-review-checklist.md` before finalizing doc changes.

## Templates

- Use `assets/doc-update-plan-template.md` to plan larger documentation updates.
- Use `assets/docs-change-checklist.md` to review documentation changes before final response.

## Coordinate With Other Skills

- Use `$tba-api` when documenting The Blue Alliance ingestion, key formats, or TBA environment variables.
- Use `$statbotics` when documenting EPA, Statbotics provider behavior, or predictive analytics data.
- Use `$frc-time-aware-analysis` when documenting leakage, validation splits, or feature timing.
- Use `$latentstrat-feature-pipeline`, `$pytorch-set-transformer`, and `$latentstrat-model-evaluation` when documenting ML feature construction, model architecture, metrics, artifacts, or embeddings.
- Use `$latentstrat-scouting-db`, `$frc-scouting-data-types`, and related scouting skills when documenting scouting ingestion or schema behavior.
- Use `$git-workflow` when documentation work is part of commit or release preparation.
