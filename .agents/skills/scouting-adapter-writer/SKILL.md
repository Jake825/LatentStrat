---
name: scouting-adapter-writer
description: Use when writing Python adapter/importer code that loads raw FRC scouting CSV or spreadsheet data into the LatentStrat SQLModel scouting database. Covers importer structure, normalization helpers, session.merge writes, schema extension proposals, tests, validation commands, and feature rebuild checks.
---

# Scouting Adapter Writer

## Overview

Use this skill to generate or review importer code for moving normalized scouting source data into LatentStrat's scouting SQLite database.

## Core Rules

- Start from a source audit and column mapping plan.
- Use current SQLModel models in `src/frc/scouting.py` unless the task explicitly includes a schema extension.
- Use `session.merge(...)`, not `session.add(...)`, so re-running importers updates rows by primary key.
- Keep source-specific fields visible. If a useful field does not fit the schema, propose explicit schema, docs, merge, and test changes.
- Add focused tests for key normalization, duplicate/update behavior, and merged feature columns.

## References

- Read [references/adapter-workflow.md](references/adapter-workflow.md) before writing importer code.
- Use [assets/csv-importer-skeleton.py](assets/csv-importer-skeleton.py) as a starting point for generated CSV importers.
- Use [assets/importer-pytest-skeleton.py](assets/importer-pytest-skeleton.py) as a starting point for adapter tests.
- Use [assets/schema-extension-proposal-template.md](assets/schema-extension-proposal-template.md) when source fields require schema edits.
- Use `$scouting-data-normalization` for conversion helpers.
- Use `$latentstrat-scouting-db` for schema, merge prefixes, and validation commands.
- Use `$tba-api` when resolving match keys or alliance colors.
- Use `$frc-time-aware-analysis` when importer fields may later feed predictions, simulations, or validation datasets.
