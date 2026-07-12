# Scouting Importer Workflow

Require a source audit, mapping plan, representative fixture, and schema decision before implementation.

## Implementation

1. Read the source with Pandas or a format-specific parser.
2. Normalize keys and values in named helpers.
3. Construct records at one explicit table grain.
4. Write with `session.merge(...)` so reruns update rows by primary key.
5. Commit once per import unless source size requires controlled batching.
6. Report rejected rows with source context.

Do not silently drop important unmapped fields. Keep source-specific logic in its adapter rather than weakening shared schema or normalization rules.

Use `assets/csv-importer-skeleton.py` and `assets/importer-pytest-skeleton.py` only as starting points; update imports, mappings, constraints, and assertions to match the actual source and current grouped CLI.
