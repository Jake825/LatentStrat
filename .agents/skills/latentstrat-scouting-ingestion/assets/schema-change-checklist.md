# Scouting Schema Change Checklist

## Proposed Change

- Model/table:
- Field/table name:
- Data type:
- Default/nullability:
- Validation:
- Timing classification:

## Required Updates

- `src/frc/scouting.py`:
- `src/latentstrat/season/features.py`:
- Importer or adapter:
- Docs:
- Tests:

## Compatibility

- Existing database impact:
- Existing Parquet impact:
- Feature merge prefix:
- Leakage risk:

## Acceptance Tests

- Database schema initializes:
- Importer writes the new field:
- Feature merge includes the expected column:
- Parquet round trip preserves dtype:
- Training inputs remain leakage-safe:
