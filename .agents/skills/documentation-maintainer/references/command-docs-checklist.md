# Command Docs Checklist

Use this before documenting commands, dependencies, or environment variables.

## Verify Commands

Inspect command definitions before writing examples:

```powershell
python -m latentstrat.cli --help
latentstrat --help
```

Use the command form already used by surrounding docs unless the task specifically changes it.

## Project Configuration

Use `pyproject.toml` for:

- Package name.
- Python version requirement.
- Runtime dependencies.
- Optional dev dependencies.
- Console script entry points.
- Ruff and pytest configuration.

Use `environment.yml` only when documenting Conda setup.

## Environment Variables

Current key environment variable:

- `TBA_API_KEY`: used for live The Blue Alliance ingestion.

Do not document `TBA_AUTH_KEY` for LatentStrat examples.

## Validation Commands

For docs-only changes:

```powershell
git diff --check
```

For skill changes:

```powershell
python C:\Users\Jelle\.codex\skills\.system\skill-creator\scripts\quick_validate.py .agents\skills\<skill-name>
```

For runtime Python changes:

```powershell
ruff check .
pytest
```

If only a narrow runtime area changed, targeted tests are acceptable when reported clearly.
