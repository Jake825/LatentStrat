# Command Documentation Checklist

Inspect command definitions and live help before writing examples:

```powershell
latentstrat --help
latentstrat pretrain --help
latentstrat season --help
latentstrat artifacts --help
latentstrat scouting --help
latentstrat dev --help
```

Use the narrowest subgroup help for options. The supported top-level families are `pretrain`, `season`, `artifacts`, `scouting`, and `dev`; `experimental` is an explicitly non-canonical top-level family, while diagnostics are nested under `dev`. Prefer grouped commands over deprecated flat compatibility aliases, and do not revive tombstoned workflows in new docs.

## Configuration

- Read package metadata, Python requirements, dependencies, and console entry points from `pyproject.toml`.
- Use `TBA_API_KEY` for live The Blue Alliance access; never publish real secret values.
- Confirm paths against `src/latentstrat/paths.py` and active documentation.

## Validation

- Documentation-only changes: `git diff --check` plus any relevant command help.
- Skill changes: run the skill-creator `quick_validate.py` against every changed skill.
- Runtime changes: run targeted tests first, then the broader Ruff and pytest checks proportional to risk.
