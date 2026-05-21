# Release Workflow

LatentStrat currently has no established tags, changelog, release automation, or GitHub workflow files. Treat release work as explicit, checklist-driven preparation unless the repo later adds a formal process.

## Release Guardrails

- Do not tag, push, publish packages, or create GitHub releases without explicit user instruction.
- Do not invent a changelog file unless the user asks for one.
- Verify version sources before changing versions. The current package version is in `pyproject.toml`.
- Keep release notes factual and based on committed or staged diffs.

## Preparation Checklist

Use `assets/release-checklist-template.md`.

Recommended checks:

```powershell
git status --short --branch
git log --oneline --decorate -10
git diff --check
ruff check .
pytest
```

Adjust validation to the release scope. Skill-only releases can validate affected skills instead of running Python tests.

## Version and Tag Policy

If the user requests a versioned release:

1. Confirm the target version.
2. Update version files only if requested.
3. Prepare release notes from Git history and user-facing changes.
4. Create a tag only after the user explicitly approves tagging.
5. Push only after the user explicitly approves pushing.

Example tag command when approved:

```powershell
git tag -a v0.1.0 -m "Release v0.1.0"
```

## Release Notes

Use `assets/release-notes-template.md`.

Group notes by user-facing area:

- Added
- Changed
- Fixed
- Documentation
- Validation

Do not list internal implementation details unless they affect users, operators, or future maintainers.
