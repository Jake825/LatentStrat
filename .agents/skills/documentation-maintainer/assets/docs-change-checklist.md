# Docs Change Checklist

## Accuracy

- Commands match current code.
- Dependency and version claims match config.
- Environment variables are correct.
- Examples are safe and do not include secrets.

## Placement

- README stays concise.
- Detailed guidance is in the most relevant `docs/*.md` file.
- Related docs are linked instead of duplicated.

## Style

- Markdown is readable.
- Code fences include language tags.
- Paths and commands use backticks.
- No stale placeholder text remains.

## Validation

- `git diff --check` passed.
- Relevant command output or CLI help was checked when documenting commands.
