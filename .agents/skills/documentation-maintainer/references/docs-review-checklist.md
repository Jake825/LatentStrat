# Docs Review Checklist

Use this checklist before finalizing Markdown documentation changes.

## Accuracy

- Commands match the current CLI.
- Dependencies match `pyproject.toml` or `environment.yml`.
- Env vars match repo conventions.
- File paths are current.
- Claims about behavior are backed by code, tests, or existing docs.

## Structure

- README remains high-level.
- Detailed workflow content lives in `docs/`.
- New content does not duplicate another doc without a reason.
- Links use relative paths where possible.

## Style

- Headings are concise.
- Code fences have useful language tags.
- Lists are not overly nested.
- Examples are runnable from the repo root.
- No stale placeholder text remains.

## Validation

- Run `git diff --check`.
- Run any doc-specific validation requested by the user.
- For docs that include generated command output, verify the command output if practical.
