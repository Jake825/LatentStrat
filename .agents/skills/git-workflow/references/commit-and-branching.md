# Commit and Branching Guidance

Use this reference when preparing commits or branch workflows.

## Commit Shape

A good commit should:

- Have one coherent purpose.
- Include required tests or validation for the changed behavior.
- Avoid mixing unrelated formatting, generated artifacts, and feature work.
- Keep user-owned unrelated changes out of the staged set.

## Commit Message Style

Use concise imperative subjects:

- `Add scouting source audit skill`
- `Fix feature merge for missing scouting rows`
- `Document Statbotics provider cache behavior`

Use the body when it helps explain:

- Why the change exists.
- Important behavior or compatibility notes.
- Validation performed.

Use `assets/commit-message-template.md` for larger commits.

## Branch Workflow

Default branch is `main`. Inspect current state before branch operations:

```powershell
git status --short --branch
git branch --show-current
```

When creating a branch:

```powershell
git switch -c short-topic-name
```

Do not switch branches if uncommitted changes would be carried unexpectedly. Inspect first and explain options.

## Reviewing Changes

Before proposing or creating a commit:

```powershell
git diff --stat
git diff
git diff --cached --stat
git diff --cached
```

For documentation-only or skill-only changes, `git diff --check` is usually enough validation unless files have their own validators.

For Python runtime changes, prefer:

```powershell
ruff check .
pytest
```

Use narrower tests when the change is small and the full suite is expensive.
