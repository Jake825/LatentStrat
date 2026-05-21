# Safe Git Workflow

Use this reference before changing Git state.

## Initial Inspection

Start with:

```powershell
git status --short --branch
```

Then inspect relevant changes:

```powershell
git diff -- path/to/file
git diff --cached -- path/to/file
git ls-files --others --exclude-standard
```

For untracked skill or docs work, inspect file contents directly before staging.

## Dirty Worktree Rules

- Treat all existing changes as user-owned unless you made them in the current task.
- Do not revert, checkout, reset, clean, or overwrite user changes.
- If unrelated changes exist, ignore them unless the requested Git operation would include them.
- If changes overlap with your work, read the files and integrate carefully.
- If safe separation is impossible, stop and explain the conflict.

## Staging Discipline

Prefer path-specific staging:

```powershell
git add path/to/file path/to/other-file
```

Avoid broad staging commands unless the user asked to commit everything and the diff has been reviewed.

Before committing:

```powershell
git diff --cached --stat
git diff --cached
```

Summarize what is staged and call out anything deliberately left unstaged.

## Destructive Commands

Do not run these unless the user explicitly requests the exact outcome:

- `git reset --hard`
- `git checkout -- <path>`
- `git restore <path>`
- `git clean`
- force-push commands
- branch deletion

If a destructive command is requested, restate the affected paths or refs before executing when practical.

## Final Git Summary

When reporting Git work, include:

- Branch name and ahead/behind state if relevant.
- Files staged or committed.
- Commit hash if a commit was created.
- Validation commands run.
- Any remaining unstaged or untracked changes that matter.
