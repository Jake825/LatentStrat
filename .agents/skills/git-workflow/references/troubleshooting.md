# Git Troubleshooting

Use this reference when Git reports confusing state or command failures.

## Common Checks

```powershell
git status --short --branch
git log --oneline --decorate -10
git remote -v
git branch -vv
```

## Untracked Files

Untracked files are not safe to delete by default. Inspect and classify them:

- New source files or docs from the current task.
- User-created files.
- Generated artifacts that should remain ignored.
- Temporary files that can be removed only with user approval if tracked status is unclear.

## Merge Conflicts

When conflicts occur:

1. Run `git status --short`.
2. Open conflicted files.
3. Preserve both user intent and requested change.
4. Resolve markers manually.
5. Run targeted validation.
6. Show the resolved files before committing.

## Detached HEAD

If `git status` reports detached HEAD:

- Do not commit unless the user requested work at that exact commit.
- Suggest creating a branch from the current commit if work should be preserved.

## Failed Push

If push fails because the remote moved:

- Do not force-push by default.
- Fetch and inspect divergence.
- Prefer a normal rebase or merge only after explaining the state.

Useful inspection:

```powershell
git fetch origin
git status --short --branch
git log --oneline --left-right --graph HEAD...@{u}
```
