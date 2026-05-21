---
name: git-workflow
description: Use when inspecting or managing Git state in the repo, preparing commits, reviewing diffs, staging changes, writing commit messages, handling branches, troubleshooting common Git problems, or planning release-safe Git operations. Covers dirty-worktree safety, user-change preservation, branch hygiene, commit discipline, and release checklist guidance.
---

# Git Workflow

Use this skill when a task involves Git state, commits, branches, tags, releases, or explaining what changed.

## Core Directives

1. Inspect before acting. Start with `git status --short --branch` and targeted diffs before staging or committing.
2. Preserve user work. Never discard, overwrite, reset, checkout away, or clean changes unless the user explicitly requests that exact destructive action.
3. Stage intentionally. Prefer path-specific staging after reviewing diffs. Do not stage unrelated user changes.
4. Keep commits focused. Group changes by coherent intent, not by convenience.
5. Treat remote actions as explicit-request-only. Do not push, tag, create releases, force-push, or modify remote branches without direct user instruction.

## Routing

- Read `references/safe-git-workflow.md` before staging, committing, rebasing, cleaning, or resolving a dirty worktree.
- Read `references/commit-and-branching.md` before preparing commits or branch workflows.
- Read `references/release-workflow.md` before version bumps, tags, release notes, or release prep.
- Read `references/troubleshooting.md` when resolving Git errors or confusing repo state.

## Templates

- Use `assets/commit-message-template.md` when drafting commit messages.
- Use `assets/release-checklist-template.md` when planning a release.
- Use `assets/release-notes-template.md` when drafting release notes.

## Coordinate With Other Skills

- Use `$documentation-maintainer` when release prep or commits require README, docs, command examples, or release notes updates.
