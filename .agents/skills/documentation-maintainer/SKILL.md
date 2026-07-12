---
name: documentation-maintainer
description: Maintain LatentStrat repository documentation. Use when the requested deliverable adds, edits, reviews, links, or reorganizes README.md, AGENTS.md, active docs/*.md pages, or the docs/ Obsidian vault. Do not use for code-only work, generic Markdown outside this repository, or a passing documentation mention in another subsystem task.
---

# Documentation Maintainer

Keep LatentStrat documentation aligned with current repository behavior.

## Workflow

1. Inspect source, tests, CLI help, configuration, or artifacts before stating behavior.
2. Use `references/repo-docs-map.md` to select the narrowest active page.
3. Use `references/command-docs-checklist.md` for commands, options, environment variables, and validation examples.
4. Use `references/markdown-docs-workflow.md` for editing conventions.
5. Use `references/docs-review-checklist.md` before finalizing changes.

Keep `README.md` concise and route details to active pages linked from `docs/index.md`. Treat `docs/archive/` as immutable historical context unless the user explicitly requests archival maintenance.

## Obsidian Mode

Use Obsidian-specific behavior only when the request explicitly involves vault links, tags, properties, tasks, backlinks, note movement, or diagnostics. The vault root is `docs/`.

- Read `references/latentstrat-vault.md` and `references/obsidian-format-guide.md` before Obsidian-native edits.
- Prefer the wrappers in `scripts/` for CLI operations and diagnostics.
- Keep repo-facing pages GitHub-readable and avoid modifying `.obsidian/workspace.json` or other local workspace churn.

Use templates in `assets/` only for substantial documentation changes that benefit from them.
