# Markdown Docs Workflow

Use this workflow for README and `docs/*.md` updates.

## Before Editing

Inspect the source of truth:

- CLI behavior in `src/latentstrat/cli.py`.
- Project metadata and dependencies in `pyproject.toml`.
- Existing docs under `docs/`.
- Tests for expected behavior.
- Relevant source modules, tests, and authoritative external references for TBA, Statbotics, scouting, timing, or FRC concepts.

## Writing Style

- Keep headings short and specific.
- Prefer direct prose and compact lists.
- For notes under `docs/`, do not manually hard-wrap prose. Let Obsidian soft-wrap paragraphs, and use explicit line breaks only for Markdown structure.
- Use fenced code blocks with language tags.
- Use `powershell` for Windows-specific local commands and `bash` only for portable shell examples.
- Keep examples runnable from the repo root.
- Use ASCII punctuation unless an existing file intentionally uses non-ASCII.

## Avoiding Stale Docs

- Do not document behavior based on memory when code can be inspected.
- Do not add roadmap claims as if they already exist.
- Avoid duplicating command lists across many files. Link to the more specific doc when possible.
- If a command requires secrets, mention the environment variable and avoid including real values.

## Final Doc Summary

When done, summarize:

- Docs changed.
- Behavior or commands documented.
- Source files inspected for accuracy.
- Validation run.
