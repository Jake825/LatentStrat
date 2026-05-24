---
name: obsidian-cli
description: Interact with this Obsidian vault using the Obsidian CLI for reading, searching, and explicit note/file changes, with Windows PowerShell-first execution guidance. Use when requests involve notes, links, tags, tasks, properties, or creating/editing/moving/renaming notes in this vault.
---

Use the Obsidian CLI as the primary interface to this vault.

LatentStrat vault contract:
- The Obsidian vault root is `docs/`.
- Repo-facing documentation in `docs/*.md` should remain GitHub-readable unless the user explicitly asks for Obsidian-native syntax.
- Before creating or modifying Obsidian-native notes, read `references/latentstrat-vault.md` and `references/obsidian-format-guide.md`.
- Do not edit or stage `.obsidian/workspace.json`, `graph.json`, appearance, or other local workspace churn unless explicitly requested.

Windows execution notes:
- This vault is on Windows.
- Prefer PowerShell-compatible commands.
- Assume `obsidian` is on PATH only after enabling Obsidian CLI in Settings > General > Command line interface and restarting the terminal.
- If `obsidian` is not found, verify Obsidian 1.12.7+ is installed and CLI registration is complete.
- Assume the first CLI call may launch Obsidian if it is not already running.
- Prefer double quotes for paths and arguments.
- Prefer exact vault names and exact vault-relative note paths.
- Do not assume Unix tools like `grep`, `sed`, or `cat` are available.
- Prefer `Get-Content`, `Set-Content`, and PowerShell string handling when shell work is needed.
- Prefer official `obsidian` CLI commands over direct file editing when possible.

Principles:
- Prefer Obsidian CLI commands over direct manual file edits when possible.
- Start with read-only commands when exploring the vault.
- Use exact paths when available.
- Keep changes small and explicit.
- For write actions, state what file is being changed and how.

Preferred Windows script entry points:
- `.agents/skills/obsidian-cli/scripts/obsidian_read.ps1 -Vault "<vault-name>" -Path "<path>"`
- `.agents/skills/obsidian-cli/scripts/obsidian_search.ps1 -Vault "<vault-name>" -Query "<query>"`
- `.agents/skills/obsidian-cli/scripts/obsidian_append.ps1 -Vault "<vault-name>" -Path "<path>" -Content "<content>"`
- `.agents/skills/obsidian-cli/scripts/obsidian_create.ps1 -Vault "<vault-name>" -Name "<note-name>"`
- `.agents/skills/obsidian-cli/scripts/obsidian_move.ps1 -Vault "<vault-name>" -From "<path>" -To "<path>"`
- `.agents/skills/obsidian-cli/scripts/Run-VaultDiagnostics.ps1 -VaultRoot "<abs-vault-path>"`

Invoke wrappers with:
- `powershell -ExecutionPolicy Bypass -File "<script>" ...`

Diagnostics report behavior:
- `Run-VaultDiagnostics.ps1` defaults to the LatentStrat `docs/` vault.
- Pass `-OutputPath` for diagnostics validation to avoid mutating tracked docs.
- Diagnostics taxonomy must come from an explicit config file; it must not assume a personal vault structure.

Capabilities:
- Read notes.
- Search notes.
- List files and folders.
- Inspect tags, properties, tasks, links, backlinks, and unresolved links.
- Create, append, prepend, move, and rename notes.
- Use history and diff tooling to inspect changes when available.

Coordinate with `$documentation-maintainer` when a note is durable repo documentation, changes CLI/runbook docs, or needs to be linked from README or a central docs page.
Do not invent metadata, folder, or naming rules.
