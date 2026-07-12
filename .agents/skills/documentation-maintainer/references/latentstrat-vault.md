# LatentStrat Vault Contract

`docs/` is both the active documentation directory and the Obsidian vault root.

## Modes

- Keep repository-facing pages, CLI examples, runbooks, and architecture material as GitHub-readable Markdown.
- Use Obsidian-native links, tags, properties, backlinks, graph diagnostics, and note movement only when the request explicitly asks for vault behavior.
- A note that is durable repository documentation should be linked from `docs/index.md`, `README.md`, or the narrowest central page.
- State explicitly when a new note is intentionally unlinked or experimental.

## Workspace Safety

Do not edit or stage `.obsidian/workspace.json`, `graph.json`, appearance settings, or other local workspace churn unless explicitly requested.

Use `docs/assets/` for durable diagrams, plots, and images. Prefer standard Markdown image links for content that should render in both GitHub and Obsidian.
