# LatentStrat Vault Contract

`docs/` is both the LatentStrat documentation directory and the Obsidian vault root.

## Skill Boundaries

- `documentation-maintainer` owns repo-facing documentation: README links, CLI examples, runbooks, architecture docs, and GitHub-readable Markdown style.
- `obsidian-cli` owns vault-native operations: wikilinks, backlinks, tags, properties, MOCs, graph diagnostics, note movement, and Obsidian-only formatting.
- Use both skills when a documentation change also affects vault navigation or graph structure.

## New Note Visibility

If a new `docs/` note is durable repo documentation, link it from `README.md` or the narrowest relevant central docs page.

If a note is intentionally unlinked, private, or experimental, state that explicitly in the final response.

## Local Workspace Files

Do not edit or stage `.obsidian/workspace.json`, `graph.json`, appearance settings, or other local workspace churn unless the user explicitly asks.

## Assets

Use `docs/assets/` for durable repo-facing diagrams, plots, and images. Repo-facing and hybrid notes should use standard Markdown image links so they render on GitHub and in Obsidian.
