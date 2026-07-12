#!/usr/bin/env python3
"""
Vault diagnostics engine.

Builds a directed graph of markdown notes using vault-relative canonical IDs and
writes a replacement markdown dashboard report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import networkx as nx
from networkx.algorithms import community

WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
EXCLUDED_DIR_NAMES = {
    ".obsidian",
    ".agents",
    ".codex",
    ".trash",
    ".git",
    "__pycache__",
}

DEFAULT_THEME_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "work",
    "note",
    "notes",
    "daily",
    "logs",
    "log",
    "meeting",
    "meetings",
    "task",
    "tasks",
    "project",
    "projects",
    "moc",
    "dashboard",
    "administration",
    "reference",
    "archive",
    "personal",
    "builders",
    "vault",
    "systems",
    "system",
    "workflow",
    "guide",
    "you",
    "your",
    "yours",
    "they",
    "them",
    "their",
    "there",
    "these",
    "those",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
    "must",
    "use",
    "using",
    "used",
    "new",
    "old",
    "also",
    "into",
    "onto",
    "over",
    "under",
    "through",
    "between",
    "about",
    "after",
    "before",
    "during",
    "within",
    "without",
    "have",
    "has",
    "had",
    "did",
    "does",
    "done",
    "not",
    "yes",
    "no",
    "etc",
    "etcetera",
    "today",
    "tomorrow",
    "yesterday",
    "http",
    "https",
    "com",
    "org",
    "www",
    "jpg",
    "png",
    "jpeg",
    "md",
    "pdf",
    "doc",
    "docs",
    "file",
    "files",
    "folder",
    "folders",
    "template",
    "templates",
    "report",
    "reports",
    "plan",
    "plans",
    "status",
    "overview",
    "general",
    "misc",
    "miscellaneous",
    "meetingnotes",
    "worklog",
    "mathrm",
    "frac",
    "left",
    "right",
    "begin",
    "end",
    "cdot",
    "text",
}

THEME_STOPWORDS = set(DEFAULT_THEME_STOPWORDS)
SECTION_THEME_MAP: dict[str, str] = {}


def load_diagnostics_config(config_path: Path) -> tuple[set[str], dict[str, str]]:
    """Load optional vault-specific theme hints from JSON."""
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid diagnostics config JSON: {config_path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError(f"diagnostics config must be a JSON object: {config_path}")

    extra_stopwords = raw_config.get("extra_stopwords", [])
    if not isinstance(extra_stopwords, list) or not all(
        isinstance(value, str) for value in extra_stopwords
    ):
        raise ValueError("diagnostics config 'extra_stopwords' must be a list of strings")

    section_theme_map = raw_config.get("section_theme_map", {})
    if not isinstance(section_theme_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in section_theme_map.items()
    ):
        raise ValueError("diagnostics config 'section_theme_map' must be an object of strings")

    normalized_stopwords = {value.lower().strip() for value in extra_stopwords if value.strip()}
    normalized_sections = {
        key.strip(): value.strip()
        for key, value in section_theme_map.items()
        if key.strip() and value.strip()
    }
    return normalized_stopwords, normalized_sections


def normalize_key(value: str) -> str:
    key = value.strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    key = key.lstrip("/")
    key = key.rstrip("/")
    return key.lower()


def strip_md_suffix(value: str) -> str:
    if value.lower().endswith(".md"):
        return value[:-3]
    return value


def should_skip(rel_path: Path, excluded_rel_paths: set[str]) -> bool:
    rel_posix = rel_path.as_posix()
    if rel_posix in excluded_rel_paths:
        return True
    parts = rel_path.parts[:-1]
    for part in parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
        if part.startswith("."):
            return True
    return False


def iter_markdown_files(vault_root: Path, excluded_rel_paths: set[str]) -> Iterable[Path]:
    for md_file in vault_root.rglob("*.md"):
        rel_path = md_file.relative_to(vault_root)
        if should_skip(rel_path, excluded_rel_paths):
            continue
        yield md_file


def parse_link_target(inner: str) -> str:
    target = inner.split("|", 1)[0].strip()
    target = target.split("#", 1)[0].strip()
    return target


def clean_segment_label(segment: str) -> str:
    value = segment
    value = re.sub(r"^\d+[_\-\s]*", "", value)
    value = value.replace("_", " ").strip()
    return value


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9&_\-]{1,}", text):
        token = raw.lower().strip("_-")
        if not token:
            continue
        token = token.replace("&", "and")
        if token.isdigit():
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
            continue
        if len(token) < 3 and not re.search(r"\d", token):
            continue
        if token in THEME_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def extract_note_tokens(title: str, rel_path: str, content: str) -> list[str]:
    tokens: list[str] = []
    tokens.extend(tokenize_text(title))

    rel_parts = Path(rel_path).parts[:-1]
    for part in rel_parts:
        tokens.extend(tokenize_text(clean_segment_label(part)))

    if content:
        heading_lines = [
            line.lstrip("# ").strip()
            for line in content.splitlines()
            if line.lstrip().startswith("#")
        ]
        if heading_lines:
            tokens.extend(tokenize_text(" ".join(heading_lines[:20])))

        link_targets: list[str] = []
        for match in WIKILINK_PATTERN.finditer(content):
            link_targets.append(parse_link_target(match.group(1)))
            if len(link_targets) >= 60:
                break
        if link_targets:
            tokens.extend(tokenize_text(" ".join(link_targets)))

    return tokens


def summarize_cluster_theme(sub_graph: nx.DiGraph, cluster_nodes: Sequence[str]) -> tuple[str, str]:
    if not cluster_nodes:
        return "No theme inferred.", "No cluster members."

    section_counter: Counter[str] = Counter()
    subsection_counter: Counter[str] = Counter()
    keyword_doc_counter: Counter[str] = Counter()

    for node_id in cluster_nodes:
        parts = Path(strip_md_suffix(node_id)).parts
        if len(parts) > 0:
            section_counter[parts[0]] += 1
        if len(parts) > 1:
            subsection_counter[parts[1]] += 1

        note_tokens = sub_graph.nodes[node_id].get("tokens", [])
        keyword_doc_counter.update(set(note_tokens))

    cluster_size = len(cluster_nodes)
    top_section, top_section_count = (
        section_counter.most_common(1)[0] if section_counter else ("", 0)
    )
    top_subsection, top_subsection_count = (
        subsection_counter.most_common(1)[0] if subsection_counter else ("", 0)
    )

    section_ratio = top_section_count / cluster_size if cluster_size else 0.0
    subsection_ratio = top_subsection_count / cluster_size if cluster_size else 0.0
    second_section_count = section_counter.most_common(2)[1][1] if len(section_counter) > 1 else 0
    second_subsection_count = (
        subsection_counter.most_common(2)[1][1] if len(subsection_counter) > 1 else 0
    )

    if (
        subsection_ratio >= 0.35
        and (top_subsection_count - second_subsection_count) >= 3
        and top_subsection
    ):
        scope = SECTION_THEME_MAP.get(top_subsection, clean_segment_label(top_subsection).lower())
    elif (
        section_ratio >= 0.35
        and (top_section_count - second_section_count) >= 3
        and top_section
    ):
        scope = SECTION_THEME_MAP.get(top_section, clean_segment_label(top_section).lower())
    else:
        scope = "mixed vault notes"

    min_doc_frequency = max(2, int(cluster_size * 0.08))
    dominant_terms: list[str] = []
    for token, count in keyword_doc_counter.most_common(40):
        if count < min_doc_frequency:
            continue
        if token in THEME_STOPWORDS:
            continue
        if re.fullmatch(r"[a-z]{1,2}", token):
            continue
        dominant_terms.append(token)
        if len(dominant_terms) >= 4:
            break

    if dominant_terms:
        theme = f"{scope} with emphasis on {', '.join(dominant_terms[:3])}"
    else:
        theme = scope

    evidence_parts: list[str] = []
    if top_section:
        evidence_parts.append(
            f"Primary section: {clean_segment_label(top_section)} "
            f"({top_section_count}/{cluster_size})"
        )
    if dominant_terms:
        evidence_parts.append(f"Top keywords: {', '.join(dominant_terms)}")

    if theme:
        theme = theme[:1].upper() + theme[1:]
    return theme, "; ".join(evidence_parts)


def resolve_target(
    target: str,
    source_rel_path: str,
    basename_map: dict[str, list[str]],
    path_map: dict[str, list[str]],
) -> tuple[str | None, str | None]:
    normalized_target = normalize_key(target)
    if not normalized_target:
        return None, "empty"

    candidates: set[str] = set()
    has_path_separator = "/" in normalized_target

    if has_path_separator:
        path_keys = {normalized_target, strip_md_suffix(normalized_target)}
        source_dir = Path(source_rel_path).parent.as_posix()
        if source_dir != ".":
            relative_key = normalize_key(f"{source_dir}/{normalized_target}")
            path_keys.add(relative_key)
            path_keys.add(strip_md_suffix(relative_key))
        for key in path_keys:
            for candidate in path_map.get(key, []):
                candidates.add(candidate)
    else:
        base_key = strip_md_suffix(Path(normalized_target).name)
        for candidate in basename_map.get(base_key, []):
            candidates.add(candidate)

    if len(candidates) == 1:
        return next(iter(candidates)), None
    if not candidates:
        return None, "missing"
    return None, "ambiguous"


def build_graph(vault_root: Path, excluded_rel_paths: set[str]) -> tuple[nx.DiGraph, list[str]]:
    graph = nx.DiGraph()
    warnings: list[str] = []

    basename_map: defaultdict[str, list[str]] = defaultdict(list)
    path_map: defaultdict[str, list[str]] = defaultdict(list)
    canonical_paths: list[str] = []

    for md_file in iter_markdown_files(vault_root, excluded_rel_paths):
        rel_with_ext = md_file.relative_to(vault_root).as_posix()
        rel_no_ext = strip_md_suffix(rel_with_ext)
        basename = md_file.stem

        graph.add_node(
            rel_with_ext,
            kind="real",
            title=basename,
            link_ref=rel_no_ext,
            tokens=extract_note_tokens(title=basename, rel_path=rel_with_ext, content=""),
        )
        canonical_paths.append(rel_with_ext)

        basename_map[basename.lower()].append(rel_with_ext)
        path_map[normalize_key(rel_with_ext)].append(rel_with_ext)
        path_map[normalize_key(rel_no_ext)].append(rel_with_ext)

    for source_rel in canonical_paths:
        source_file = vault_root / source_rel
        try:
            content = source_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = source_file.read_text(encoding="utf-8", errors="replace")
                message = f"WARNING: decode fallback used for {source_rel}"
                print(message, file=sys.stderr)
                warnings.append(message)
            except Exception as exc:  # noqa: BLE001
                message = f"WARNING: failed to read {source_rel}: {exc}"
                print(message, file=sys.stderr)
                warnings.append(message)
                continue
        except Exception as exc:  # noqa: BLE001
            message = f"WARNING: failed to read {source_rel}: {exc}"
            print(message, file=sys.stderr)
            warnings.append(message)
            continue

        for match in WIKILINK_PATTERN.finditer(content):
            raw_link = match.group(1).strip()
            target = parse_link_target(raw_link)
            if not target:
                continue

            resolved_path, reason = resolve_target(
                target=target,
                source_rel_path=source_rel,
                basename_map=basename_map,
                path_map=path_map,
            )
            if resolved_path:
                graph.add_edge(source_rel, resolved_path)
                continue

            if reason == "empty":
                continue

            unresolved_id = f"UNRESOLVED/{reason.upper()}/{target}"
            unresolved_kind = f"unresolved_{reason}"
            if not graph.has_node(unresolved_id):
                graph.add_node(
                    unresolved_id,
                    kind=unresolved_kind,
                    title=target,
                )
            graph.add_edge(source_rel, unresolved_id)

        graph.nodes[source_rel]["tokens"] = extract_note_tokens(
            title=graph.nodes[source_rel].get("title", Path(source_rel).stem),
            rel_path=source_rel,
            content=content,
        )

    return graph, warnings


def format_note_link(graph: nx.DiGraph, node_id: str) -> str:
    link_ref = graph.nodes[node_id].get("link_ref", strip_md_suffix(node_id))
    return f"[[{link_ref}]]"


def top_nodes_by_metric(metric: dict[str, float], limit: int = 5) -> list[str]:
    ordered = sorted(metric.items(), key=lambda item: item[1], reverse=True)
    return [node for node, _score in ordered[:limit]]


def generate_report(graph: nx.DiGraph, warnings: Sequence[str]) -> str:
    real_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "real"]
    unresolved_missing = [
        n for n, d in graph.nodes(data=True) if d.get("kind") == "unresolved_missing"
    ]
    unresolved_ambiguous = [
        n for n, d in graph.nodes(data=True) if d.get("kind") == "unresolved_ambiguous"
    ]

    sub_graph = graph.subgraph(real_nodes).copy()
    undirected = sub_graph.to_undirected()

    lines: list[str] = []
    lines.append("# Network Diagnostics Report")
    lines.append("")
    lines.append(f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Real notes analyzed: {sub_graph.number_of_nodes()}")
    lines.append(f"- Real note links analyzed: {sub_graph.number_of_edges()}")
    lines.append(f"- Unresolved missing targets: {len(unresolved_missing)}")
    lines.append(f"- Unresolved ambiguous targets: {len(unresolved_ambiguous)}")
    lines.append(f"- Parse warnings: {len(warnings)}")
    lines.append("")

    lines.append("## High-Value Translation Nodes (Top 5 Bridges)")
    if sub_graph.number_of_nodes() > 0:
        betweenness = nx.betweenness_centrality(sub_graph)
        for node_id in top_nodes_by_metric(betweenness, limit=5):
            lines.append(
                f"- {format_note_link(sub_graph, node_id)} "
                f"(score: {betweenness[node_id]:.4f})"
            )
    else:
        lines.append("- None")
    lines.append("")

    isolated = [n for n in sub_graph.nodes if sub_graph.degree(n) == 0]
    orphaned = [
        n
        for n in sub_graph.nodes
        if sub_graph.in_degree(n) == 0 and sub_graph.out_degree(n) > 0
    ]
    dead_ends = [
        n
        for n in sub_graph.nodes
        if sub_graph.out_degree(n) == 0 and sub_graph.in_degree(n) > 0
    ]

    lines.append(f"## Isolated Notes ({len(isolated)} detected)")
    lines.append("_No incoming or outgoing real-note links._")
    for node_id in isolated[:5]:
        lines.append(f"- {format_note_link(sub_graph, node_id)}")
    if not isolated:
        lines.append("- None")
    lines.append("")

    lines.append(f"## Orphaned Notes ({len(orphaned)} detected)")
    lines.append("_Links out, but nothing links back._")
    for node_id in orphaned[:5]:
        lines.append(f"- {format_note_link(sub_graph, node_id)}")
    if not orphaned:
        lines.append("- None")
    lines.append("")

    lines.append(f"## Dead-End Notes ({len(dead_ends)} detected)")
    lines.append("_Linked to by others, but no outgoing real-note links._")
    for node_id in dead_ends[:5]:
        lines.append(f"- {format_note_link(sub_graph, node_id)}")
    if not dead_ends:
        lines.append("- None")
    lines.append("")

    lines.append("## Emergent Themes (Louvain)")
    if undirected.number_of_nodes() > 0:
        try:
            communities = list(community.louvain_communities(undirected, seed=42))
            lines.append(
                "_Heuristically suggested MOCs based on network density "
                f"({len(communities)} clusters)._"
            )
            for index, cluster in enumerate(
                sorted(communities, key=len, reverse=True)[:3],
                start=1,
            ):
                sample_nodes = list(sorted(cluster))[:4]
                sample_links = [format_note_link(sub_graph, node_id) for node_id in sample_nodes]
                if len(sample_links) >= 2:
                    preview = ", ".join(sample_links[:2])
                    if len(sample_links) > 2:
                        preview = f"{preview}, ..."
                elif len(sample_links) == 1:
                    preview = sample_links[0]
                else:
                    preview = "No sample notes"
                theme, evidence = summarize_cluster_theme(sub_graph, list(cluster))
                lines.append(f"- Cluster {index} ({len(cluster)} notes): {preview}")
                lines.append(f"  - Likely theme (2nd pass): {theme}")
                if evidence:
                    lines.append(f"  - Evidence: {evidence}")
        except Exception as exc:  # noqa: BLE001
            warning = f"WARNING: failed to compute communities: {exc}"
            print(warning, file=sys.stderr)
            lines.append(f"- Community detection failed: `{exc}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Structural Overlap Suggestions (High Jaccard)")
    if undirected.number_of_nodes() > 1:
        suggestions: list[tuple[str, str, float]] = []
        for u, v, score in nx.jaccard_coefficient(undirected):
            if score <= 0.10:
                continue
            if sub_graph.has_edge(u, v) or sub_graph.has_edge(v, u):
                continue
            suggestions.append((u, v, score))
        suggestions.sort(key=lambda item: item[2], reverse=True)

        for u, v, score in suggestions[:5]:
            lines.append(
                f"- Consider linking {format_note_link(sub_graph, u)} and "
                f"{format_note_link(sub_graph, v)} (similarity: {score:.2f})"
            )
        if not suggestions:
            lines.append("- None above threshold (0.10)")
    else:
        lines.append("- Not enough notes to score overlaps.")
    lines.append("")

    if warnings:
        lines.append("## Parse Warnings")
        for warning in warnings[:20]:
            lines.append(f"- {warning}")
        if len(warnings) > 20:
            lines.append(f"- ... and {len(warnings) - 20} more")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a vault topology diagnostics report.")
    parser.add_argument("--vault", required=True, help="Absolute path to vault root")
    parser.add_argument("--output", required=True, help="Absolute output markdown file path")
    parser.add_argument(
        "--config",
        help="Optional JSON config with extra_stopwords and section_theme_map",
    )
    args = parser.parse_args()

    vault_root = Path(args.vault).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None

    if not vault_root.exists():
        print(f"ERROR: vault root does not exist: {vault_root}", file=sys.stderr)
        return 1

    if config_path:
        if not config_path.exists():
            print(f"ERROR: diagnostics config does not exist: {config_path}", file=sys.stderr)
            return 1
        try:
            extra_stopwords, section_theme_map = load_diagnostics_config(config_path)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        THEME_STOPWORDS.update(extra_stopwords)
        SECTION_THEME_MAP.update(section_theme_map)

    excluded_rel_paths: set[str] = set()
    try:
        output_rel = output_path.relative_to(vault_root).as_posix()
        excluded_rel_paths.add(output_rel)
    except ValueError:
        pass

    graph, warnings = build_graph(vault_root=vault_root, excluded_rel_paths=excluded_rel_paths)
    report = generate_report(graph=graph, warnings=warnings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote diagnostics report: {output_path}")
    if warnings:
        print(f"Completed with {len(warnings)} parse warning(s).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
