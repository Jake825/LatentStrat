---
name: frc-game-manual
description: Guide for understanding, searching, interpreting, and referencing the official FIRST Robotics Competition game manual, Team Updates, and Q&A. Use when answering questions about FRC game rules, robot design legality, field layout, scoring, penalties, tournament structure, manual PDFs, or when extracting official text and diagrams from the manual.
---

# FRC Game Manual Reference

Use this skill to answer FRC rules, scoring, field, strategy, and robot legality questions from official sources. The bundled 2026 assets are official FIRST-hosted PDFs retrieved on May 21, 2026:

- Manual: `assets/manuals/2026GameManual-TU22.pdf`
- Combined Team Updates: `assets/manuals/REBUILT_TeamUpdate-Combined.pdf`

## Core Directives

1. Ground decisions in the official Game Manual, Team Updates, and the Official Q&A. Do not use unofficial forums or community summaries as legality authority.
2. Cite rule numbers, section headings, table names, page numbers, or Team Update numbers. Never make confident legality claims without current manual support.
3. Apply the hierarchy of truth:
   - Rule text is binding.
   - Headlines summarize but do not supersede rule text.
   - Blue boxes provide examples, intent, and guidance; they are not the rule itself.
   - Team Updates officially modify the manual.
   - Official Q&A clarifies application but does not rewrite the manual text.
4. Treat FRC seasons as isolated rule systems. Never assume a rule, field element, score value, or robot constraint carries across years.
5. The bundled 2026 manual is Version `TU22`. For future legality answers, verify the live FIRST Season Materials page and current Official Q&A before finalizing.

## PDF Workflow

- Use the bundled manual first for 2026 REBUILT questions, then check the combined Team Updates PDF for changed rules.
- Use `pdfplumber` or `pypdf` to extract exact text, search rules, and parse tables.
- Render pages visually before answering questions about diagrams, field geometry, AprilTags, bumper dimensions, robot size envelopes, or other spatial details.
- Prefer the `$pdf` skill patterns when PDF extraction or rendering matters.

## Reference Materials

Load only the reference needed for the task:

- `references/manual-conventions.md`: rule categories, defined terms, penalty vocabulary, and manual structure.
- `references/2026-rebuilt.md`: quick-reference guide for the 2026 game, REBUILT. Use it for orientation only, then verify exact wording in the manual.
- `references/pdf-patterns.md`: Python snippets and workflow notes for extracting text, tables, and diagrams from the manual PDFs.
- `references/source-links.md`: official source URLs, retrieval date, and bundled asset mapping.
- Use `$frc-competition-structure` when tournament structure, advancement, district/regional semantics, or multi-division context must be interpreted as data rather than cited as rule text.
- Use `$frc-field-geometry` when field layout questions become coordinate, pathing, pose, plotting, AprilTag, or spatial-analysis tasks rather than rule/legal interpretation.
