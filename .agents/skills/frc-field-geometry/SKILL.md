---
name: frc-field-geometry
description: Apply FRC field coordinates, units, transforms, zones, paths, poses, AprilTag layouts, and spatial feature conventions. Use only when explicitly invoked for geometric or coordinate reasoning; do not use for prose-only field descriptions, rule legality, ordinary scouting columns, or non-spatial LatentStrat work.
---

# FRC Field Geometry

Treat FRC field space as explicit geometry.

## Core Rules

- Declare the season, field source, coordinate frame, origin, axes, distance units, and angle convention.
- Default to WPILib's always-blue NWU field frame, meters, radians, and counter-clockwise-positive rotation unless the task establishes another convention.
- Convert manual dimensions explicitly and never mix field-relative and alliance-relative values in one column.
- Verify exact dimensions and tag poses against official layouts before precision use.
- Render plots with equal aspect ratio.

## References

- Read `references/coordinate-systems.md` for frames and transforms.
- Read `references/2026-rebuilt.md` only for 2026 REBUILT orientation, then verify precision constants from official sources.
- Read `references/python-patterns.md` for geometry helpers, plots, heatmaps, and spatial feature patterns.
- Read `references/source-links.md` for official FIRST and WPILib sources.
