---
name: frc-field-geometry
description: Code-friendly geometric and coordinate reference for the FIRST Robotics Competition field. Use when reasoning about field layouts, autonomous paths, spatial scouting analysis, plotting trajectories, pose estimation, AprilTag layouts, scoring locations, zone membership, or converting physical field features into code.
---

# FRC Field Geometry

Use this skill when FRC field space must be treated as geometry, not prose. It complements `$frc-game-manual`: use the manual skill for rules and legality, and this skill for coordinates, pathing, plotting, poses, zones, and spatial analysis.

## Core Directives

1. Avoid vague spatial descriptions. Use explicit points, named field elements, zones, coordinate frames, and assumptions.
2. Declare the coordinate frame. State whether coordinates are field-relative with the always-blue origin or alliance-relative.
3. Use WPILib conventions by default: NWU axes, meters for distance, radians for rotation, and counter-clockwise-positive angles.
4. Convert units explicitly. The game manual often uses inches, while WPILib geometry classes and AprilTag layouts use meters.
5. Treat official FIRST field drawings, official AprilTag layouts, Team Updates, the game manual, and WPILib field-layout docs as sources of truth. Community constants are architecture examples, not precision authority.

## Routing

- Read `references/coordinate-systems.md` for WPILib NWU, field-relative vs alliance-relative coordinates, and transform patterns.
- Read `references/2026-rebuilt.md` for 2026 REBUILT field elements, dimensions, zones, and AprilTag layout notes.
- Read `references/python-patterns.md` for Python dataclass constants, plotting, heatmaps, geometry helpers, and spatial ML feature representation.
- Read `references/source-links.md` for official WPILib and FIRST source links.

## Coordinate Safety Checklist

- Identify the field year and game.
- Identify the source of constants.
- State the origin, axes, units, and angle convention.
- Normalize raw scouting or path coordinates before joining datasets.
- Render plots with equal aspect ratio so distances are not distorted.
- Verify exact tag poses and element dimensions against official layouts before precision use.
