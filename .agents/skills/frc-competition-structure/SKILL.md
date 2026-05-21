---
name: frc-competition-structure
description: Use when interpreting FIRST Robotics Competition event structure, event type semantics, regional vs district systems, district championships, FIRST Championship divisions, Einstein or finals fields, offseason or preseason events, advancement rules, rankings, awards, and the correct grain for FRC analytics.
---

# FRC Competition Structure

Use this skill when event context changes how data should be interpreted. TBA remains the primary data access layer, but this skill explains what event and tournament structures mean for analysis.

## Core Directives

1. Identify the event system before interpreting rankings, awards, advancement, playoffs, or season summaries.
2. Choose the correct grain: team-season, team-event, team-division, match, alliance, district, regional pool, championship division, or Einstein/finals field.
3. Do not treat regional events, district qualifiers, district championships, FIRST Championship divisions, Einstein, preseason events, and offseason events as interchangeable.
4. For current-year advancement or tournament rules, verify against the official FIRST Game Manual, season materials, and Team Updates.
5. Use `$tba-api` for event metadata, teams, matches, rankings, awards, district points, and official keys.

## Routing

- Read `references/event-systems.md` for regional, district, district championship, and advancement semantics.
- Read `references/multi-division-events.md` for FIRST Championship divisions, Einstein, district championship divisions, and parent/finals field distinctions.
- Read `references/offseason-events.md` before using preseason or offseason matches in scouting, modeling, rankings, or season summaries.
- Read `references/source-links.md` for official FIRST and TBA source URLs.

## Coordinate With Other Skills

- Use `$tba-api` to fetch event, match, ranking, award, alliance, and district data.
- Use `$frc-game-manual` when a question needs binding rule or advancement text.
- Use `$frc-time-aware-analysis` when the answer depends on when the information was known.
- Use `$statbotics` when event structure affects EPA, simulations, or predictive features.
