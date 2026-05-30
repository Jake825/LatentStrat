---
name: statbotics
description: Use when working with the Statbotics FRC analytics platform, the statbotics Python package, Expected Points Added (EPA), match predictions, event simulations, bulk historical FRC analytics, or leakage-safe prediction features. Covers querying Statbotics data, interpreting EPA, joining Statbotics with TBA data, and navigating Statbotics codebase concepts.
---

# Statbotics and EPA Analytics

## Overview

Use this skill for Statbotics workflows in FIRST Robotics Competition analytics, especially when building prediction features, comparing team strength, simulating events, or using EPA alongside The Blue Alliance data.

LatentStrat already depends on `statbotics==3.0.0` and provides `frc.providers.statbotics_provider.StatboticsProvider`, a SQLite-backed wrapper around `statbotics.Statbotics` using `data/cache/statbotics.sqlite` by default.

## Core Directives

1. Treat EPA as a model, not ground truth. EPA estimates scoring contribution and is useful for prediction, but it is not a direct scouting measurement.
2. Prevent data leakage. Do not use post-match, end-of-event, current, or end-of-season EPA to predict earlier matches.
3. Prefer time-appropriate fields. In `statbotics==3.0.0`, match `pred` fields represent pre-match alliance expectations; `result` fields are post-match outcomes.
4. Inspect returned keys before hard-coding field paths. Statbotics endpoint shapes and EPA field names vary by endpoint and API version.
5. Use bulk list endpoints for ML. Avoid thousands of narrow API calls when `get_matches`, `get_team_matches`, `get_team_events`, or `get_team_years` can fetch pages of records.

## Routing

- Read [references/epa-model.md](references/epa-model.md) before designing predictive features, simulations, backtests, or evaluation datasets.
- Read [references/data-access.md](references/data-access.md) before writing Python code with `statbotics`, `StatboticsProvider`, Pandas, pagination, or TBA joins.
- Read [references/codebase-structure.md](references/codebase-structure.md) when asked to understand, adapt, or contribute to the Statbotics platform itself.
- Use `$frc-time-aware-analysis` for non-EPA temporal leakage, event phase boundaries, live-event dashboards, and `known_as_of` feature eligibility decisions.
- Use `$latentstrat-feature-pipeline` when Statbotics data is being joined into LatentStrat Pandas/PyArrow feature tables.

## TBA vs. Statbotics

- Use TBA for raw event truth: schedules, official scores, alliances, awards, rankings, and game-specific score breakdowns.
- Use Statbotics for derived analytics: EPA, pre-match predictions, expected scores, win probabilities, and team strength summaries.
- Join datasets on shared keys such as `event_key`/`event` and `match_key`/`match`; normalize team numbers to TBA-style `frc####` keys when needed.
- Use `$frc-competition-structure` when event type or multi-division structure affects simulation or feature interpretation.
