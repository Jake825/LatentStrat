---
name: frc-time-aware-analysis
description: Use when FRC analytics, scouting, dashboards, simulations, backtests, feature engineering, or model validation depend on what information was known at a specific time. Covers known_as_of boundaries, season and event phases, live vs post-event data, temporal feature eligibility, and leakage prevention beyond EPA-specific rules.
---

# FRC Time-Aware Analysis

Use this skill to prevent temporal leakage and to make FRC analytics honest about what was known at the time of a prediction, decision, or report.

## Core Directives

1. Define `known_as_of` before selecting features, labels, rankings, scouting fields, EPA values, OPR values, awards, or event summaries.
2. Separate pre-match inputs from post-match observations and post-event summaries.
3. Do not use final rankings, completed playoff results, final awards, final OPR, full-event averages, post-match EPA, or current team strength to predict earlier matches.
4. Label live-event, between-event, and post-season outputs differently.
5. Use `$statbotics` for EPA-specific field semantics and `$frc-competition-structure` when event phase or event type changes interpretation.

## Routing

- Read `references/known-as-of.md` before designing feature eligibility, backtests, or predictions.
- Read `references/season-event-phases.md` when classifying offseason, preseason, pre-event, live-event, pre-playoff, post-event, or post-season work.
- Read `references/leakage-patterns.md` before building or reviewing model features, scouting joins, simulations, dashboards, or validation splits.
- Read `references/source-links.md` for official and data-source links.

## Coordinate With Other Skills

- Use `$tba-api` for raw schedules, match results, rankings, awards, and score breakdowns.
- Use `$statbotics` for EPA, prediction fields, and Statbotics API shape.
- Use `$frc-scouting-data-types` and `$latentstrat-scouting-db` when scouting rows may become model inputs.
- Use `$latentstrat-feature-pipeline` when assigning feature eligibility before Parquet construction.
- Use `$latentstrat-model-evaluation` when interpreting backtest metrics, calibration, baselines, controls, and evidence packets.
- Use `$frc-game-manual` when rule changes or Team Updates affect whether a data point existed at a given time.
