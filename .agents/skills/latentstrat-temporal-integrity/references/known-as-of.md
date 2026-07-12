# Known-As-Of Boundaries

Every predictive or historical FRC workflow should state what was known at the decision time.

## Define the Boundary

Use a `known_as_of` statement before selecting data:

- `known_as_of`: timestamp, match key, event phase, or season phase.
- `target`: what is being predicted or explained.
- `eligible inputs`: data available before the target.
- `excluded data`: data only known after the target.

Examples:

- Predicting `2026ilch_qm12`: use schedules and completed matches before `qm12`; exclude `qm12` result and later matches.
- Pre-event simulation: use prior events, team history, pre-event scouting, and schedule if released; exclude event results.
- Pre-playoff analysis: use completed qualification matches and alliance selection context if known; exclude playoff results.
- Post-event report: all event results are available, but do not reuse this dataset as if it were pre-event training data.

## Feature Eligibility Rules

Eligible pre-target inputs can include:

- Event metadata and schedule released before the target.
- Team history from earlier events or earlier matches.
- Static pit scouting known before the target.
- Team-event status known before the target, such as inspection or robot functionality.
- Statbotics pre-match prediction fields for the target match, when verified.

Post-target data must be labels, diagnostics, or post-event analysis only:

- Match score and score breakdown for the target match.
- Final event rankings.
- Playoff results.
- Awards announced after matches.
- Full-event averages and final OPR.
- Post-match or current EPA values.
- Team-match scouting observations from the target match.

## Implementation Habit

When writing code, include timing comments or metadata near feature selection. A useful column can still be unsafe if it was not known before the target.
