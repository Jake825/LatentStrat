# Baselines And Controls

Use baselines and controls to decide whether a model result is meaningful.

## Current Implemented Baselines

LatentStrat currently implements:

- Mean baselines.
- Ridge match-OPR baselines.

Do not claim that a Statbotics baseline exists in the repo unless a future runtime task explicitly adds one.

## External Statbotics Comparison

Statbotics can be used as an external comparison if integrated deliberately:

- Define which endpoint and fields are used.
- Verify whether fields are pre-match or post-match.
- Align by `match_key` and season/event timing.
- Keep the comparison separate from current in-repo baselines unless code adds it.

## Controls

Useful controls include:

- Shuffled targets to detect leakage-sensitive pipelines.
- Null or mean baselines.
- Availability slices for rows with and without scouting enrichment.
- Feature ablations when deciding if a new source is useful.

An improvement that only appears in a leakage-prone split, a tiny slice, or a post-event feature set should not be treated as real production quality.
