---
name: tba-api
description: Work directly with The Blue Alliance API, tbapy, or LatentStrat's TbaProvider to retrieve and normalize teams, events, matches, rankings, awards, alliances, and score breakdowns. Use for endpoint behavior, authentication, keys, caching, payloads, and season-specific TBA fields. Do not use for already-local feature tables, Statbotics, model internals, or generic FRC interpretation.
---

# The Blue Alliance API

Use `tbapy==1.3.2` or `frc.providers.tba_provider.TbaProvider` according to the repository boundary being changed. Use `TBA_API_KEY` for live access.

## Key Contracts

- Team keys use `frc####` without leading zeros.
- Event keys combine year and event code.
- Qualification match keys use `_qm#`; playoff keys include competition level, set, and match numbers.
- Preserve raw payload context for season-specific score breakdowns and tournament structures.
- Handle unplayed and incomplete matches defensively.
- Never assume score-breakdown fields carry across seasons.

## References

- Read `references/tbapy-patterns.md` for retrieval, pagination, normalization, and missing-data patterns.
- Read `references/frc-concepts.md` only when basic FRC terminology is needed to interpret a TBA payload.

Verify current or uncommon endpoint behavior against official TBA API documentation before hard-coding it.
