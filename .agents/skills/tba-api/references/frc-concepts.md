# FRC Domain Concepts

Use this reference when TBA data needs FRC domain interpretation, historical context, or cross-season normalization.

## Event Structure and Matches

- Qualification matches (`qm`) are scheduled matches where teams earn ranking points. Rankings determine playoff selection order.
- Standard FRC matches are red alliance vs. blue alliance, usually three teams per alliance.
- Alliance selection forms fixed playoff alliances after qualification matches at most events.
- Playoff matches use competition levels such as `ef`, `qf`, `sf`, and `f`.
- Autonomous mode occurs first; teleoperated mode follows. Endgame scoring, bonus objectives, and ranking point conditions are game-specific.

## Statistical Metrics

- OPR is a linear algebra estimate of a team's average offensive contribution to alliance score.
- DPR estimates defensive effect, but it is indirect and should be interpreted cautiously.
- CCWM estimates contribution to winning margin.
- These metrics are useful exploratory features, not authoritative scouting truth. Consider EPA or direct scouting data when available.

## District and Regional Systems

- Regional events are independent events. Winners and selected award winners traditionally advance directly to the FIRST Championship.
- Starting in 2025, Regional events may use Championship pool points, exposed in TBA data as fields such as `regional_champs_pool_points`.
- District teams earn district points across district events. Top point-earners advance to a District Championship, then to the FIRST Championship.
- District Championships can be split into divisions that feed a finals field, similar to FIRST Championship division winners feeding Einstein.

## Tournament Structures and Einstein

- FIRST Championship is split into divisions. Division champions advance to Championship Playoffs on the Einstein fields.
- TBA represents the final Championship tournament with event type `CMP_FINALS`.
- Large District Championships may split into divisions, then run a final field. Smaller District Championships may run one standard tournament.
- For 2023 and newer standard events, FRC uses a double-elimination playoff bracket.
- For 2022 and older standard events, FRC used best-of-three single-elimination playoff series.

## Data Idiosyncrasies and Historical Quirks

FRC data completeness and terminology vary by era. Account for the following when writing analysis scripts:

- Score breakdowns and foul-point details exist only for 2015 and later. Scripts analyzing pre-2015 matches must rely on total scores.
- The Chairman's Award was renamed to the FIRST Impact Award in fall 2022. Treat these as the same award in cross-era analysis.
- The Dean's List Award was renamed to the FIRST Leadership Award in fall 2025. Treat these as the same award in cross-era analysis.
- Data from 2010 to present is generally reliable.
- Data from 2006 to 2009 is spotty; rankings, alliances, and match results may be partially missing.
- Pre-2006 data is rare and heavily incomplete. Do not assume fields or endpoints exist.
- For current-year tournament structures and scoring rules, use the official FIRST Game Manual and season materials.

## Year-Specific Score Breakdowns

FRC releases a new game every year. The `score_breakdown` object in TBA match data changes radically by season. Never assume the schema for one year applies to another.

When using score breakdowns:

1. Check the event year before reading game-specific fields.
2. Handle missing fields and `None` payloads gracefully.
3. Preserve raw score breakdowns for downstream feature work and audits.
4. Verify current-season fields against TBA API responses and official FIRST materials.

Examples of recent season concepts:

- 2024 CRESCENDO: Amp notes, Speaker notes, Stage points, harmony, trap.
- 2025 REEFSCAPE: Algae, Coral, Reef rows, Barge scoring.
- 2026 REBUILT: Fuel scoring, Hub elements, Tower climbing, obstacles, and match bonuses.
