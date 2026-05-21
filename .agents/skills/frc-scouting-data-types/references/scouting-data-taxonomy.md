# FRC Scouting Data Taxonomy

Use this taxonomy to classify scouting fields before writing adapters or schema changes.

## Pit or Team Scouting

Pit scouting describes the robot or team independent of a specific match.

Examples:

- Drive base.
- Robot weight and frame size.
- Programming language.
- Mechanism notes.

LatentStrat target: `TeamScouting`.

Timing: usually static or pre-match.

## Event Context

Event context describes venue or field conditions that affect all teams.

Examples:

- Carpet condition.
- Venue notes.
- Field-specific quirks.

LatentStrat target: `EventScouting`.

Timing: usually pre-match or event-level context.

## Match Context

Match context describes the match environment, independent of a specific alliance or robot.

Examples:

- Field fault.
- Audience delay.
- Referee strictness.

LatentStrat target: `MatchScouting`.

Timing: often post-match unless known before the match.

## Team Event Status

Team-event scouting describes a team's state during one event.

Examples:

- Passed inspection.
- Robot functional.
- Major breakdown notes.

LatentStrat target: `TeamEventScouting`.

Timing: can be pre-match if known before a match; otherwise treat as event-state data with explicit timing.

## Alliance Match Strategy

Alliance scouting describes one alliance in one match.

Examples:

- Coordinated autonomous run.
- Coopertition agreement.
- Strategic meltdown.

LatentStrat target: `MatchAllianceScouting`.

Timing: often during or post-match.

## Team Match Scouting

Team-match scouting describes one team in one match.

Examples:

- Auto pieces scored.
- Teleop pieces scored.
- Endgame status.
- Driver ability rating.
- Played defense.
- Defender team.
- Scouter name.

LatentStrat target: `TeamMatchScouting`.

Timing: dynamic match observation. Store it safely, but do not use as default pre-match input.

## Classification Rules

- If a value can change from match to match for the same team, it is not pit data.
- If a value describes the red or blue alliance as a whole, use alliance-match grain.
- If a value describes a team's event readiness, use team-event grain.
- If a value is a measured match performance, treat it as post-match unless the source proves it was known before the match.
- If a field does not fit any current model and has analytical value, propose a schema extension.
