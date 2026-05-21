# Event Systems and Advancement Semantics

Use this reference to decide what an FRC event means before aggregating rankings, awards, or results.

## Regional Events

- Regional events are generally open to regional teams and can include teams from many areas.
- Regional teams may qualify for FIRST Championship directly from an event or through the Regional Pool system, depending on the current season rules.
- Starting in 2025, some TBA data may expose regional advancement point fields such as `regional_champs_pool_points`.
- A regional event ranking is local to that event. It is not the same as a season-wide Regional Pool ranking.

## District Events

- District qualifiers are part of a district season path.
- District teams usually play multiple smaller events and earn district points toward district championship qualification.
- District points are not equivalent to event ranking points from match play.
- A district event ranking describes qualification performance at that event, while district points describe advancement position across the district system.

## District Championships

- District Championships are later-stage district events.
- Small District Championships may run as one standard tournament.
- Large District Championships may be split into divisions that feed a finals field.
- Do not mix division rankings, division playoffs, and finals-field results without labeling the grain.

## FIRST Championship

- FIRST Championship is represented through division-level tournaments plus an Einstein or championship finals field.
- Division winners advance to the championship playoff field.
- Division rankings and awards are not overall championship rankings unless the source explicitly says so.
- Einstein results are not division playoff results.

## Analysis Grain Checklist

Before writing code or analysis, state the grain explicitly:

- `team-season`: one team across one season.
- `team-event`: one team at one event.
- `team-division`: one team within a championship or DCMP division.
- `match`: one official match.
- `alliance-match`: red or blue alliance in one match.
- `district`: a district-level season ranking or advancement context.
- `regional-pool`: regional advancement pool context.
- `championship-division`: one FIRST Championship division.
- `championship-finals`: Einstein or finals field context.

## Practical Guardrails

- Do not compare district points and regional advancement points as if they are the same metric.
- Do not treat event ranking points, district points, and championship qualification status as interchangeable.
- Do not aggregate multi-division events without deciding whether the output is per-division, per-finals-field, or combined for a clearly labeled purpose.
- Preserve event metadata and raw advancement fields when building feature tables.
