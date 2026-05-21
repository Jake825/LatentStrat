# Multi-Division Events

Use this reference when an event may have multiple divisions, a parent event, or a finals field.

## Core Rule

Do not assume there is only one event-level dataset. District Championships and FIRST Championship may have division-level events with their own teams, schedules, rankings, awards, playoffs, and match results, plus an overall championship or finals event that combines division winners.

## FIRST Championship

- TBA represents FIRST Championship divisions as separate events.
- Division events have their own team lists, qualification schedules, rankings, awards, playoff alliances, and match results.
- The Einstein or championship finals event contains the final tournament among division winners.
- A team can have division data and Einstein data in separate event contexts.

## District Championship Divisions

- Large District Championships can split into divisions.
- Division winners may advance to a finals field that mirrors the FIRST Championship structure.
- Smaller District Championships may be a single event without division/finals separation.
- Always inspect event metadata and related events instead of assuming a district championship layout.

## Aggregation Rules

- Division rankings are not overall event rankings.
- Division playoff results are not the same as Einstein or finals-field results.
- Awards, webcasts, schedules, and team lists may be attached to a division event or a parent/finals event depending on the data source and season.
- When merging data, keep `event_key` and any parent or related event identifiers available.

## Common Failure Modes

- Counting Einstein matches as extra division playoff matches.
- Treating division winners as overall championship winners before finals are played.
- Combining all division qualification rankings into a fake global ranking.
- Assigning parent-event awards to every division team without checking award recipient context.
- Joining by year and city instead of explicit event keys.
