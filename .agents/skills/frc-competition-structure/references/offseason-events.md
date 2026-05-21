# Preseason and Offseason Events

Use this reference before treating non-official events as equivalent to official in-season matches.

## Preseason and Week 0 Style Events

- Preseason events can be useful for early scouting and robot readiness signals.
- Rules, field elements, inspection status, timing, match schedules, and team availability may differ from official events.
- Treat preseason observations as contextual priors, not official season results.

## Offseason Events

Offseason events are useful but different from official in-season events.

- They may use modified rules, modified field elements, unusual team lists, nonstandard playoff formats, replayed matches, incomplete data, or relaxed inspection constraints.
- They may include practice robots, drive-team experiments, repaired robots, graduating students, or strategic testing.
- They can be valuable for scouting and historical context.
- Do not automatically treat them as equivalent to official season matches for EPA modeling, qualification analysis, district points, regional advancement, or season-level predictions.

## Data Use Guidance

- Label preseason/offseason rows explicitly before model training.
- Keep them out of official-season standings and advancement calculations unless the task explicitly asks for an exhibition or all-events view.
- For predictive models, include them only with an explicit feature or weighting policy.
- For scouting, preserve notes but avoid assuming they represent inspected, official-event robot capability.

## Recommended Questions

- Was this event official, preseason, offseason, or practice-only?
- Were official rules and field elements used?
- Were matches complete and reliably scored?
- Were team rosters and robot configurations representative of the official season?
- Should the task output include these matches, exclude them, or report them separately?
