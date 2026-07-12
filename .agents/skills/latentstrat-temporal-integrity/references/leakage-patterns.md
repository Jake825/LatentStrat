# Leakage Patterns in FRC Data

Use this reference before building prediction features, simulations, scouting joins, or validation datasets.

## Common Leakage Sources

- Final event ranking used to predict a qualification match.
- Final OPR, DPR, or CCWM used for a match before the event ended.
- Full-event team averages used to predict early matches.
- Playoff alliance membership used before alliance selection.
- Award outcomes used before awards are announced.
- Target-match score breakdown used as an input.
- Post-match or current EPA used instead of pre-match prediction fields.
- Team-match scouting observations used as pre-match features for the same match.
- Calibration, imputation, or normalization fit on validation or future rows.

## Safe Alternatives

- Use rolling averages computed only from prior matches.
- Use prior-event or prior-season features with explicit season handling.
- Use pre-match Statbotics `pred` fields when verified for the endpoint.
- Use pit scouting or team-event status only when it was known before the target.
- Use post-match observations as labels, diagnostics, auxiliary targets, or reports.

## Review Checklist

For every feature, ask:

- When was it first known?
- Is it known before the row's target match or decision?
- Is it derived from the target match or later matches?
- Does it summarize the whole event or season?
- Was any transformation fit using future data?
- Is the source official, modeled, scouted, or manually entered?

## LatentStrat Guidance

- Treat scouting match observations as post-match by default.
- Treat pit and team-event data as pre-match candidates only if collection timing supports it.
- Keep temporal eligibility separate from database row grain. A field can have the right table and still be unsafe as a model input.
- Prefer explicit feature names or metadata that indicate pre-match, rolling, prior-event, post-match, or summary timing.
