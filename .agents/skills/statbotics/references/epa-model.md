# EPA Model Concepts

Expected Points Added (EPA) is Statbotics' core FRC strength metric. It is inspired by Elo-style updating, but it operates in score space instead of abstract rating or pure win-probability space.

## What EPA Means

- EPA is measured in FRC point units. A team EPA around 15 means the model expects that team to contribute about 15 points to an alliance score in that season's scoring environment.
- EPA is additive by design. Alliance expectations are modeled from the sum of team contributions, then converted into expected scores, win probabilities, and ranking point probabilities.
- EPA can include season-specific components such as auto, teleop, endgame, and ranking point components when Statbotics has enough data for that game.
- EPA is a model estimate. It can lag real robot improvements, overreact to small samples, or miss scouting context such as defense, failures, schedule strength, and strategic role.

## Season Boundaries

FRC games change every season, so raw point scales are not comparable across years without normalization.

- A new season applies mean reversion from previous-year performance toward the global mean.
- Early-season EPA can be heavily influenced by the previous robot and should be treated cautiously.
- Do not mix raw EPA values from different seasons in one model without normalization, season indicators, or year-specific handling.
- Component EPA names and meanings are game-specific. Never assume a 2024 component maps cleanly to a 2025 or 2026 component.

## Preventing Data Leakage

Data leakage is the main failure mode when using EPA in ML or backtesting.

Do not use any value that was only known after the match being predicted:

- Post-match team-match EPA, such as nested `epa["post"]`.
- End-of-event or current team-event EPA when predicting earlier event matches.
- End-of-season team-year EPA or current team EPA when predicting historical matches.
- Convenience fields from older examples such as `epa_end`, `epa_pre`, or `red_epa_pre` unless you have verified what the endpoint returns and that the value is time-appropriate for the prediction target.

For `statbotics==3.0.0`, `get_matches(...)` returns match-level dictionaries with nested `pred` and `result` sections. Use `pred` for pre-match alliance expectations and `result` for actual outcomes. Confirm field names with a small sample before building production features.

## Practical Modeling Rules

- Use Statbotics `pred` fields for pre-match alliance expected scores, win probability, and ranking point probabilities.
- Use TBA or Statbotics `result` fields only as labels or evaluation truth.
- When training LatentStrat models, keep temporal splits honest. Do not let validation rows influence target normalization, EPA feature fitting, or any learned calibration.
- Document whether a feature is pre-match, post-match, event summary, or season summary.
- If a task needs robot capability rather than realized output, combine EPA with scouting data instead of treating EPA as a scouting substitute.
