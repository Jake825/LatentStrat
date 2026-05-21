# Scouting Normalization Rules

Normalize before writing rows to LatentStrat's scouting database. Most merge failures come from key mismatches, inconsistent categories, or missing alliance colors.

## Keys

- Team keys must be `frc####`, with no leading zeros unless the official team number contains them.
- Event keys must be official TBA keys such as `2026ilch`.
- Qualification match keys must look like `2026ilch_qm12`.
- Playoff keys must use the real TBA match key, such as `sf1m2` or `f1m3`. Do not invent playoff keys from match labels.
- Alliance colors must be exactly `red` or `blue` for alliance and team-match scouting.

If raw data lacks alliance color, resolve it from TBA match data or reject the row with a clear error. Do not write `unknown` if the row must merge into features.

## Missing Values

- Empty strings, whitespace-only cells, spreadsheet nulls, and NaN values should become `None` for optional strings/numerics.
- Required keys should fail fast with source row context.
- Default values should be used only when the schema defines a real default and the absence is meaningful.

## Booleans

Accept common source values such as `yes`, `y`, `true`, `1`, `no`, `n`, `false`, and `0`. For custom labels such as `worked`, `broken`, `passed`, or `failed`, document the mapping in the column plan.

## Numbers and Ratings

- Parse spreadsheet numbers through float first when sources may encode `254.0`.
- Validate integer fields after conversion.
- Validate rating ranges, especially fields like driver ability from 1 to 5.
- Record units for measurements such as weight, height, dimensions, or cycle time.

## Categories and Text

- Strip whitespace and normalize obvious spelling/case variants.
- Keep meaningful free text as text; do not over-normalize notes.
- For categorical fields, define an allowed vocabulary in the mapping plan.
- Keep original source labels in the audit report when labels are team-specific or ambiguous.

## Multi-Scout Rows

When multiple scouts record the same team-match:

- Decide whether to aggregate, choose an authoritative row, or preserve separate source rows through a schema extension.
- Do not average categorical fields without an explicit rule.
- For numeric performance fields, document whether using mean, median, max, or scouter-weighted values.

## Leakage Timing

Each normalized field should keep a timing classification:

- Static or pre-match: possible model input.
- During-match or post-match: label, auxiliary target, or diagnostic.
- Unknown: exclude from model inputs until clarified.
