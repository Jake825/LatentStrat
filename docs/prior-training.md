---
tags:
  - latentstrat
  - prior-training
  - model-architecture
aliases:
  - "Prior Training"
  - "Day Zero Prior"
  - "V5.6.4 Prior"
related:
  - "[[ghost robot]]"
  - "[[data-sources]]"
  - "[[model-architecture-reference]]"
  - "[[experiment-ledger]]"
---

# Prior Training

Prior training builds the Day Zero team identity table. Day Zero means what the
model knows before the current target season or event has produced new match
evidence.

The current prior design is V5.6.4.

For exact layer dimensions, parameter counts, and loss equations, see the
[Model Architecture Reference](model-architecture-reference.md).

## What The Prior Produces

The production output is a stripped checkpoint containing:

- `embedding_table`, normally shaped `[12501, 16]`.
- Metadata.
- Training history.

The live match model copies that table directly into `Z_base.weight`.

The decoder and prior heads are sacrificial. They exist only to sculpt the
embedding table during training and are discarded before the checkpoint is used
by season training.

## Team Numbers As Dictionary Rows

V5.6 moved LatentStrat from text-at-runtime to a transductive coordinate map:

```text
team_number -> Embedding(max_team_number + 1, latent_dim)
```

For the current default:

- Team `254` maps to row `254`.
- Team `2290` maps to row `2290`.
- Team `0` maps to the learned [ghost robot](ghost%20robot.md).
- Team numbers above the checkpoint cap require rebuilding the prior.

The default cap is `12500`, so the table has rows `0..12500`.

## The Learned Ghost Robot

Team `0` is not padding. It is a learned missing-robot token.

Its narrative is:

```text
This is a null robot. It does not exist on the field. It scores zero points. It has no autonomous routine. It does not play defense.
```

The season model can use row `0` when an alliance slot is empty or explicitly
missing. This keeps the Set Transformer processing a consistent number of
slots without attention-deleting the missing robot.

## Prior Feature Rows

`build-prior-features` writes one row per team number. Each row contains:

- `team_number`
- `team_key`
- `archetype`
- `target_season`
- `narrative`
- `narrative_hash`
- `embedding_model`
- `llm_dim`
- `openai_narrative_vector`
- normalized EPA trajectory columns
- raw cultural target columns

The archetypes are:

- `ghost_token`: team `0`.
- `anchor`: known team active in the target season.
- `ghost`: known historical team not active in the target season.
- `sibling`: unassigned historical number below the future-start region.
- `future`: projected future rookie number.

## Narrative Targets

Narratives are used as one-time semantic targets. They are not live model
inputs.

Known team narratives include:

- Sponsor/name text.
- Location.
- Active era.
- Event wins.
- Championship appearances.
- Award chronology.

Performance facts are limited to years before the target season. For
`target_season=2026`, a 2026 award must not appear in the prior narrative.

Sibling and future rows use generated narratives so the table is complete even
for unassigned numbers.

## OpenAI Target

The narrative is embedded into a 256-dimensional vector. That vector is the
OpenAI semantic target.

V5.6.4 changed the OpenAI loss reduction. Instead of averaging element-wise MSE
across all 256 dimensions, it sums squared error across the vector and averages
across teams:

```text
openai_mse = mean(sum((prediction - target) ^ 2 over embedding dimensions))
```

This gives the text task enough gradient strength to compete with smaller
scalar cultural targets.

## Statbotics Normalized EPA Trajectory

The prior also predicts a four-year normalized EPA trajectory. For target
season `2026`, those source years are:

- `t_minus_4`: 2022
- `t_minus_3`: 2023
- `t_minus_2`: 2024
- `t_minus_1`: 2025

The values are Statbotics normalized EPA values. They are not local z-scores of
raw EPA.

Missing values stay `NaN` and use observed masks:

- `norm_epa_observed_t_minus_4`
- `norm_epa_observed_t_minus_3`
- `norm_epa_observed_t_minus_2`
- `norm_epa_observed_t_minus_1`

The EPA trajectory is trained as one grouped 4D task with one homoscedastic
log variance.

## Cultural Targets

V5.6.2 added raw cultural targets:

- `raw_rookie_year_delta`
- `raw_seasons_played`
- `raw_total_award_count`
- `raw_blue_banner_count`
- `raw_championship_appearance_count`
- `raw_championship_win_count`
- `raw_technical_award_count`

They are deliberately raw and unnormalized. The homoscedastic loss balancer
learns how much each target scale should matter.

Culture helps the vector learn institutional signals such as long-term
experience, technical-award history, and banner history.

## Model Shape

The V5.6.4 prior distiller uses:

```text
team_number
  -> team_embedding
  -> deep shared decoder
  -> openai_head
  -> norm_epa_head
  -> culture_head
```

The forward pass returns:

```text
pred_openai, pred_norm_epa_trajectory, pred_culture, z_latent
```

Only `z_latent` is kept for production.

## Training Command

Recommended current command:

```bash
latentstrat build-prior-features --target-season 2026 \
  --max-team-number 12500 \
  --output data/prior_features_v564_2026.parquet

latentstrat train-prior \
  --features data/prior_features_v564_2026.parquet \
  --output artifacts/prior_v564_latent16 \
  --epochs 1000 \
  --latent-dim 16 \
  --max-team-number 12500 \
  --tensorboard \
  --tensorboard-run-name v564_prior_latent16_2026
```

Inspect with:

```bash
latentstrat inspect-prior \
  --checkpoint artifacts/prior_v564_latent16/checkpoint.pt \
  --features data/prior_features_v564_2026.parquet \
  --output artifacts/prior_v564_latent16/inspection
```

## Inspection Outputs

Prior inspection writes artifacts such as:

- `prior_latent_table.csv`
- `prior_nearest_neighbors.csv`
- `prior_sanity_checks.csv`
- PCA plots.
- Norm histograms.
- Training history copies.

Use these to look for:

- Future or sibling rows collapsing into near duplicates.
- Strange norm distributions.
- Whether EPA and culture targets are visible in PCA colorings.
- Whether row `0` has a distinct learned ghost position.

## Important Lessons From The Prior Experiments

The prior evolved because several ideas failed in useful ways:

- A text autoencoder was too complex for live use.
- Direct team-number lookup made the live model simpler.
- 20,000 rows were unnecessary for the 2026 use case, so the default cap moved
  to 12,500.
- 8D underperformed 16D in the local walk-forward comparison.
- V5.6.2 had an EPA masking issue that gave no useful EPA gradient.
- V5.6.4 changed OpenAI loss reduction to strengthen narrative gradients.

Those results are local project evidence, not proof that the current design is
final.

## Related

- [Ghost robot](ghost%20robot.md): the learned team `0` row used for missing
  robot slots.
- [Data sources](data-sources.md): where OpenAI, Statbotics, TBA history, and
  caches enter the prior feature table.
- [Model architecture reference](model-architecture-reference.md): exact prior
  layer sizes, loss equations, and checkpoint shape.
- [Metrics and artifacts](metrics-and-artifacts.md): how to inspect prior
  latent tables, PCA plots, and nearest-neighbor outputs.
- [Experiment ledger](experiment-ledger.md): prior run evidence from V5.6.1
  through V5.6.4.
