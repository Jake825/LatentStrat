---
tags:
  - latentstrat
  - v6-lite
  - world-model
aliases:
  - "LatentStrat V6-Lite"
  - "Frozen FRC Target Embeddings"
related:
  - "[[current-state]]"
  - "[[season-training]]"
  - "[[metrics-and-artifacts]]"
  - "[[schemas-and-artifacts-reference]]"
---

# LatentStrat V6-Lite: Frozen FRC Target Embeddings

V6-Lite keeps the supervised Set Transformer and adds JEPA-inspired auxiliary tasks that predict frozen, offline target embeddings. V5.8 is preserved as the historical comparison boundary by the annotated `v5.8-baseline` Git tag and the tracked `baselines/v5.8-baseline.json` hash manifest.

The first V6-Lite V1 milestone is deliberately offline-only: build and validate a historical match-breakdown representation artifact before attaching score auxiliaries to season training.
An explicit opt-in V2 ablation adds denoising, winner orientation, and audited score-equation
consistency while preserving the V1 artifact as the inspection baseline.

## Central Latent

The team-event latent remains the source of truth:

```text
z_team_event = Z_base[team] + Z_event[team, event]
```

Award, ranking, and pick predictor scaffolds remain available for later phased integration:

```text
z_team_event -> AwardPrototypePredictor -> predicted_e_award
z_team_event -> RankOutcomePredictor -> predicted_z_rank_event
(z_team_event_captain, z_team_event_candidate) -> SelectionEmbeddingPredictor -> predicted_z_pick
```

Score attachment is deferred. Its future runtime contract is per alliance:

```text
z_red  -> ScoreEmbeddingPredictor -> frozen_red_16d
z_blue -> ScoreEmbeddingPredictor -> frozen_blue_16d
```

## Phase 1: Historical Score Archetypes

The V1 encoder ingests TBA match breakdowns from `2015..2026`. The current TBA API v3 OpenAPI spec is `3.15.0` and defines breakdown schemas for `2015..2020` and `2022..2026`, but not `2021`.

Raw payloads are preserved in:

```text
data/pretraining/match-breakdown/corpus.sqlite
```

The transport cache at `data/cache/tba.sqlite` is separate and disposable. The durable corpus records ingestion runs, raw endpoint responses, endpoint ETags, event objects, match objects, payload hashes, and refresh metadata.

The default event filter is TBA event types `0..5`. FOC `6` and remote `7` are included only when explicitly requested. Unplayed matches and null breakdowns remain in the corpus for later refreshes but do not become training rows.

### Season-Specific Schema

Each eligible season has its own typed vector schema:

```text
concat(values_y, masks_y)
  -> E_y: 2N_y -> 128 -> 64
  -> G:   64 -> 32 -> 16
  -> D_y: 16 -> 64 -> 128 -> N_y
```

`E_y` and `D_y` are season-specific. `G` is the shared bottleneck. The exported match-archetype embedding is the 16D output of `G`.

The model never trains on a padded union schema. A union-schema presence audit exists only for diagnostics.

### Raw Values And Masks

The offline encoder does not write or consume normalization files. It uses raw score-breakdown values and observed masks:

```text
missing field:       value=0, mask=0
observed zero field: value=0, mask=1
```

Numeric leaves use `SmoothL1`; boolean and categorical leaves use BCE-with-logits. Categorical fields use stable lexicographically sorted one-hot categories and share one observed state across their category columns. Missingness is not a category.

Losses average active fields within these deterministic groups:

```text
total_score
fouls
auto
teleop
endgame
bonus
misc
```

`misc` fields are recorded in the schema audit so grouping gaps remain visible.

### Training And Artifacts

Run:

```powershell
latentstrat sync-match-breakdowns `
  --start-season 2015 `
  --end-season 2026

latentstrat train-match-breakdown-encoder `
  --start-season 2015 `
  --end-season 2026 `
  --epochs 50 `
  --seasons-per-step 4 `
  --rows-per-season 64 `
  --learning-rate 0.001 `
  --seed 2026
```

Training uses AdamW with `weight_decay=0.01`, mixed-season batches, deterministic seeded queues, and gradient clipping at `1.0`. Evaluation holds out a seeded 10 percent of match keys per season so red and blue rows from one match stay together.

The artifact directory is:

```text
artifacts/pretraining/match-breakdown/v1_2015_2026/
  eval_model.pt
  model.pt
  embeddings.parquet
  schema.json
  union_schema_audit.json
  bundle.json
  validation_report.json
  training_history.csv
```

`eval_model.pt` supplies holdout diagnostics. `model.pt` is retrained on all eligible rows and supplies exported embeddings. `bundle.json` records that distinction and sets `promotion_eligible=false`: this all-years representation artifact is not leakage-safe walk-forward evidence.

### Structured V2 Ablation

V2 keeps the same season-specific architecture and raw-value contract. It is selected explicitly:

```powershell
latentstrat train-match-breakdown-encoder `
  --config configs/pretraining/match-breakdown-v2.yaml
```

The objective adds:

```text
L_total =
  L_reconstruction
  + 0.10 * MSE(S_corrupt, stopgrad(S_full))
  + 0.02 * MarginRankingLoss(q(S_red), q(S_blue))
  + 0.05 * L_score_equation_residual
```

Denoising begins at epoch `5`, winner ranking at epoch `10`, and score consistency at epoch `15`.
Corruption hides 10 percent of observed source fields and masks categorical one-hot groups
atomically. The quality head is diagnostic-only; exported targets remain the 16D `S` vectors.

Score-equation YAML files live beside the match-breakdown package. Startup audits numeric decoder
fields on raw TBA rows before training. Required equations abort on drift. Candidate equations only
enter the loss after passing their configured threshold. Disabled hypotheses remain documented in
`rule_audit.json` but never contribute gradients.

V2 writes a separate artifact:

```text
artifacts/pretraining/match-breakdown/v2_2015_2026/
```

It remains `promotion_eligible=false` until its reconstruction, effective rank, neighbors, quality
orientation, and consistency residuals have been compared against V1.

### Static Inspection Report

Generate a separate post-training visualization report without retraining or mutating
`bundle.json`:

```powershell
latentstrat inspect-match-breakdown-encoder `
  --artifact-dir artifacts/pretraining/match-breakdown/v1_2015_2026 `
  --output artifacts/pretraining/match-breakdown/v1_2015_2026/inspection `
  --seed 2026 `
  --tsne-max-rows 11000 `
  --network-node-limit 400 `
  --neighbors-per-node 2
```

The report writes deterministic PCA and t-SNE multi-view plots, PCA variance, parallel-coordinate
archetype profiles, cross-season nearest-neighbor tables and plots, component-source metadata, and
`inspection_manifest.json`. Component colors use within-season percentiles and one prioritized safe
aggregate field per component. Seasons without a safe aggregate remain `NaN`.

Production plots use `embeddings.parquet` from the all-data retrain. Holdout plots recreate the
match-grouped validation split from `eval_model.pt`. The holdout vectors are orthogonally aligned
for display in the production PCA frame because independently trained autoencoders may rotate an
equivalent latent geometry.

These plots are interpretation aids. They can expose score gradients, calendar-year separation, and
cross-season analogies, but they do not prove successful transfer or replace leakage-safe promotion
metrics.

For an explicit V2 inspection, add:

```powershell
latentstrat inspect-match-breakdown-encoder `
  --artifact-dir artifacts/pretraining/match-breakdown/v2_2015_2026 `
  --baseline-artifact-dir artifacts/pretraining/match-breakdown/v1_2015_2026
```

This writes keyed nearest-neighbor and score-sorting drift tables. They are review diagnostics, not
automatic promotion evidence.

## Later Target Spaces

### Phase 2: Award Semantic Prototypes

`data/awards/award_descriptions.yaml` contains official FIRST descriptions and checked-in paraphrases for semantic team awards. Variants are embedded with `text-embedding-3-small` at 256D, normalized, averaged, and normalized again. PCA is intentionally not used.

Only positive award rows contribute cosine-distance loss. Missing awards are unlabeled, not hard negatives. Individual awards and robot-performance outcomes such as Winner, Finalist, and Wildcard are excluded.

### Phase 3: Ranking Outcomes

The 16D ranking target encoder uses completed team-event rank, record, ranking-stat, and playoff advancement outcomes. Fold-local builders exclude future rows and reject rows marked as belonging to incomplete events.

### Phase 4: Pick Desirability

The offline contrastive builder creates ranked-unselected opportunity snapshots from captain, candidate, event, and available-pool context. The runtime predictor receives only captain and candidate latents, never a variable-sized pool.

## Configuration

Tracked configs reserve score embeddings but keep them disabled until per-alliance runtime integration exists:

```yaml
world_model:
  enabled: true
  baseline_ref: v5.8-baseline
  score_embedding: {enabled: false, width: 16}
  award_embedding: {enabled: false, width: 256}
  rank_embedding: {enabled: false, width: 16}
  pick_embedding: {enabled: false, width: 16}
```

Attempting to enable score embeddings currently fails with a clear integration-pending message. `world_model.enabled=false` remains a valid supervised-only V6 ablation.

## Promotion Gates

After score attachment is implemented, promote phases sequentially. Compare paired fold-match predictions against both the prior accepted phase and archived V5.8 with deterministic bootstrap resampling within folds using seed `2026`.

| Metric | Maximum regression |
|---|---:|
| Brier score | `+0.002` |
| Log loss | `+0.01` |
| Total-score MSE | `+1%` |

Each phase must also improve its own latent diagnostics, retrieval, or probing target.

## Excluded Designs

V6-Lite does not add EMA updates, VICReg, end-to-end target unfreezing, graph Laplacian losses, award autoencoders, union-schema score training, offline score normalizers, or full draft graph encoders.
