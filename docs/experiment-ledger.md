---
tags:
  - latentstrat
  - experiment-ledger
  - project-history
aliases:
  - "Experiment Ledger"
  - "Run Ledger"
related:
  - "[[archive/project-history]]"
  - "[[changelog]]"
  - "[[evaluation-and-artifacts]]"
  - "[[current-state]]"
---

# Experiment Ledger

This page records local LatentStrat experiments as engineering evidence. It is not a paper, leaderboard, or universal claim. Use it to understand why the current design moved in a particular direction.

For the narrative version of older decisions, see [Archived project history](archive/project-history.md). For the Git and semantic-version timeline, see [Changelog](changelog.md).

## Prior Runs

| Run | Artifact | Input | Epochs | Purpose | Local result | Design lesson |
|---|---|---|---:|---|---|---|
| V5.6 text-only 16D | `artifacts/archive/legacy/prior_v56_latent16/` | migrated prior feature table | `1000` | Replace V5.5 text autoencoder with direct team-number dictionary | Produced `[20001, 16]` checkpoint-style latent table | Direct coordinate maps are simpler than runtime text encoders |
| V5.6.1 16D | `artifacts/prior_v561_latent16/` | `data/prior_features_v561_2026.parquet` | `1000` | Add single EPA target beside OpenAI target | Walk-forward comparison later gave better results than 8D and V5.6.2 | EPA-like quantitative strength signal helps the prior |
| V5.6.1 8D | `artifacts/prior_v561_latent8/` | `data/prior_features_v561_2026.parquet` | `1000` | Test smaller bottleneck | Walk-forward accuracy `0.6594`, worse than V5.6.1 16D `0.7067` | Default returned to 16D |
| V5.6.2 16D | `artifacts/prior_v562_latent16/` | `data/prior_features_v562_2026.parquet` | `1000` | Add raw cultural targets and 4-year normalized EPA trajectory | `norm_epa_mse` stayed `0.0`; EPA masks were effectively empty | Feature-family sanity checks must fail when a target family has no observations |
| V5.6.3 16D | `artifacts/prior_v563_latent16/` | `data/prior_features_v563_2026.parquet` | `1000` | Fix Statbotics normalized EPA extraction and grouped EPA trajectory loss | EPA contributed nonzero training loss; near-duplicate clustering remained for future/sibling rows | Correct EPA path and grouped trajectory loss were necessary but not sufficient |
| V5.6.4 16D | `artifacts/pretraining/prior/prior_v564_latent16/` | `data/features/pretraining/prior/prior_features_v563_2026.parquet` | `1000` | Change OpenAI loss from per-element mean to feature-summed vector distance | OpenAI loss had restored scale; prior inspection still showed dense future/sibling/ghost-token neighbors | Loss reduction is a first-class design choice in multi-task pretraining |

## Prior Inspection Evidence

The V5.6.4 inspection summary is:

```text
artifacts/pretraining/prior/prior_v564_latent16/inspection/v564_vs_v563_prior_summary.csv
```

Important local findings:

- V5.6.4 future rows had high norm concentration: mean norm about `1.91` with standard deviation about `0.015`.
- V5.6.4 future/sibling/ghost-token rows still had near-duplicate cosine neighbors at a high rate: `near_duplicate_neighbor_rate_0_999` about `0.997`.
- Anchor rows were less collapsed than synthetic future/sibling rows.

Interpretation: feature-summed OpenAI loss improved the training path but did not fully solve synthetic-row clustering. Future work should inspect narrative diversity, synthetic-row targets, and whether future/sibling rows need distinct structural targets.

## Season And Walk-Forward Runs

| Run | Artifact | Input prior | Epochs/folds | Purpose | Local result | Design lesson |
|---|---|---|---|---|---|---|
| V5.7 full season | `artifacts/v57_full_season/` | V5.6.1-era prior | `19` epochs | First full V5.7 match-spine and sidecar run | Validation loss ended around `10.97` | Pipeline could train with sidecars, but longer stability was unproven |
| V5.7 long 100 | `artifacts/v57_full_season_long_100/` | V5.6.1-era prior | `100` epochs | Stress-test long heterogeneous training | Validation loss exploded to about `78.12` while train loss kept dropping | Homoscedastic overconfidence and sidecar overfit needed explicit controls |
| V5.7.2 stable 100 | `artifacts/v57_full_season_stable_100/` | V5.6.1-era prior | `100` epochs | Add log-var clamp, best restoration, cosine LR, and AdamW decay | Final validation loss still high, but LR and stability diagnostics were recorded | Long runs need best-validation restoration, not final-epoch saving |
| V5.8 common metrics 100 | `artifacts/v58_full_season_common_metrics_100/` | V5.6.1-era prior | `100` epochs | Add community-facing metrics to feature training | Wrote `feature_common_metrics.csv` | Score MSE, accuracy, Brier, and log loss are easier to compare than total homoscedastic loss |
| V5.8 walk-forward 16D | `artifacts/v58_walk_forward_tensorboard_2026/` | V5.6.1 16D | `10` folds, `5` epochs/fold | First TensorBoard walk-forward reference | Accuracy `0.7067`, Brier `0.1951`, log loss `0.5716` | Temporal validation became the primary model-quality path |
| V5.8 walk-forward 8D | `artifacts/v58_walk_forward_latent8_2026/` | V5.6.1 8D | `10` folds, `5` epochs/fold | Test 8D prior in season validation | Accuracy `0.6594`, score MSE worse than 16D | 8D was too tight for this local setup |
| V5.8 walk-forward V5.6.2 | `artifacts/v58_walk_forward_v562_latent16_2026/` | V5.6.2 16D | `10` folds, `5` epochs/fold | Test cultural-prior attempt | Accuracy `0.6604`, worse than V5.6.1 16D | EPA-mask bug likely hurt the prior despite richer targets |
| V5.8 walk-forward V5.6.4 50ep | `artifacts/v58_walk_forward_v564_latent16_50ep_2026/` | V5.6.4 16D | `10` folds, `50` epochs/fold | Long folded validation with latest prior | Best local score MSE and accuracy, but weaker Brier/log loss | Calibration is now the main follow-up |
| Frozen V5.8 replay for V6 comparison | `artifacts/baselines/v5.8/walk-forward/` | tagged `v5.8-baseline`, V5.6.4 16D | `10` folds, `50` epochs/fold | Archive paired prediction export and exact hashes before V6-Lite | Phase score MSE `8829.77`, total score MSE `8971.60`, accuracy `0.7164`, Brier `0.2090`, log loss `0.7227` | Use this manifest-backed replay as the V6 promotion boundary |
| 2026 static robot-state reference | `artifacts/reference/2026-static-z-base/` | V5.6.4 prior selected over random | `1` seed; weeks 6/8/10 selection and refit | Establish a deliberately simple static-state architecture reference | Completed; full-match score bias was about `-58`, `-69`, and `-67` points by test week | Static state is a useful control but does not track the later-season score distribution |
| Initial static reliability smoke | `artifacts/smoke/2026-static-reliability/` | V5.6.4 prior | `18` two-epoch training phases | Exercise the reliability workflow end to end | Training completed; final Parquet writing rejected mixed integer/string seed identifiers | Artifact schema failures must remain distinct from model-quality failures |
| 2026 static reliability smoke | `artifacts/smoke/2026-static-reliability-smoke/` | V5.6.4 prior | `1` seed, week 6, `2` epochs | Exercise the multi-seed runner, artifacts, TensorBoard, and Workbench without making an architecture claim | Completed as a non-scientific smoke; Statbotics was excluded from the postmortem while its API was unreliable | The observability and review contract works end to end |
| 2026 static reliability development grid | `artifacts/reference/2026-static-reliability-full/` | V5.6.4 prior | `3` architectures x `3` clipping thresholds x `40` epochs | Freeze one optimizer threshold and duration before multi-seed test training | Rejected on week 4: clip `5` passed, but no common epoch from 15-40 met all score-differential, Brier, and log-loss tolerances; no test folds ran | Score and probability optima occur at materially different durations, so the project is paused before another architecture claim |
| Fixed-100 Championship diagnostic smoke | `artifacts/smoke/2026-static-championship-core-browser/` | V5.6.4 prior | `12` leaves x `2` epochs | Validate exact-key splits, fixed losses, resume, artifacts, TensorBoard, and Workbench before the formal campaign | Completed as non-scientific smoke; all labels remained finite and expected clean-row counts matched | The formal campaign can proceed only after the implementation commit and timing gate |
| Fixed-100 Championship diagnostic | `artifacts/reference/2026-static-championship-core/` | V5.6.4 prior | `6` development + `6` refit leaves x `100` epochs, seed `2026` | Test direct totals and teammate/opponent interactions under one fixed static-state horizon | Completed in `159.65` minutes; official-total additive reduced score RMSE `2.76%` and bias from `-60.9` to `-56.6`, while teammate interaction was harmful in both supervision modes and opponent interaction remained uncertain versus teammate-only | More epochs reduced training loss but forecast optima occurred much earlier; next isolate time/event context using additive official-total as the static control |

The exact run inventory, TensorBoard scalar export, score-bias slices, and interpretation boundaries are recorded in the [2026 static training postmortem](2026-static-training-postmortem.md).
The complete fixed-horizon evidence and decision rules are recorded in the [Championship diagnostic report](2026-static-championship-core-report.md).

## Current Walk-Forward Comparison

| Run | Phase score MSE | Total score MSE | Accuracy | Brier | Log loss | Matches |
|---|---:|---:|---:|---:|---:|---:|
| V5.6.1 16D, 5 epochs | `9869.29` | `10021.99` | `0.7067` | `0.1951` | `0.5716` | `18164` |
| V5.6.1 8D, 5 epochs | `12601.13` | `12792.16` | `0.6594` | `0.2190` | `0.6269` | `18164` |
| V5.6.2 16D, 5 epochs | `10760.56` | `10950.33` | `0.6604` | `0.2091` | `0.6037` | `18164` |
| V5.6.4 16D, 50 epochs | `8953.20` | `9055.87` | `0.7192` | `0.2025` | `0.6796` | `18164` |

Reading:

- V5.6.4 plus longer folded training gave the best local score MSE and match accuracy.
- The same run did not give the best Brier score or log loss.
- Future improvements should treat calibration as a first-class acceptance criterion, not a secondary chart.

## Historical Match-Breakdown Pretraining

| Artifact | Input | Purpose | Status |
|---|---|---|---|
| `artifacts/pretraining/match-breakdown/v1_2015_2026/` | `data/pretraining/match-breakdown/corpus.sqlite` | Train season-specific raw match-breakdown encoders and decoders around one shared 16D bottleneck | Completed locally: `328076` alliance embeddings, combined holdout effective rank `13.26`, eval loss `17.29 -> 0.52`, all-data loss `16.50 -> 0.51` |
| `artifacts/pretraining/match-breakdown/v2_2015_2026/` | same durable corpus; explicit `configs/pretraining/match-breakdown-v2.yaml` | Add latent denoising, winner orientation, and audit-first decoder score consistency without replacing V1 | Implemented as an opt-in ablation; full local training and V1 comparison pending |

This V1 artifact is intentionally offline-only. It exports one latent per played alliance, keeps `2021` as an audited schema exclusion, uses normal official TBA event types `0..5` by default, and writes no normalization file. Its bundle records `promotion_eligible=false` because all-years representation training is not a leakage-safe 2026 walk-forward result.

The local corpus synchronized `1965` eligible events and `164211` matches. `164054` matches had posted scores and alliance breakdowns; `173` incomplete or null-breakdown rows remain in SQLite for refreshes. Per-season vector widths range from `24` in 2015 to `222` in 2023. The union-schema audit records presence only and confirms `training_uses_union_padding=false`.

### Match-Breakdown Inspection Report

The standalone local inspection report is:

```text
artifacts/pretraining/match-breakdown/v1_2015_2026/inspection/
```

It was generated from `328076` production alliance embeddings and `32818` reconstructed
eval-model holdout rows. The report samples `11000` production and `5500` holdout rows for t-SNE,
exports `400` extreme network seeds with `800` different-season neighbor edges, and leaves the
trained `bundle.json` hash unchanged.

Local visual review:

- Production PCA and t-SNE retain strong season-specific regions. The shared bottleneck has not
  erased game identity, so season overlap should not be treated as established.
- Score and auto percentiles show visible gradients. Endgame extremes occupy narrower regions,
  while foul extremes are comparatively diffuse.
- The first three production PCA components explain about `51.80%` of variance; the first eight
  explain about `87.31%`.
- The cross-season network finds different-season analog edges throughout the PCA frame, but this
  is qualitative evidence only.
- Safe aggregate endgame fields are unavailable for `2015`, `2016`, `2017`, and `2023`; those
  seasons remain `NaN` in endgame coloring rather than using invented proxies.

These plots are interpretation aids, not promotion evidence. They help locate follow-up questions
before per-alliance score attachment and leakage-safe walk-forward evaluation.

### Structured V2 Rule Audit

V2 rule YAML files treat score equations as audited hypotheses. Required exact equations abort on
drift. Candidate equations only contribute loss after passing raw-row audit. Known false hypotheses
stay documented as disabled. Startup audit against the synchronized local corpus currently resolves
the intended families for both eval-training and all-data scopes, including corrected 2023 link
handling and corrected 2026 hub subtotal semantics.

The next milestone is per-alliance Set Transformer attachment:

```text
z_red  -> ScoreEmbeddingPredictor -> frozen_red_16d
z_blue -> ScoreEmbeddingPredictor -> frozen_blue_16d
```

## V6.1 Stabilization

V6.1 migrated generated local outputs into the pretraining, season, walk-forward, experimental, and
archive directory families. The migration verified SHA-256 hashes after every move and wrote the
ignored `artifacts/archive/index.json` record. A post-migration V1 inspection rerun resolved the
historical bundle's old feature-table path through the compatibility reader without mutating V1.

The full all-years V2 retrain remains the next match-breakdown experiment. It is intentionally not a
V6.1 completion requirement.

## Open Follow-Ups

- Add in-repo Statbotics EPA and pRidge baselines only after explicitly designing leakage-safe source timing.
- Investigate calibration methods for walk-forward predictions.
- Review why synthetic future/sibling rows remain near-duplicate in prior space.
- Add clearer slice reports by week, event type, team archetype, and data availability.
