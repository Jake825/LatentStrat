# Roadmap

LatentStrat's north star is an FRC prediction and decision system built around one unconstrained multidimensional robot state. The state should capture individual capability, teammate compatibility, opponent matchup effects, sparse scouting evidence, and change through the competition season. The system should ultimately support calibrated match forecasts, adaptive alliance-selection recommendations, and tournament simulation.

The conceptual entity hierarchy is:

```text
program identity -> season robot -> event state -> match realization
```

These levels describe different time scales and evidence boundaries. They do not require semantic partitions inside the learned robot vector.

## Representation Constraints

- Keep one unconstrained robot representation; do not assign dimensions to predefined concepts.
- Do not add Matryoshka Representation Learning or other coordinate-ordering objectives.
- Treat PCA, clustering, cosine neighbors, trajectories, attention, and zero-out reports as post-training diagnostics.
- Require predictive evidence before interpreting latent geometry as capability, archetype, synergy, or matchup behavior.
- Preserve permutation-tolerant alliances and explicit partner/opponent interaction modeling.

## Milestones

The implemented first reference milestone is documented in the
[2026 static robot-state reference study](reference-2026-static-study.md). Its smoke mode validates
the code and dashboards; only a complete weeks 6/8/10 matrix can classify interaction evidence,
and even that one-season, one-seed study remains non-promotion evidence. The milestones below are
the roadmap after that fixed reference point.

The immediate follow-up is the
[2026 static architecture reliability study](reference-2026-static-reliability.md). It freezes
optimization on development week 4, repeats the architecture comparison across three seeds, and
uses hierarchical seed/event uncertainty before any event-state architecture is introduced.

### 1. Honest Temporal Evaluation

Replace validation-selected walk-forward reporting with nested temporal evaluation. Tune only on past weeks, refit on all information available before the test snapshot, and evaluate the next week once after training. Keep normalization, calibration, sidecars, and all learned preprocessing inside the same temporal boundary.

Promotion gate: test-week outcomes cannot affect checkpoint selection, hyperparameters, preprocessing, or calibration, and the protocol must reproduce deterministic prediction artifacts.

### 2. Calibration And Strong Controls

Compare LatentStrat against Statbotics pre-match predictions, rolling pRidge, additive alliance models, and model ablations. Add past-only probability calibration and report Brier score, log loss, score error, calibration, coverage, and paired uncertainty.

Promotion gate: demonstrate out-of-time improvement over strong controls or complementary value in a calibrated ensemble. Interaction-aware models must beat an otherwise comparable additive model.

### 3. Temporal Robot State

Evaluate evidence-driven updates in place of unconstrained event lookup deltas. Program history may initialize or regularize a season robot state; official and scouted observations update the same unconstrained state over weeks and matches. Match realization remains contextual rather than a permanent team attribute.

Promotion gate: improve Thursday, Friday, and Saturday historical snapshot forecasts without degrading unseen-team or low-data behavior.

### 4. Time-Aware Scouting Observations

Store scouting at its native team, team-event, team-match, alliance-match, match, or event grain. Every observation carries source/schema identity, `observed_at`, `known_as_of`, units, missingness, and provenance. Source-specific adapters use masked auxiliary losses so sparse observations remain useful without treating missing values as zero.

Team-match observations must be conditioned on the focal robot, partners, opponents, strategy, and match context.

Promotion gate: scouting improves future predictions for directly observed rows and demonstrates transfer to unscouted teams, matches, or events without temporal leakage.

### 5. Operational Snapshots

- Thursday: global weekly rebuild and updated state for all teams.
- Friday: ingest that event's TBA and scouting observations for remaining-match forecasts and a preliminary pick list.
- Saturday: laptop-feasible event-state inference or small adapter updates followed by alliance-selection analysis; avoid unrestricted trunk fine-tuning on one small event.

Promotion gate: snapshot generation meets its known-as-of contract and Saturday processing completes within the documented laptop runtime budget.

### 6. Multi-Season Representation

Use season-specific match-breakdown adapters around a shared unconstrained representation. Preserve game-specific score semantics while testing whether a common bottleneck transfers useful capability and interaction structure across seasons.

Promotion gate: improve held-out-season or early-season forecasts over season-local and prior-season baselines.

### 7. Alliance Selection And Tournament Simulation

Rank candidates conditionally on existing alliance members, remaining teams, likely opponents, uncertainty, and the event's actual tournament structure. Recompute after each selection and optimize expected tournament advancement or win probability rather than individual team rating.

Promotion gate: outperform EPA-greedy and additive selection policies across historical selection-time snapshots, with calibrated tournament probabilities and explicit uncertainty.

### 8. Latent Analysis

Fit PCA only after model training. Use a fixed reference basis for temporal trajectories, and supplement raw-state projections with nearest neighbors and counterfactual partner/opponent fingerprints.

Promotion gate: diagnostic conclusions are stable across seeds or aligned snapshots and agree with held-out scouting or predictive behavior; raw coordinates are never presented as ground-truth robot traits.
