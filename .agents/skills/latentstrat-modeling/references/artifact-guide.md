# Modeling Artifact Guide

## Saved Predictions

Candidate prediction artifacts require event and match keys, predicted and actual red/blue totals, and red win probability. Statbotics artifacts provide the corresponding pre-match baseline values. Validate uniqueness, finite scores, probability bounds, season consistency, and coverage before pairing.

The saved-prediction evaluator writes paired predictions, aggregate and historical-week metrics, ten-bin calibration, coverage, event-cluster bootstrap results, and `manifest.json`.

## Training and Walk-Forward Evidence

Review continuous and binary metrics separately. Check fold policy, checkpoint selection, target availability, optional-data slices, and manifest provenance before comparing runs.

The historical V5.8 replay is development evidence rather than unbiased promotion evidence because its reported held-out weeks also influenced checkpoint selection. Preserve `promotion_eligible=false` for comparisons derived from that protocol.

## Diagnostics

PMA attention, zero-slot ablation, PCA, t-SNE, cosine neighbors, and archetype reports are model-behavior probes. Use them for debugging and interpretation, never as standalone promotion or causal evidence.
