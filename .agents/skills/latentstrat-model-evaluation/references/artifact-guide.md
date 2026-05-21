# Artifact Guide

Use this reference when reading training output folders or evidence packets.

## Metrics Artifacts

Review both continuous and binary metrics. For binary outcomes, Brier score, log loss, and calibration bins are more important than accuracy alone.

## Availability Slices

Availability slices help reveal whether a model depends on rows that have optional scouting or enrichment data. Compare slices before claiming a new data source improved the whole model.

## Evidence Packets

Evidence packets should combine:

- Metrics and calibration.
- Baseline and control comparisons.
- Feature availability context.
- Attention and zero-out diagnostics when relevant.
- Embedding inspection only as supporting evidence.

## Attention And Zero-Out Outputs

PMA attention and zero-out diagnostics are model-behavior probes. They are useful for review and debugging but should not be presented as definitive causal explanations.
