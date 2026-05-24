---
name: pytorch-set-transformer
description: Use when working with LatentStrat's PyTorch Set Transformer model architecture, tensor contracts, alliance slot handling, team embeddings, self-attention blocks, cross-alliance attention, PMA pooling, zero_slot diagnostics, model heads, loss behavior, or architecture documentation. Covers current behavior and prevents inventing padding masks, ghost teams, or unimplemented model features.
---

# PyTorch Set Transformer

Use this skill for LatentStrat model architecture work and model-facing tensor contracts.

## Core Directives

1. Describe current model behavior, not aspirational architecture.
2. Treat alliances as unordered sets of three team slots. Preserve permutation-tolerant behavior when editing model code or tests.
3. Keep the current representation explicit: shared team embeddings, shared SAB, cross-alliance attention, PMA pooling, then `z_match = [z_red, z_blue, diff, abs(diff), z_red * z_blue]`.
4. Keep current ghost-slot behavior explicit: missing/dropout slots route to learned `Z_base[0]` plus null `Z_event[0]` and remain visible to attention.
5. Treat `zero_slot` as a diagnostic sensitivity mechanism, distinct from normal ghost-slot routing.

## References

- Read `references/architecture.md` for the current forward pass and representation choices.
- Read `references/tensor-contracts.md` before changing input tensors, team indexing, or missing-team behavior.
- Read `references/training-and-regularization.md` for loss heads, optimizer groups, and active embedding regularization.
- Read `references/diagnostics.md` for PMA attention, zero-out diagnostics, and architecture-sensitive evaluation outputs.

## Coordinate With Other Skills

- Use `$latentstrat-feature-pipeline` for Parquet, team indexing, and tensor construction.
- Use `$latentstrat-model-evaluation` when interpreting metrics, attention artifacts, zero-out diagnostics, or embeddings.
- Use `$frc-time-aware-analysis` before using features whose availability depends on match or event timing.
