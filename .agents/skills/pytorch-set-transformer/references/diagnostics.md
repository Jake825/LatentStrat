# Diagnostics

LatentStrat exposes model diagnostics that depend on the Set Transformer architecture.

## PMA Attention

PMA attention artifacts can help explain which team embeddings influenced pooled alliance vectors. Treat attention as a diagnostic aid, not a complete causal explanation.

## Zero-Out Diagnostics

Zero-out diagnostics evaluate how outputs change when one team slot is masked. Use these as sensitivity checks for model behavior and evidence packets.

Remember that `zero_slot` is diagnostic only. It does not mean the model can train on incomplete alliances.

## Embeddings

Team embeddings are learned representations. Interpret them alongside metrics, neighbors, archetypes, and known data availability. Do not treat embedding coordinates as direct scouting measurements.
