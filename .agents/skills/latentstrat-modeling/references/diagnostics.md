# Diagnostics

LatentStrat exposes model diagnostics that depend on the Set Transformer architecture.

## PMA Attention

PMA attention artifacts can help explain which team embeddings influenced pooled alliance vectors. Treat attention as a diagnostic aid, not a complete causal explanation.

## Zero-Out Diagnostics

Zero-out diagnostics evaluate how outputs change when one team slot is masked.
Use these as sensitivity checks for model behavior and evidence packets.

Remember that `zero_slot` is diagnostic only. Normal missing robot slots are
routed through learned ghost base row `0` plus null event row `0`, and those
slots remain visible to attention.

## Embeddings

Team embeddings are learned representations. Interpret them alongside metrics, neighbors, archetypes, and known data availability. Do not treat embedding coordinates as direct scouting measurements.
