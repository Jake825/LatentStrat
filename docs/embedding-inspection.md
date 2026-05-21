# Embedding Inspection

Embedding inspection is the post-training workflow for understanding the learned
`team_embedding` weight matrix and the PMA attention tables. It does not retrain
the model.

Read embedding outputs alongside the broader guidance in
[Model Evaluation](model-evaluation.md), especially calibration, baselines,
controls, and feature availability.

## Entry Point

Run the CLI command:

```bash
latentstrat inspect-embeddings --event-key 2026ilch
```

Outputs are written as `.csv` files under `artifacts/inspection/`.

## Team Embedding Table

The `team_embeddings.csv` file extracts the raw vectors from PyTorch's
`nn.Embedding` and joins them with team-perspective analytics, including:

- `team_key` and `team_index`.
- The raw embedding vector and its L2 norm.
- `pc1`, `pc2`, `pc3`, computed via `numpy.linalg.svd`.
- Match count and event count.
- `avg_alliance_score`, `avg_point_differential`, `rate_win`.
- Ridge match-OPR anchors.

## Sanity Checks

The inspection workflow runs immediate data-quality assertions:

- All embeddings must be finite.
- No zero norms or collapsed norm distributions.
- No single dominant embedding dimension.
- Embedding norms cannot be nearly perfectly correlated with match count.

These are quality gates for mathematical interpretation, not validation metrics.

## Cosine Neighbors And Archetypes

Cosine-neighbor tables (`nearest_neighbors.csv`) calculate the distance between
normalized embedding rows using dot products. Similar teams should have similar
observed target profiles more often than random teams with matching match
counts.

Archetype reports (`archetype_similarity.csv`) are data-derived. The pipeline
finds the top teams for statistical metrics such as `high_win_rate`, averages
their normalized embeddings to create a prototype vector, and reports the cosine
similarity of all other teams to that prototype. They are not hand-labeled claims
about team reputation.

## PMA Attention And Zero-Out Diagnostics

PMA weights describe how the learned `pma_seed` summarized a contextualized
alliance block prior to prediction. They are not literal causal explanations.

Read the PMA outputs against `zero_out_diagnostics.csv`. Representation utility
is stronger when artificially masking a specific slot to `0.0` causes a measured
spike in validation RMSE.
