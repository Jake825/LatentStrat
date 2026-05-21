# Team Indexing And Tensors

Team indexing bridges FRC string keys and PyTorch embeddings.

## Team Vocabulary

`make_team_index_map` scans the six team slot columns:

- `red_team_1_key`, `red_team_2_key`, `red_team_3_key`.
- `blue_team_1_key`, `blue_team_2_key`, `blue_team_3_key`.

It maps non-empty `frc####` keys to contiguous integer ids and writes nullable pandas `Int64` index columns for each slot. The mapping is created for the current training table.

## Per-Run Mapping

The current mapping is not a persistent public vocabulary. If an inference workflow sees a team key that was not present during training, the workflow must either:

- Rebuild training artifacts with a table that includes the team.
- Define an explicit future inference policy.

Do not invent an unknown-team embedding or hidden OOV index unless the runtime model code is intentionally changed.

## Tensor Contract

`MatchTensorDataset` expects nonmissing team index columns. The current `match_team_matrices` path raises an error if any team index is missing.

The model receives:

- Red team index tensor shaped `[batch, 3]`.
- Blue team index tensor shaped `[batch, 3]`.
- Continuous and binary target tensors as `torch.float32`.

Use `$pytorch-set-transformer` before changing shapes, team-slot semantics, or model-facing tensors.
