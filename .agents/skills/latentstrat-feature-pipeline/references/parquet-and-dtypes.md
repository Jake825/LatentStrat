# Parquet And Dtypes

LatentStrat uses PyArrow-backed Parquet as the boundary between feature construction and training.

## Persisted Feature Tables

`write_feature_table` validates the table, removes training-derived `*_idx` and `*_z` columns, and writes Parquet with the `pyarrow` engine. `read_feature_table` reads that file back with the same engine.

Persisted Parquet should keep source-friendly dtypes:

- String keys remain strings.
- Nullable integer columns can remain pandas nullable `Int64`.
- Timestamps and ordering columns should keep their appropriate pandas dtype.
- Categorical or string fields should not be forced into floats.
- Continuous feature columns should only be numeric when their meaning is numeric.

## Do Not Blanket Cast To Float32

Avoid advice such as "cast every numeric DataFrame column to `float32` before writing Parquet." That can corrupt nullable integer semantics, timestamps, IDs, and categorical encodings.

The current training path converts tensor-bound targets at the dataset boundary:

- `MatchTensorDataset` converts team index matrices to `torch.long`.
- Continuous and binary target tensors are converted to `torch.float32`.
- New model-facing tensors should follow the same principle: preserve useful DataFrame dtypes, then cast deliberately when constructing tensors.

## Derived Training Columns

`make_team_index_map` creates `*_idx` columns and `apply_target_stats` creates normalized `*_z` target columns during training. These are derived from the current train run and are not persisted by `write_feature_table`.

If generated code saves feature tables manually, ensure it does not accidentally persist stale team indices or normalization columns as if they were raw features.
