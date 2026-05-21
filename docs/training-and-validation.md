# Training And Validation

LatentStrat separates network data ingestion from PyTorch training using a
data-lake workflow. This prevents memory leaks, makes hyperparameter tuning
fast, and allows offline training from a local Parquet feature file.

## 1. Feature Building

Run:

```bash
latentstrat build-features --season 2026 --output data/features_2026.parquet
```

1. **Extract**: `tbapy` pulls TBA data. Network calls are cached locally via
   `requests-cache`.
2. **Transform**: Pandas builds a flat match table with string team keys such as
   `frc254` and raw target columns. If `data/scouting.db` exists, SQLModel
   scouting rows are left-joined into the same flat table.
3. **Load**: The CLI saves the typed Pandas DataFrame to Parquet with the
   explicit `pyarrow` engine.

Parquet is used because CSV cannot reliably preserve nullable integer and
timestamp dtypes across the feature boundary.

Use `latentstrat init-scouting-db` to create the optional SQLite scouting
database. Training never queries SQLite directly.

## 2. Splits And Normalization

LatentStrat supports `chronological-holdout`, `week-held-out`, and
`event-held-out`.

Target normalization (`fit_target_stats`) is split-local. Validation rows never
influence the `mu` or `sigma` calculations. Constant training targets
automatically receive `sigma = 1.0` to prevent division-by-zero failures.

To map string FRC team keys to contiguous integer tensors, `make_team_index_map`
runs in memory after loading the Parquet file. This ensures 0-based integer
indices match the teams present in that specific training run.

## 3. PyTorch Training Loop

Run:

```bash
latentstrat train-features data/features_2026.parquet
```

- Data is loaded into a `MatchTensorDataset` and served through PyTorch
  `DataLoader`s.
- CUDA runs use pinned-memory transfer when available.
- Hardware permitting, LatentStrat uses Automatic Mixed Precision via
  `torch.amp.autocast` and safe `torch.compile()` acceleration.
- Training can use early stopping on validation loss and restore the best epoch's
  weights.

## 4. Baselines And Controls

`fit_baselines` computes:

- A `MeanBaseline`, using the mean of training targets.
- A `RidgeMatchOPR` baseline, using a sparse match design matrix and
  `sklearn.linear_model.Ridge(fit_intercept=False, solver="sparse_cg")`.

`build_evidence_packet` compares the official model against:

- `shuffled_set_control`: shuffles red/blue team slots inside split masks.
- `null_label_control`: shuffles complete target tuples inside split masks.

These controls are applied at the Pandas DataFrame level before training. The
PyTorch model does not need special control-mode branches.

## 5. Evaluation Context

During evaluation, the model runs under `torch.inference_mode()` with
`model.eval()` active. Prediction is batched through a `DataLoader` so full
seasons do not allocate one large tensor on GPU or MPS.
