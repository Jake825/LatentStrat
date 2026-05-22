# Training And Validation

LatentStrat separates network data ingestion from PyTorch training using a
data-lake workflow. This prevents memory leaks, makes hyperparameter tuning
fast, and allows offline training from a local Parquet feature file.

## 1. Feature Building

Detailed feature construction, scouting merge prefixes, Parquet dtype behavior,
and team indexing are covered in [Feature Pipeline](feature-pipeline.md).

Run:

```bash
latentstrat build-features --season 2026 --output data/features_2026.parquet
```

1. **Extract**: `tbapy` pulls TBA data. Network calls are cached locally via
   `requests-cache`.
2. **Transform**: Pandas builds a flat match table with string team keys such as
   `frc254`, V5 match targets, endgame labels, and post-event award auxiliary
   labels. If `data/scouting.db` exists, SQLModel scouting rows are left-joined
   into the same flat table.
3. **Load**: The CLI saves the typed Pandas DataFrame to Parquet with the
   explicit `pyarrow` engine.

Parquet is used because CSV cannot reliably preserve nullable integer and
timestamp dtypes across the feature boundary. Tensor-bound values are converted
deliberately when `MatchTensorDataset` builds PyTorch tensors; Parquet columns
are not blanket-cast to `float32`.

Use `latentstrat init-scouting-db` to create the optional SQLite scouting
database. Training never queries SQLite directly.

## 2. V5.5 Prior Pretraining

V5.5 is an offline Day Zero initializer for `Z_base`. It builds one
target-season-quarantined narrative per team, embeds that text with
`text-embedding-3-small` at 256 dimensions, then trains a denoising autoencoder
into the V5 16-D latent bottleneck.

Run:

```bash
latentstrat build-prior-features --target-season 2026 \
  --teams-from data/features_2026.parquet \
  --output data/prior_features_2026.parquet
latentstrat train-prior --features data/prior_features_2026.parquet \
  --output data/pretrained_prior_2026.pt
latentstrat inspect-prior --checkpoint data/pretrained_prior_2026.pt \
  --features data/prior_features_2026.parquet \
  --output artifacts/prior_2026
latentstrat train-features data/features_2026.parquet \
  --prior-checkpoint data/pretrained_prior_2026.pt
```

`build-prior-features` uses the target-season feature Parquet only to discover
team keys. Historical TBA profile, event, and award records are included in the
narrative only when `year < target_season`; target-season and future events or
awards are excluded from the prior text. OpenAI embeddings are cached in SQLite
by a stable hash of `(model, dimensions, narrative_text)`. Uncached narratives
are embedded in batches with bounded exponential backoff for rate-limit and
transient server errors, and each successful batch is cached immediately so a
rerun can resume without repeating completed API calls.

The prior checkpoint is keyed by stable `team_key`, not by the generated
`team_base_idx`. During `train-features --prior-checkpoint`, matching team
vectors overwrite the corresponding `Z_base` rows after the current feature
table creates its own team index map. Null row `0` and unknown rookie teams keep
their normal random initialization.

`inspect-prior` writes PCA plots, nearest-neighbor tables, reconstruction MSE,
sanity checks, and training-loss artifacts for the prior latent vectors before
downstream V5 match training changes `Z_base`.

## 3. Splits And Normalization

LatentStrat supports `chronological-holdout`, `week-held-out`, and
`event-held-out`.

Target normalization (`fit_target_stats`) is split-local. Validation rows never
influence the `mu` or `sigma` calculations. Constant training targets
automatically receive `sigma = 1.0` to prevent division-by-zero failures.

To map string FRC team keys to tensors, `make_v5_team_index_maps` runs in memory
after loading the Parquet file. Real teams and real event-team pairs start at
index `1`; index `0` is reserved for null slots. Unknown teams require
rebuilding the mapping or a deliberate future inference policy.

## 4. PyTorch Training Loop

Run:

```bash
latentstrat train-features data/features_2026.parquet
```

- Data is loaded into a `MatchTensorDataset` containing base indices, event
  indices, missing masks, continuous targets, win targets, ordinal endgame
  labels, and NaN-masked award targets.
- Healthy team slots are randomly swapped with the null token during training at
  `opts.team_dropout_rate` (default `0.03`) so missing-team behavior is learned
  before it is needed in inference.
- CUDA runs use pinned-memory transfer when available.
- Hardware permitting, LatentStrat uses Automatic Mixed Precision via
  `torch.amp.autocast` and safe `torch.compile()` acceleration.
- Training can use early stopping on validation loss and restore the best epoch's
  weights.

## 5. Baselines And Controls

`fit_baselines` computes:

- A `MeanBaseline`, using the mean of training targets.
- A ridge match-OPR-style baseline, using a sparse match design matrix and
  `sklearn.linear_model.Ridge(fit_intercept=False, solver="sparse_cg")`.

`build_evidence_packet` compares the official model against:

- `shuffled_set_control`: shuffles red/blue team slots inside split masks.
- `null_label_control`: shuffles complete target tuples inside split masks.

These controls are applied at the Pandas DataFrame level before training. The
PyTorch model does not need special control-mode branches.

## 6. Evaluation Context

During evaluation, the model runs under `torch.inference_mode()` with
`model.eval()` active. Prediction is batched through a `DataLoader` so full
seasons do not allocate one large tensor on GPU or MPS.

## 7. Venue Mode And Consolidation

Venue mode fine-tunes only `Z_event` for a supplied event:

```bash
latentstrat train-features data/features_2026.parquet --venue-mode --event-key 2026ilch \
  --checkpoint artifacts/features_run/v5_checkpoint.pt
```

After the event, `consolidate-event` folds event deltas into durable base
embeddings using `DeltaIntegrationGate(Z_base, Z_event, delta_weeks)` and writes
the result to `data/latentstrat_embeddings.sqlite`:

```bash
latentstrat consolidate-event artifacts/features_run/v5_checkpoint.pt --event-key 2026ilch
```
