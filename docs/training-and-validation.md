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

## 2. V5.6.1 Prior Distillation

V5.6.1 is an offline Day Zero initializer for `Z_base`. It builds a fixed
transductive dictionary for team numbers `0..12500` by default, where row `0` is the
learned ghost robot.
Historical known teams, unassigned gap numbers, and future rookie numbers each
receive target-season-quarantined narrative text. Those narratives are embedded
with `text-embedding-3-small` at 256 dimensions, then a `TeamPriorDistiller`
trains `nn.Embedding[12501, 16] -> deep decoder -> OpenAI target + EPA target`.

Run:

```bash
latentstrat build-prior-features --target-season 2026 \
  --output data/prior_features_2026.parquet
latentstrat train-prior --features data/prior_features_2026.parquet \
  --output artifacts/prior_v56_day_zero \
  --epochs 1000 \
  --latent-dim 16 \
  --tensorboard
tensorboard --logdir=runs
latentstrat inspect-prior \
  --checkpoint artifacts/prior_v56_day_zero/checkpoint.pt \
  --output artifacts/prior_2026
latentstrat train-features data/features_2026.parquet \
  --prior-checkpoint artifacts/prior_v56_day_zero/checkpoint.pt
```

`build-prior-features` pages the TBA team universe and writes exactly one row
per team number from `0` through `max_team_number`. Row `0` uses a fixed learned
ghost robot narrative. Known teams active in the target season become Anchors; known
inactive teams become Ghosts; unknown historical gaps become Siblings using
nearby team-number context; unknown future numbers become Future rookies using
the projected registration formula. Performance facts in narrative text are
limited to `year < target_season`; target-season and future results or awards
are excluded.

The feature table also includes `target_epa`, a z-scored prior-season Statbotics
EPA target. By default `target_season - 1` is used, so a 2026 prior uses final
2025 team-year EPA. Team `0` uses raw EPA `0.0`; teams without prior-season EPA
use the rookie baseline `mean - 0.2 * std` before normalization.

OpenAI embeddings are cached in SQLite by a stable hash of
`(model, dimensions, narrative_text)`. Uncached narratives are embedded in
batches with bounded exponential backoff for rate-limit and transient server
errors, and each successful batch is cached immediately so a rerun can resume
without repeating completed API calls.

The production prior checkpoint is stripped after training. It exports
`embedding_table: Tensor[12501, 16]` for the default cap, metadata, and training history, but not the
decoder, EPA head, or full model state. During `train-features --prior-checkpoint`, that
table is copied directly into `Z_base.weight`, including the learned row `0`
ghost robot. Team numbers above the checkpoint bounds raise a clear error and
require rebuilding the V5.6 prior with a larger maximum.

`train-prior` writes local TensorBoard event files under `runs/` by default.
Use `tensorboard --logdir=runs` to view total loss, OpenAI MSE, EPA MSE, learned
log variances, and precision weights while the prior trains. Use
`--no-tensorboard` to disable this side artifact. TensorBoard logs are ignored
by Git and are not stored in the stripped checkpoint.

`inspect-prior` writes PCA plots, nearest-neighbor tables, sanity checks, norm
histograms, and training-loss artifacts for the prior latent vectors before
downstream V5 match training changes `Z_base`. Feature Parquet is optional; when
a stripped production checkpoint is inspected, reconstruction MSE is reported as
`NaN` because the sacrificial decoder was intentionally discarded.

### Prior Bottleneck Grid

`run-prior-grid` is an experiment-only command for measuring the prior
information bottleneck. It does not change production `TeamPriorDistiller`
training or V5 checkpoint handoff. Each run uses the same production coordinate
map and sweeps only the latent dimension:

```text
team_number -> Embedding(max_team_number + 1, latent_dim)
            -> decoder -> 256-D OpenAI target + normalized EPA target
```

Smoke run:

```bash
latentstrat run-prior-grid --features data/prior_features_2026.parquet \
  --output artifacts/prior_grid_smoke \
  --epochs 1 \
  --latent-dims 2,4,8
```

Full elbow sweep:

```bash
latentstrat run-prior-grid --features data/prior_features_2026.parquet \
  --output artifacts/prior_elbow_grid \
  --epochs 500 \
  --latent-dims 2,4,8,16,32,64,128,256
```

The command writes CSV summaries, per-epoch history, failure records, and
`prior_elbow_curve.png` under the output directory. Failed or OOM runs are
recorded and the remaining dimensions continue.

## 3. Splits And Normalization

LatentStrat supports `chronological-holdout`, `week-held-out`, and
`event-held-out`.

Target normalization (`fit_target_stats`) is split-local. Validation rows never
influence the `mu` or `sigma` calculations. Constant training targets
automatically receive `sigma = 1.0` to prevent division-by-zero failures.
V5.7 atomic-count and committed-foul regression targets are normalized with the
same train-split-only rule. Missing score-breakdown values remain `NaN` and are
masked at loss time.

To map string FRC team keys to tensors, `make_v5_team_index_maps` runs in memory
after loading the Parquet file. For V5.6, `frc####` maps directly to
`team_base_idx=####` for team numbers `1..12500` by default, and `team_base_idx=0`
is the learned ghost robot used for blank or explicit missing slots. Real
event-team pairs still map to contiguous `team_event_idx` values starting at
`1`, with `team_event_idx=0` reserved for no event-local delta.

## 4. PyTorch Training Loop

Run:

```bash
latentstrat train-features data/features_2026.parquet
```

- Data is loaded into a `MatchTensorDataset` containing base indices, event
  indices, missing masks, continuous targets, win targets, ordinal endgame
  labels, NaN-masked award targets, and optional V5.7 atomic/foul/bonus/special
  targets.
- Healthy team slots are randomly routed to the learned ghost row during
  training at `opts.team_dropout_rate` (default `0.03`) so missing-team behavior
  is learned before it is needed in inference.
- CUDA runs use pinned-memory transfer when available.
- Hardware permitting, LatentStrat uses Automatic Mixed Precision via
  `torch.amp.autocast` and safe `torch.compile()` acceleration.
- Training can use early stopping on validation loss and restore the best epoch's
  weights.
- V5.7 uses a homoscedastic task balancer for match-spine tasks and optional
  sidecar ranking tasks. Binary V5.7 heads emit logits and are trained with
  masked `BCEWithLogitsLoss`.

Optional sidecar losses can be supplied at training time:

```bash
latentstrat train-features data/features_2026.parquet \
  --rankings-sidecar data/v57_sidecars/rankings_2026.parquet \
  --selections-sidecar data/v57_sidecars/selections_2026.parquet \
  --playoffs-sidecar data/v57_sidecars/playoffs_2026.parquet
```

The match loader drives each epoch. Non-empty sidecar loaders are cycled so
short selection or playoff datasets do not stop the match epoch early. Empty
sidecars are skipped.

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
