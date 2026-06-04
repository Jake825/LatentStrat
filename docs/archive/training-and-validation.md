---
tags:
  - latentstrat
  - season-training
  - prior-training
  - metrics
aliases:
  - "Training And Validation"
  - "Training Guide"
related:
  - "[[prior-training]]"
  - "[[season-training]]"
  - "[[metrics-and-artifacts]]"
  - "[[cli-reference]]"
---

# Training And Validation

For a guided rebuild path, start with the [documentation hub](index.md), [rebuild guide](rebuild-from-scratch.md), [prior training](prior-training.md), and [season training](season-training.md).

LatentStrat separates network data ingestion from PyTorch training using a data-lake workflow. This prevents memory leaks, makes hyperparameter tuning fast, and allows offline training from a local Parquet feature file.

## 1. Feature Building

Detailed feature construction, scouting merge prefixes, Parquet dtype behavior, and team indexing are covered in [Feature Pipeline](feature-pipeline.md).

Run:

```bash
latentstrat build-features --season 2026 --output data/features/season/features_2026.parquet
```

1. **Extract**: `tbapy` pulls TBA data. Network calls are cached locally via `requests-cache`.
2. **Transform**: Pandas builds a flat match table with string team keys such as `frc254`, V5 match targets, endgame labels, and post-event award auxiliary labels. If `data/scouting/scouting.db` exists, SQLModel scouting rows are left-joined into the same flat table.
3. **Load**: The CLI saves the typed Pandas DataFrame to Parquet with the explicit `pyarrow` engine.

Parquet is used because CSV cannot reliably preserve nullable integer and timestamp dtypes across the feature boundary. Tensor-bound values are converted deliberately when `MatchTensorDataset` builds PyTorch tensors; Parquet columns are not blanket-cast to `float32`.

Use `latentstrat init-scouting-db` to create the optional SQLite scouting database. Training never queries SQLite directly.

## 2. V5.6.4 Prior Distillation

V5.6.4 is an offline Day Zero initializer for `Z_base`. It builds a fixed transductive dictionary for team numbers `0..12500` by default, where row `0` is the learned ghost robot. Historical known teams, unassigned gap numbers, and future rookie numbers each receive target-season-quarantined narrative text. Those narratives are embedded with `text-embedding-3-small` at 256 dimensions, then a `TeamPriorDistiller` trains `nn.Embedding[12501, 16] -> deep decoder -> OpenAI target + normalized EPA trajectory target + cultural targets`.

Run:

```bash
latentstrat build-prior-features --target-season 2026 \
  --output data/features/prior/prior_features_2026.parquet
latentstrat train-prior --features data/features/prior/prior_features_2026.parquet \
  --output artifacts/prior/prior_v564_latent16 \
  --epochs 1000 \
  --latent-dim 16 \
  --tensorboard
tensorboard --logdir=runs
latentstrat inspect-prior \
  --checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt \
  --output artifacts/prior/prior_v564_latent16/inspection
latentstrat train-features data/features/season/features_2026.parquet \
  --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt
```

`build-prior-features` pages the TBA team universe and writes exactly one row per team number from `0` through `max_team_number`. Row `0` uses a fixed learned ghost robot narrative. Known teams active in the target season become Anchors; known inactive teams become Ghosts; unknown historical gaps become Siblings using nearby team-number context; unknown future numbers become Future rookies using the projected registration formula. Performance facts in narrative text are limited to `year < target_season`; target-season and future results or awards are excluded.

The feature table also includes four years of Statbotics normalized EPA trajectory targets. For `target_season=2026`, those columns are `norm_epa_t_minus_4` through `norm_epa_t_minus_1`, sourced from 2022 through
2025. These values use Statbotics' own normalized EPA field and are not local
z-scores of raw EPA. Missing team-years stay `NaN` and are masked from the EPA trajectory loss.

V5.6.4 also adds raw cultural decoder targets: rookie-year delta from 1992, seasons played, total award count, blue banner count, championship appearance count, championship win count, and technical award count. These are deliberately unnormalized; per-axis homoscedastic log variances balance their scales during prior training. Team `0`, sibling rows, and future rows use finite zero cultural targets.

OpenAI embeddings are cached in SQLite by a stable hash of `(model, dimensions, narrative_text)`. Uncached narratives are embedded in batches with bounded exponential backoff for rate-limit and transient server errors, and each successful batch is cached immediately so a rerun can resume without repeating completed API calls.

The production prior checkpoint is stripped after training. It exports `embedding_table: Tensor[12501, 16]` for the default cap, metadata, and training history, but not the decoder, normalized-EPA head, culture head, or full model state. During `train-features --prior-checkpoint`, that table is copied directly into `Z_base.weight`, including the learned row `0` ghost robot. Team numbers above the checkpoint bounds raise a clear error and require rebuilding the V5.6 prior with a larger maximum.

`train-prior` writes local TensorBoard event files under `runs/` by default. Use `tensorboard --logdir=runs` to view total loss, feature-summed OpenAI vector distance, EPA trajectory MSE, learned log variances, per-year normalized EPA MSE, per-axis culture MSE, and precision weights while the prior trains. Use `--no-tensorboard` to disable this side artifact. TensorBoard logs are ignored by Git and are not stored in the stripped checkpoint.

`inspect-prior` writes PCA plots, nearest-neighbor tables, sanity checks, norm histograms, and training-loss artifacts for the prior latent vectors before downstream V5 match training changes `Z_base`. Feature Parquet is optional; when a stripped production checkpoint is inspected, reconstruction MSE is reported as `NaN` because the sacrificial decoder was intentionally discarded.

### Prior Bottleneck Grid

`run-prior-grid` is an experiment-only command for measuring the prior information bottleneck. It does not change production `TeamPriorDistiller` training or V5 checkpoint handoff. Each run uses the same production coordinate map and sweeps only the latent dimension:

```text
team_number -> Embedding(max_team_number + 1, latent_dim)
            -> decoder -> 256-D OpenAI target + normalized EPA target
```

Smoke run:

```bash
latentstrat run-prior-grid --features data/features/prior/prior_features_2026.parquet \
  --output artifacts/prior-grid/prior_grid_smoke \
  --epochs 1 \
  --latent-dims 2,4,8
```

Full elbow sweep:

```bash
latentstrat run-prior-grid --features data/features/prior/prior_features_2026.parquet \
  --output artifacts/prior-grid/prior_elbow_grid \
  --epochs 500 \
  --latent-dims 2,4,8,16,32,64,128,256
```

The command writes CSV summaries, per-epoch history, failure records, and `prior_elbow_curve.png` under the output directory. Failed or OOM runs are recorded and the remaining dimensions continue.

## 3. Splits And Normalization

LatentStrat supports `chronological-holdout`, `week-held-out`, and `event-held-out`.

Target normalization (`fit_target_stats`) is split-local. Validation rows never influence the `mu` or `sigma` calculations. Constant training targets automatically receive `sigma = 1.0` to prevent division-by-zero failures. V5.7 atomic-count and committed-foul regression targets are normalized with the same train-split-only rule. Missing score-breakdown values remain `NaN` and are masked at loss time.

To map string FRC team keys to tensors, `make_v5_team_index_maps` runs in memory after loading the Parquet file. For V5.6, `frc####` maps directly to `team_base_idx=####` for team numbers `1..12500` by default, and `team_base_idx=0` is the learned ghost robot used for blank or explicit missing slots. Real event-team pairs still map to contiguous `team_event_idx` values starting at `1`, with `team_event_idx=0` reserved for no event-local delta.

## 4. V6-Lite Offline Score Artifact

V6-Lite V1 builds the first score target space offline before attaching it to the Set Transformer:

```powershell
latentstrat sync-match-breakdowns `
  --start-season 2015 `
  --end-season 2026

latentstrat train-match-breakdown-encoder `
  --start-season 2015 `
  --end-season 2026 `
  --epochs 50 `
  --seasons-per-step 4 `
  --rows-per-season 64 `
  --learning-rate 0.001 `
  --seed 2026
```

The reconstruction-only V1 command remains the baseline. Run the structured V2 ablation explicitly:

```powershell
latentstrat train-match-breakdown-encoder `
  --config configs/pretraining/match-breakdown-v2.yaml
```

V2 retains the offline-only boundary. Review its raw-rule audit, reconstruction quality, effective
rank, quality correlations, and V1/V2 inspection drift before considering runtime integration.

The historical score encoder uses one raw typed schema per eligible season and a shared 16D
bottleneck. It writes no normalization file and never pads inputs to a cross-season union schema.
Its default event filter is TBA types `0..5`; FOC `6` and remote `7` are opt-in.

This artifact is not attached to full-season or walk-forward training yet. Score auxiliary
configuration remains disabled until a follow-up adds per-alliance frozen targets. Omitting
`--world-model-bundle` remains the supervised-only V6 ablation.

Read [V6-Lite](V6-Lite.md) for the phase order and promotion gates.

## 5. PyTorch Training Loop

Run:

```bash
latentstrat train-features data/features/season/features_2026.parquet
tensorboard --logdir=runs
```

- Data is loaded into a `MatchTensorDataset` containing base indices, event indices, missing masks, continuous targets, win targets, ordinal endgame labels, NaN-masked award targets, and optional V5.7 atomic/foul/bonus/special targets.
- Healthy team slots are randomly routed to the learned ghost row during training at `opts.team_dropout_rate` (default `0.03`) so missing-team behavior is learned before it is needed in inference.
- CUDA runs use pinned-memory transfer when available.
- Hardware permitting, LatentStrat uses Automatic Mixed Precision via `torch.amp.autocast` and safe `torch.compile()` acceleration.
- Training can use early stopping on validation loss. Best-validation weights are tracked independently, so `--no-early-stopping --restore-best` can run every requested epoch for TensorBoard while still saving the best validation state.
- V5.7 uses a homoscedastic task balancer for match-spine tasks and optional sidecar ranking tasks. Binary V5.7 heads emit logits and are trained with masked `BCEWithLogitsLoss`.
- V5.7.2 clamps task `log_var` values to `[-5, 5]`, caps precision blow-ups, applies cosine learning-rate decay down to `opts.lr_eta_min`, and uses AdamW decay on trainable Set Transformer/head weights while embeddings keep active-row L2.
- `train-features` writes TensorBoard event files under `runs/` by default. It logs aggregate train/validation loss, raw task losses, task activity, homoscedastic log variances, precision weights, and learning rate. Use `--no-tensorboard` to disable the side artifact.

### TensorBoard Policy

Every LatentStrat training CLI should support local TensorBoard observability. New or modified training commands should expose `--tensorboard/--no-tensorboard`, `--tensorboard-logdir`, and an optional `--tensorboard-run-name`. CLI training should default to TensorBoard on; direct Python APIs and tests should stay quiet unless a writer or log directory is supplied.

Training loops should log total train/validation loss, primary task losses, learning rate, and any learned loss weights or log variances. Writers must be closed on success or failure, and TensorBoard state must remain a side artifact under `runs/`; it should not be saved into model checkpoints.

Optional sidecar losses can be supplied at training time:

```bash
latentstrat train-features data/features/season/features_2026.parquet \
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet \
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet \
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet
```

The match loader drives each epoch. Non-empty sidecar loaders are cycled so short selection or playoff datasets do not stop the match epoch early. Empty sidecars are skipped.

For long monitored runs, keep TensorBoard live while disabling early stopping:

```bash
latentstrat train-features data/features/season/features_2026.parquet \
  --epochs 100 \
  --no-early-stopping \
  --restore-best \
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet \
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet \
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet
```

Walk-forward validation uses canonical `event_week` values. TBA Week 0 is bundled into model-facing Week 1, and missing TBA weeks fall back to dense chronological event order. Each fold starts from the Day Zero prior, trains on weeks `<= N`, validates on week `N + 1`, and pre-filters sidecars. `walk_forward_metrics.csv` includes row-weighted next-match phase score MSE, total score MSE when available, red-win accuracy, Brier score, and log loss. `walk_forward_predictions.parquet` enables paired fold-match bootstrap comparisons. Fold-local score-target rebuilding remains a future integration milestone.

```bash
latentstrat validate-walk-forward \
  --features data/features/season/features_2026.parquet \
  --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt \
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet \
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet \
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet \
  --output artifacts/walk-forward/v58_walk_forward_2026 \
  --epochs 5 \
  --latent-dim 16 \
  --tensorboard
```

## 6. Baselines And Controls

`fit_baselines` computes:

- A `MeanBaseline`, using the mean of training targets.
- A ridge match-OPR-style baseline, using a sparse match design matrix and `sklearn.linear_model.Ridge(fit_intercept=False, solver="sparse_cg")`.

`build_evidence_packet` compares the official model against:

- `shuffled_set_control`: shuffles red/blue team slots inside split masks.
- `null_label_control`: shuffles complete target tuples inside split masks.

These controls are applied at the Pandas DataFrame level before training. The PyTorch model does not need special control-mode branches.

## 7. Evaluation Context

During evaluation, the model runs under `torch.inference_mode()` with `model.eval()` active. Prediction is batched through a `DataLoader` so full seasons do not allocate one large tensor on GPU or MPS.

## 8. Venue Mode And Consolidation

Venue mode fine-tunes only `Z_event` for a supplied event:

```bash
latentstrat train-features data/features/season/features_2026.parquet --venue-mode --event-key 2026ilch \
  --checkpoint artifacts/season/features_run/v6_checkpoint.pt
```

After the event, `consolidate-event` folds event deltas into durable base embeddings using `DeltaIntegrationGate(Z_base, Z_event, delta_weeks)` and writes the result to `data/embeddings/latentstrat_embeddings.sqlite`:

```bash
latentstrat consolidate-event artifacts/season/features_run/v6_checkpoint.pt --event-key 2026ilch
```

## Related

- [Prior training](prior-training.md): offline Day Zero prior distillation.
- [Season training](season-training.md): match-spine, sidecar, and walk-forward training flow.
- [Metrics and artifacts](metrics-and-artifacts.md): how to read validation outputs.
- [CLI reference](cli-reference.md): command-by-command details.
- [Current state](current-state.md): current defaults and promoted artifacts.
