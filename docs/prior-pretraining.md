# Prior Pretraining

Prior pretraining creates Day Zero team embeddings before current-event matches are available.

## Inputs

- TBA team history and award signals.
- OpenAI narrative embeddings cached locally.
- Statbotics normalized EPA trajectory selected by `epa_source_year`.

`epa_source_year` is active configuration. It selects the four-year EPA trajectory window and must
not be removed as dead code.

## Workflow

```powershell
latentstrat pretrain prior build `
  --target-season 2026 `
  --output data/features/pretraining/prior/prior_features_2026.parquet

latentstrat pretrain prior train `
  --features data/features/pretraining/prior/prior_features_2026.parquet `
  --output artifacts/pretraining/prior/prior_run `
  --epochs 1000 `
  --latent-dim 16

latentstrat pretrain prior inspect `
  --checkpoint artifacts/pretraining/prior/prior_run/checkpoint.pt `
  --output artifacts/pretraining/prior/prior_run/inspection
```

OpenAI requests require `OPENAI_API_KEY` only for cache misses. The cache lives at
`data/cache/openai_embeddings.sqlite`.

Prior training and every latent-dimension grid trial use the shared CPU runtime: AdamW, gradient clipping at `1.0`, optimizer-step cosine decay, deterministic seed streams, and TensorBoard observability. Training writes `resume/fit/latest.ckpt`; grid trials write `resume/<latent-dim>/latest.ckpt`. Pass `--resume-checkpoint` to continue the matching phase with identical data and configuration.

## Contract

The checkpoint exports a team embedding table used to initialize season-model `Z_base`. Prior
inspection writes PCA, neighborhood, and diagnostic artifacts plus `manifest.json`.

See [Evaluation and artifacts](evaluation-and-artifacts.md).
