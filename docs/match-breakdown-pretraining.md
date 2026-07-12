# Match-Breakdown Pretraining

Match-breakdown pretraining learns an offline 16D representation for one played alliance result.
It is not wired into supervised season training yet.

## Corpus

```powershell
latentstrat pretrain match-breakdown sync `
  --start-season 2015 `
  --end-season 2026
```

The durable corpus is `data/pretraining/match-breakdown/corpus.sqlite`. TBA transport caches remain
separate under `data/cache/`. Training defaults to official event types `0..5`; use
`--include-foc` or `--include-remote` only for explicit extended runs. Rows with null score
breakdowns remain in the corpus for refreshes but are excluded from training.

## V1 Baseline

Each eligible season has its own typed schema and encoder/decoder route. All routes share one 16D
bottleneck:

```text
concat(values_y, masks_y)
  -> E_y: 2N_y -> 128 -> 64
  -> G:   64 -> 32 -> 16
  -> D_y: 16 -> 64 -> 128 -> N_y
```

Missing values remain `value=0, mask=0`; physical zero remains `value=0, mask=1`. No normalization
file is written or consumed.

```powershell
latentstrat pretrain match-breakdown train `
  --start-season 2015 `
  --end-season 2026 `
  --epochs 50 `
  --seasons-per-step 4 `
  --rows-per-season 64
```

## V2 Ablation

V2 adds latent denoising consistency, winner ranking, and audit-first numeric scoring equations.
It is explicit opt-in:

```powershell
latentstrat pretrain match-breakdown train `
  --config configs/pretraining/match-breakdown-v2.yaml
```

V2 writes a separate artifact and remains `promotion_eligible=false`. Preserve V1 unchanged and
compare the V2 inspection against V1 before promotion.

## Runtime and Resume

Both evaluation and all-data phases use the shared CPU optimizer runtime with AdamW, gradient clipping at `1.0`, optimizer-step cosine decay, stable phase/epoch seeds, and TensorBoard enabled by default. Epoch checkpoints are written to `resume/eval/latest.ckpt` and `resume/all_data/latest.ckpt` inside the artifact directory. Continue a phase with `--resume-checkpoint`; the source tensors and resolved configuration must match exactly.

## Inspection

```powershell
latentstrat pretrain match-breakdown inspect `
  --artifact-dir artifacts/pretraining/match-breakdown/v1_2015_2026
```

Projection plots and cross-season neighbor tables are interpretation aids, not standalone
promotion evidence.
