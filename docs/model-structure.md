# Model Structure

LatentStrat is a cross-alliance Set Transformer match simulator written natively
in PyTorch. Each row represents a played match with three red teams and three
blue teams. The model encodes both alliances as unordered sets, lets each
alliance attend to the other, pools each side into a fixed-length vector, and
predicts multi-task match targets.

Agent-facing architecture guidance lives in
[`$pytorch-set-transformer`](../.agents/skills/pytorch-set-transformer/SKILL.md).

## Symbols

- `N`: total number of indexed teams.
- `d`: latent dimension, `opts.latent_dim`.
- `f`: feed-forward hidden width, `opts.set_ffn_dim`.
- `Kc`: number of continuous targets in `opts.target_map`.
- `Kb`: number of binary targets in `opts.binary_targets`.
- `B`: mini-batch size.

## Learnable Parameters

All learnable weights are encapsulated in standard `torch.nn.Module` components.

| Component | PyTorch Module | Shape | Meaning |
| :--- | :--- | :---: | :--- |
| `team_embedding` | `nn.Embedding(N, d)` | `N x d` | Global team embedding matrix. |
| `sab` | `SetAttentionBlock` | `d x d`, `d x f` | Shared self-attention block for red and blue alliance encoders. |
| `cross` | `SetAttentionBlock` | same as SAB | Shared cross-alliance attention block. |
| `pma` | `SetAttentionBlock` | same as SAB | Pooling-by-multihead-attention block. |
| `pma_seed` | `nn.Parameter` | `d` | Learned query seed for alliance pooling. |
| `cont_head` | `nn.Linear(5d, Kc)` | `Kc x 5d` | Continuous match-target prediction head. |
| `bin_head` | `nn.Linear(5d, Kb)` | `Kb x 5d` | Binary logit prediction head. |

`SetAttentionBlock` uses PyTorch's optimized `nn.MultiheadAttention`
(`batch_first=True`) alongside `nn.LayerNorm` and a `nn.Sequential` GELU
feed-forward network.

## Forward Pass

For each batch, `model(red_team_idx, blue_team_idx)` looks up embeddings as
`[B, 3, d]` tensors:

```python
X_red = model.team_set(red_team_idx)
X_blue = model.team_set(blue_team_idx)
```

The shared self-attention block contextualizes teammates:

```python
H_red, _ = sab(query=X_red, key_value=X_red)
H_blue, _ = sab(query=X_blue, key_value=X_blue)
```

The shared cross-attention block lets each alliance query the opponent:

```python
H_red_interacted, _ = cross(query=H_red, key_value=H_blue)
H_blue_interacted, _ = cross(query=H_blue, key_value=H_red)
```

PMA pools each interacted alliance using the learned seed:

```python
z_red, red_pma_weights = pma_pool(H_red_interacted)
z_blue, blue_pma_weights = pma_pool(H_blue_interacted)
```

The match representation concatenates the pooled vectors and their symmetric
differences:

```python
diff = z_red - z_blue
z_match = torch.cat([z_red, z_blue, diff, diff.abs(), z_red * z_blue], dim=1)
```

The multi-task heads generate predictions:

```python
pred.cont_z = cont_head(z_match)
pred.bin_logits = bin_head(z_match)
```

## Tensor Contract

The current model expects complete red and blue team index tensors shaped
`[B, 3]`. Team indices are generated from `frc####` keys during training and are
fed to `nn.Embedding` as `torch.long` tensors.

There is no padding mask or ghost-team embedding in the current architecture.
Training and evaluation require nonmissing team indices. The `zero_slot`
mechanism is for diagnostics that intentionally mask one slot during analysis;
it is not general missing-team handling.

## Loss And Active-Team Regularization

Continuous loss uses standard `F.mse_loss`. Binary loss uses numerically stable
`F.binary_cross_entropy_with_logits`. Degenerate all-positive or all-negative
targets automatically receive a positive weight of `1.0`.

If PyTorch's `AdamW` weight decay were applied globally to the `team_embedding`
matrix, the optimizer would push unobserved teams' embeddings toward zero every
step.

To prevent this, standard weight decay is disabled for embeddings in
`optimizer_parameter_groups`. Instead, LatentStrat applies a custom L2 penalty
only to the unique active teams physically present in the current mini-batch via
`active_embedding_l2()`.
