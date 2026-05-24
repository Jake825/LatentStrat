---
tags:
  - latentstrat
  - model-architecture
aliases:
  - "Architecture Reference"
  - "Tensor Shape Reference"
related:
  - "[[model-structure]]"
  - "[[prior-training]]"
  - "[[season-training]]"
  - "[[ghost robot]]"
---

# Model Architecture Reference

This page records the exact V5.6.4/V5.8 model shapes, parameter sizes, and loss equations used by the current LatentStrat code. It is intended as the technical companion to the more readable [Model Structure](model-structure.md), [Prior Training](prior-training.md), and [Season Training](season-training.md) pages.

The values below were checked against:

- `src/latentstrat/config.py`
- `src/latentstrat/model.py`
- `src/latentstrat/prior_model.py`
- `src/latentstrat/training.py`
- `src/latentstrat/evaluation.py`

## Current Defaults

| Setting | Current value | Source |
|---|---:|---|
| Prior latent dimension | `16` | `PriorOpts.latent_dim` |
| Season latent dimension | `16` | `LatentStratOptions.latent_dim` |
| OpenAI target dimension | `256` | `PriorOpts.llm_dim` |
| Maximum team number | `12500` | `PriorOpts.max_team_number` |
| Prior embedding rows | `12501` | rows `0..12500` |
| Attention heads | `1` | `LatentStratOptions.attention_heads` |
| Set Transformer FFN dimension | `16` | `LatentStratOptions.set_ffn_dim` |
| Team dropout rate | `0.03` | `LatentStratOptions.team_dropout_rate` |
| Training mini-batch size | `256` | `LatentStratOptions.mini_batch_size` and `PriorOpts.batch_size` |
| Season validation fraction | `0.20` | `LatentStratOptions.validation_fraction` |
| Prior training epochs | `1000` | `PriorOpts.epochs` |
| Season training epochs | `200` | `LatentStratOptions.epochs` |
| Loss log-var clamp | `[-5.0, 5.0]` | `LatentStratOptions.loss_log_var_min/max` |

If `latent_dim` changes, every `[16]` latent shape in this page scales with that value. The output target dimensions stay fixed unless the feature schema changes.

## V5.6.4 Prior Model

The prior model is a training-only coordinate map. It learns a table of team vectors and uses sacrificial decoder heads to force those vectors to explain narrative, normalized EPA trajectory, and cultural history targets.

### Prior Layers

| Component | Shape | Parameters |
|---|---:|---:|
| `team_embedding` | `Embedding(12501, 16)` | `200,016` |
| Decoder layer 1 | `Linear(16, 256)` | `4,352` |
| Decoder layer 2 | `Linear(256, 512)` | `131,584` |
| Decoder layer 3 | `Linear(512, 512)` | `262,656` |
| `openai_head` | `Linear(512, 256)` | `131,328` |
| `norm_epa_head` | `Linear(512, 4)` | `2,052` |
| `culture_head` | `Linear(512, 7)` | `3,591` |
| Log variance parameters | OpenAI scalar, EPA scalar, culture `[7]` | `9` |
| Full prior training model | all rows and heads | `735,588` |

The decoder trunk is:

```text
16 -> Linear(256) -> GELU -> Linear(512) -> GELU -> Linear(512) -> GELU
```

The exported production checkpoint strips every decoder, head, and log-var parameter. It keeps only:

```text
embedding_table: [12501, 16]
```

That table contains `200,016` FP32 values, or `800,064` bytes, about `0.76 MiB` before metadata and serialization overhead.

### Prior Forward Shapes

Let `B` be the batch size.

| Tensor | Shape | Meaning |
|---|---:|---|
| `team_number` | `[B]` | integer team numbers, including learned ghost team `0` |
| `z_latent` | `[B, 16]` | learned team coordinate |
| decoder hidden state | `[B, 512]` | shared sacrificial representation |
| `pred_openai` | `[B, 256]` | OpenAI narrative vector prediction |
| `pred_norm_epa` | `[B, 4]` | normalized EPA trajectory prediction |
| `pred_culture` | `[B, 7]` | raw cultural history prediction |

The forward contract is:

```python
pred_openai, pred_norm_epa, pred_culture, z_latent = model(team_number)
```

### Prior Loss Math

The OpenAI loss is a per-team vector squared distance, not per-element MSE:

$$ L_{openai} = \frac{1}{B} \sum_{b=1}^{B} \sum_{j=1}^{256} (\hat{x}_{b,j} - x_{b,j})^2 $$

The normalized EPA trajectory is one grouped 4D task. Missing team-years are masked. With observed mask `m` and valid count `C = sum(m)`:

$$ L_{epa} = \left( \frac{\sum m_{b,j}(\hat{e}_{b,j} - e_{b,j})^2}{\max(C, 1)} \right) \cdot 4 $$

If `C == 0` for a batch, the EPA task contributes no loss and no log-var term for that batch.

Culture is seven raw, unnormalized scalar targets. Each axis has its own MSE:

$$ L_{culture,i} = \frac{1}{B} \sum_{b=1}^{B} (\hat{c}_{b,i} - c_{b,i})^2 $$

Homoscedastic balancing is applied as:

$$ L_t^{balanced} = \exp(-s_t)L_t + s_t $$

For prior training, `s_openai` and `s_epa` are scalars. `s_culture` is a 7-element vector, one value per culture target.

## Season Model Embeddings

The season model learns match behavior from the prior and event data.

```text
z_team = Z_base[team_base_idx] + Z_event[team_event_idx]
```

| Table | Shape | Behavior |
|---|---:|---|
| `Z_base` | `[num_teams, 16]` | long-term team identity, initialized from prior when supplied |
| `Z_event` | `[num_event_teams, 16]` | event-local delta table |

With the standard V5.6.4 prior cap, `Z_base` has at least:

```text
12501 * 16 = 200,016 parameters
```

`Z_event` is variable-sized because it depends on the loaded feature table. Event row `0` is the null delta. Missing slots still use the learned ghost base row `0`, but `Z_event[0]` contributes no event-local movement.

The season non-embedding core is about `9.4k` trainable parameters, plus `Z_base` and variable-size `Z_event`.

## Set Transformer Tensor Shapes

Let `B` be the batch size.

| Tensor | Shape | Meaning |
|---|---:|---|
| Red base indices | `[B, 3]` | direct team-number indices into `Z_base` |
| Blue base indices | `[B, 3]` | direct team-number indices into `Z_base` |
| Red event indices | `[B, 3]` | contiguous event-team indices into `Z_event` |
| Blue event indices | `[B, 3]` | contiguous event-team indices into `Z_event` |
| Red slot vectors | `[B, 3, 16]` | `Z_base + Z_event` for red slots |
| Blue slot vectors | `[B, 3, 16]` | `Z_base + Z_event` for blue slots |
| Red self-attended context | `[B, 3, 16]` | red SAB output |
| Blue self-attended context | `[B, 3, 16]` | blue SAB output |
| Red cross-attended context | `[B, 3, 16]` | red queries blue |
| Blue cross-attended context | `[B, 3, 16]` | blue queries red |
| Six-slot context | `[B, 6, 16]` | red and blue cross-attended slots concatenated |
| `z_red` | `[B, 16]` | pooled red alliance representation |
| `z_blue` | `[B, 16]` | pooled blue alliance representation |
| PMA attention weights | `[B, 1, 3]` | one set per alliance |

The match-level representation concatenates five 16D vectors:

```text
z_match = concat(z_red, z_blue, z_red - z_blue, abs(z_red - z_blue), z_red * z_blue)
```

So:

```text
z_match: [B, 80]
```

### Attention Blocks

The main attention block is `MAB`:

```text
MultiheadAttention(embed_dim=16, num_heads=1, batch_first=True)
Residual + LayerNorm
Linear(16, 16) -> GELU -> Dropout -> Linear(16, 16) -> Dropout
Residual + LayerNorm
```

The instantiated blocks are:

| Block | Parameters |
|---|---:|
| Red/blue shared `SAB` | `1,696` |
| Cross-alliance `CROSS` | `1,696` |
| PMA pooling block and seed | `1,712` |

Missing slots and random team dropout route slots to base row `0` and event row `0`. The current model does not attention-mask these ghost slots; it keeps all three alliance slots visible.

## Season Model Heads

| Head | Input | Output | Parameters |
|---|---:|---:|---:|
| Continuous phase head | per-alliance `[B, 16]` | `[B, 4]` | `34` |
| Atomic count head | per-alliance `[B, 16]` | `[B, 14]` | `119` |
| Foul head | per-alliance `[B, 16]` | `[B, 6]` | `51` |
| Bonus binary head | per-alliance `[B, 16]` | `[B, 6]` logits | `51` |
| Special binary head | per-alliance `[B, 16]` | `[B, 2]` logits | `17` |
| Siamese win head | `z_red`, `z_blue` | `[B, 1]` logit | `544` |
| Ordinal endgame head | six-slot context | `[B, 6, 3]` logits | `51` |
| Judges room head | six-slot context | `[B, 6, 7]` logits | `119` |
| Team value head | one team latent | scalar | `17` |
| Alliance value head | one alliance latent | scalar | `17` |
| Delta integration gate | `Z_base`, `Z_event`, `delta_weeks` | `[16]` gate | `3,216` |
| Homoscedastic balancer | task names | `11` scalar log vars | `11` |

The continuous phase output order follows the configured continuous targets:

```text
red_auto_pts, red_teleop_pts, blue_auto_pts, blue_teleop_pts
```

The atomic, foul, bonus, and special heads each produce red outputs followed by blue outputs for the configured target groups.

Binary heads output raw logits. Training uses masked `BCEWithLogitsLoss`; the model does not apply sigmoid inside the forward pass.

## Season Loss Math

All match-spine and sidecar task losses pass through the same homoscedastic balancer:

$$ L^{balanced} = \sum_{t \in active} \left(\exp(-s_t)L_t + s_t\right) $$

After each optimizer step, task log variances are clamped:

$$ -5 \le s_t \le 5 $$

The maximum precision weight is therefore:

$$ \exp(5) \approx 148.4 $$

### Continuous And Count Targets

Continuous, atomic, and foul losses are masked MSE losses. Missing target values remain `NaN` in the feature table and are ignored at loss time:

$$ L_{mse} = \frac{\sum m_i(\hat{y}_i - y_i)^2}{\max(\sum m_i, 1)} $$

### Binary Targets

Win, bonus, special, and award targets use logits and masked binary cross-entropy. Conceptually:

$$ L_{bce} = BCEWithLogits(\hat{l}, y) $$

where entries with missing labels are removed before reduction.

### Ordinal Endgame Targets

Endgame labels are cumulative ordinal logits over:

```text
None < Level1 < Level2 < Level3
```

With four classes, each slot emits three cumulative logits:

```text
[B, 6, 3]
```

### Sidecar Losses

Qualification and playoff ordering use margin ranking losses:

$$ L_{rank} = \max(0, margin - (score_{better} - score_{worse})) $$

Alliance selection uses a triplet loss:

$$ L_{triplet} = \max(0, d(captain, pick) - d(captain, passed\_over) + margin) $$

Sidecar loaders are optional. Empty or missing sidecars make their task inactive instead of producing dummy losses.

### Embedding Regularization

The optimizer excludes embeddings from global AdamW decay. Embeddings use active-row L2 penalties instead, so only rows touched by a batch are pulled toward zero.

`Z_base` active-row L2 includes row `0`, because the ghost robot is a learned live prior row. `Z_event` active-row L2 excludes row `0`, because event row `0` is the null delta.

## Consolidation Math

The offline consolidation gate merges event evidence back into the base prior:

```text
gate = sigmoid(MLP(concat(Z_base, Z_event, delta_weeks)))
Z_new = Z_base + gate * Z_event
```

For the default latent width:

```text
concat(Z_base, Z_event, delta_weeks): [33]
MLP: 33 -> 64 -> 16
```

`delta_weeks` is included because evidence from a recent event should not necessarily be trusted the same way as evidence from an older event.

## Reporting Metrics

LatentStrat reports red-win probability:

$$ p_{red} = sigmoid(red\_win\_logit) $$

Match accuracy:

$$ accuracy = \frac{1}{N} \sum_{i=1}^{N} 1[(p_{red,i} \ge 0.5) = red\_win_i] $$

Brier score:

$$ Brier = \frac{1}{N} \sum_{i=1}^{N} (p_{red,i} - red\_win_i)^2 $$

Binary log loss:

$$ LogLoss = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

Phase score MSE compares predicted `auto + teleop` points against actual `auto + teleop` points for both alliances.

Total score MSE uses the V5.7 foul inversion direction. A team's FMS total includes points awarded from opponent committed fouls:

```text
red_total_pred = red_phase_pred + predicted_blue_committed_foul_points
blue_total_pred = blue_phase_pred + predicted_red_committed_foul_points
```

If total score columns or foul statistics are missing, total score MSE is reported as `NaN` instead of failing.

## Rebuild Notes

- `Z_base` prior handoff requires checkpoint width to match `LatentStratOptions.latent_dim`.
- Team numbers above the prior row bound require rebuilding the prior with a larger `max_team_number`.
- `Z_event` size is feature-table dependent and cannot be fully parameter-counted without a concrete dataset.
- The current architecture assumes the learned ghost team `0` is visible to attention, not masked out.
