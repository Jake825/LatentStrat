---
tags:
  - latentstrat
  - model-architecture
aliases:
  - "Model Structure"
  - "Set Transformer Structure"
related:
  - "[[Set Transformer]]"
  - "[[model-architecture-reference]]"
  - "[[prior-training]]"
  - "[[season-training]]"
  - "[[ghost robot]]"
---

# Model Structure

For a higher-level explanation of how the model fits into the project, see the [Documentation Hub](index.md), [Prior Training](prior-training.md), and [Season Training](season-training.md).

For the exact tensor shapes, layer dimensions, parameter counts, and loss equations, see the [Model Architecture Reference](model-architecture-reference.md).

LatentStrat V6-Lite is a cross-alliance Set Transformer written in PyTorch. Each match row contains three red slots and three blue slots. Each slot carries a durable team base index, an event-team delta index, and a missing-team diagnostic mask. V6-Lite retains the supervised V5 trunk and reserves frozen-target auxiliary predictors. The first historical score encoder is offline-only and is not attached to this runtime model yet.

Agent-facing architecture guidance lives in [`$pytorch-set-transformer`](../.agents/skills/pytorch-set-transformer/SKILL.md).

## Latent Trunk

- `Z_base`: durable team embedding table. Index `0` is a learned live ghost robot; V5.6 maps `frc####` directly to row `####` for supported team numbers.
- `Z_event`: zero-initialized event-team delta table. Index `0` is reserved for null; real `(event_key, team_key)` pairs start at `1`.

For a normal slot, the model uses `Z_base[team_base_idx] + Z_event[team_event_idx]`. For a missing or randomly dropped slot, it routes the slot to `Z_base[0]` and `Z_event[0]`. The transformer still attends to all three slots; missing/dropout masks are retained as diagnostics and for auxiliary target masking.

## Day Zero Prior

V5.6.4 can initialize `Z_base` from an offline text-plus-normalized-EPA trajectory and cultural transductive prior checkpoint. The pretraining artifact stores `embedding_table`, normally shaped `[12501, 16]`, where row `0` is the learned ghost robot and row `254` is the learned prior vector for `frc254`. During `train-features --prior-checkpoint`, the table is copied directly into `Z_base.weight`.

The checkpoint width must match `opts.latent_dim` exactly. Team numbers above the prior table bound are unsupported until the prior is rebuilt with a larger `max_team_number`. `Z_event` remains event-local and is not part of the prior distillation artifact. The sacrificial pretraining decoder predicts both the OpenAI narrative vector and normalized prior-season Statbotics EPA, then is discarded before checkpoint export.

## Attention Topology

V5 uses explicit module boundaries:

- `MAB`: multihead attention plus residual feed-forward refinement.
- `SAB`: self-attention inside each alliance.
- `CROSS`: red queries blue and blue queries red.
- `PMA`: learned pooling seed for each alliance.

The forward pass is:

```python
red_slots = Z_base[red_base_idx] + Z_event[red_event_idx]
blue_slots = Z_base[blue_base_idx] + Z_event[blue_event_idx]
red_context = SAB(red_slots, all_slots_unmasked)
blue_context = SAB(blue_slots, all_slots_unmasked)
red_cross = CROSS(red_context, blue_context, all_slots_unmasked)
blue_cross = CROSS(blue_context, red_context, all_slots_unmasked)
z_red = PMA(red_cross, all_slots_unmasked)
z_blue = PMA(blue_cross, all_slots_unmasked)
```

During `model.train()`, healthy non-missing slots are randomly dropped at `opts.team_dropout_rate` (default `0.03`) by routing them to the learned ghost row. Dropout is disabled in evaluation and inference.

## Task Heads

- Continuous alliance heads predict red/blue auto and teleop points.
- Atomic alliance heads predict generic score-breakdown count arrays such as auto count, shift counts, transition count, and endgame count.
- Foul heads predict committed foul points and committed foul counts after TBA's awarded foul columns are inverted.
- Bonus and special binary heads output raw logits for capability thresholds and special penalty flags. Training uses `BCEWithLogitsLoss`; the model does not apply sigmoid internally.
- `SiameseWinHead` predicts `red_win` with an anti-symmetric, bias-free logit.
- `OrdinalEndgameHead` predicts cumulative logits for `None < Level1 < Level2 < Level3` per robot slot.
- `JudgesRoomHead` predicts post-event judged-award auxiliary logits per slot.
- `TeamValueHead` predicts a scalar team value from one `Z_base + Z_event` vector for qualification rank pair losses.
- `AllianceValueHead` predicts a scalar alliance value from a pooled alliance representation for playoff ordering losses.

V6-Lite runtime scaffolding includes three lightweight predictors:

- `AwardPrototypePredictor(z_team_event)` predicts a normalized 256D semantic award prototype.
- `RankOutcomePredictor(z_team_event)` predicts a frozen 16D completed-event outcome latent.
- `SelectionEmbeddingPredictor(z_captain, z_candidate)` predicts a frozen 16D pick-desirability latent.

The offline historical match-breakdown encoder exports frozen 16D latents per played alliance. A follow-up will add the runtime score contract:

```text
z_red  -> ScoreEmbeddingPredictor -> frozen_red_16d
z_blue -> ScoreEmbeddingPredictor -> frozen_blue_16d
```

The preserved V1 artifact is reconstruction-only. An explicit V2 offline ablation keeps the same
16D export and adds masked-source latent denoising, a diagnostic winner-quality projection, and
audit-first numeric score-equation residuals. The V2 quality scalar is not a downstream target.

The pick target builder uses ranked-unselected pool context offline. The runtime predictor only
receives the captain and candidate latents, so the match loader never carries a variable-sized
draft pool.

Judged awards use NaN-masked targets. Impact and EI share cultural gradients, while machine awards are masked single-hot targets so an Autonomous win does not create false negative labels for Quality, Design, Control, or Excellence.

All task losses are routed through a task-name keyed homoscedastic balancer. If a target group is unavailable or all entries in a batch are `NaN`, that task is inactive for the step and does not update its log variance.

Read [V6-Lite](V6-Lite.md) for the target-space builders, fold-safety boundary, and phase gates.

## Consolidation

`DeltaIntegrationGate` is the offline Sunday-night consolidation module. It accepts `(Z_base, Z_event, delta_weeks)` and emits a trust gate used as:

```python
Z_new_base = Z_base + trust_gate * Z_event
```

`delta_weeks` is required because the physical time between events changes how much an event delta should be trusted.

## Related

- [Model architecture reference](model-architecture-reference.md): exact tensor shapes, parameter counts, and equations.
- [Prior training](prior-training.md): how `Z_base` is initialized.
- [Season training](season-training.md): how match and sidecar training use the model.
- [Ghost robot](ghost%20robot.md): the learned missing-slot row.
- [Embedding inspection](embedding-inspection.md): how to inspect learned team vectors.
