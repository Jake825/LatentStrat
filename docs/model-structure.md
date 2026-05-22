# Model Structure

LatentStrat V5 is a masked cross-alliance Set Transformer written in PyTorch.
Each match row contains three red slots and three blue slots. Each slot carries a
durable team base index, an event-team delta index, and a missing-team mask.

Agent-facing architecture guidance lives in
[`$pytorch-set-transformer`](../.agents/skills/pytorch-set-transformer/SKILL.md).

## Latent Trunk

- `Z_base`: durable team embedding table. Index `0` is reserved for null; real
  teams start at `1`.
- `Z_event`: zero-initialized event-team delta table. Index `0` is reserved for
  null; real `(event_key, team_key)` pairs start at `1`.
- `null_team_token`: learned ghost slot used for explicit missing teams and
  training-time random team dropout.

For a normal slot, the model uses `Z_base[team_base_idx] + Z_event[team_event_idx]`.
For a missing or randomly dropped slot, it uses `null_team_token` and passes the
slot through the attention mask.

## Day Zero Prior

V5.5 can initialize `Z_base` from an offline text-only prior checkpoint. The
pretraining artifact stores `team_vectors` keyed by stable TBA team keys such as
`frc254`. During `train-features --prior-checkpoint`, the current feature table
first regenerates its per-run `team_base_idx` map, then matching prior vectors
are copied into `Z_base.weight[team_base_idx]`.

Prior vectors must match `opts.latent_dim` exactly. Row `0`, `null_team_token`,
and teams missing from the prior checkpoint retain the normal model
initialization.

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
red_context = SAB(red_slots, red_missing_mask)
blue_context = SAB(blue_slots, blue_missing_mask)
red_cross = CROSS(red_context, blue_context, blue_missing_mask)
blue_cross = CROSS(blue_context, red_context, red_missing_mask)
z_red = PMA(red_cross, red_missing_mask)
z_blue = PMA(blue_cross, blue_missing_mask)
```

During `model.train()`, healthy non-missing slots are randomly dropped at
`opts.team_dropout_rate` (default `0.03`) so the model learns 2v3 and 1v3
behavior. Dropout is disabled in evaluation and inference.

## Task Heads

- Continuous alliance heads predict red/blue auto and teleop points.
- `SiameseWinHead` predicts `red_win` with an anti-symmetric, bias-free logit.
- `OrdinalEndgameHead` predicts cumulative logits for
  `None < Level1 < Level2 < Level3` per robot slot.
- `JudgesRoomHead` predicts post-event judged-award auxiliary logits per slot.

Judged awards use NaN-masked targets. Impact and EI share cultural gradients,
while machine awards are masked single-hot targets so an Autonomous win does not
create false negative labels for Quality, Design, Control, or Excellence.

## Consolidation

`DeltaIntegrationGate` is the offline Sunday-night consolidation module. It
accepts `(Z_base, Z_event, delta_weeks)` and emits a trust gate used as:

```python
Z_new_base = Z_base + trust_gate * Z_event
```

`delta_weeks` is required because the physical time between events changes how
much an event delta should be trusted.
