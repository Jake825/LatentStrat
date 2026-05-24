# Tensor Contracts

Use this reference before changing model inputs or dataset construction.

## Inputs

Current model-facing inputs are:

- Red base indices: `[batch, 3]`, dtype `torch.long`.
- Blue base indices: `[batch, 3]`, dtype `torch.long`.
- Red event indices: `[batch, 3]`, dtype `torch.long`.
- Blue event indices: `[batch, 3]`, dtype `torch.long`.
- Missing-team diagnostic masks: `[batch, 3]` per alliance.
- Continuous targets: `torch.float32`.
- Binary targets: `torch.float32`.
- V5.7 atomic, foul, bonus, and special targets: `torch.float32`, with `NaN`
  entries masked at loss time.

The three slots are alliance members, not independent row labels. Slot order should not encode scouting rank, driver station quality, or team strength.

## Ghost Slot Routing

The current model has no attention padding mask for missing robot slots.
Explicit missing slots and random team-dropout slots are routed to:

```text
team_base_idx = 0
team_event_idx = 0
```

`Z_base[0]` is a learned ghost robot from the prior. `Z_event[0]` is the
zero/null event delta. These slots remain visible to SAB, CROSS, and PMA
attention.

Do not add advice that missing teams can be handled by passing
`padding_mask=True`. That is not current LatentStrat behavior.

## zero_slot

`zero_slot` can mask a specific team slot during forward passes for diagnostics. It is used to estimate slot contribution and sensitivity. Normal missing-slot support should use ghost slot routing, not `zero_slot`.

If a future task changes missing-team behavior, it must change runtime code, tests, docs, and evaluation expectations together.
