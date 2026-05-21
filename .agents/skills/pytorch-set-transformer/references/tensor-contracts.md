# Tensor Contracts

Use this reference before changing model inputs or dataset construction.

## Inputs

Current model-facing inputs are:

- Red team indices: `[batch, 3]`, dtype `torch.long`.
- Blue team indices: `[batch, 3]`, dtype `torch.long`.
- Continuous targets: `torch.float32`.
- Binary targets: `torch.float32`.

The three slots are alliance members, not independent row labels. Slot order should not encode scouting rank, driver station quality, or team strength.

## No Padding Mask

The current model has no padding mask and no ghost-team embedding. Training and evaluation require nonmissing team indices. The feature pipeline should reject missing team inputs before tensors reach the model.

Do not add advice that missing teams can be handled by passing `padding_mask=True`. That is not current LatentStrat behavior.

## zero_slot

`zero_slot` can mask a specific team slot during forward passes for diagnostics. It is used to estimate slot contribution and sensitivity. It is not a general missing-team or no-show modeling mechanism.

If a future task adds missing-team support, it must change runtime code, tests, docs, and evaluation expectations together.
