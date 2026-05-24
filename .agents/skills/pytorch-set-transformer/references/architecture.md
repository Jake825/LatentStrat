# Architecture

LatentStrat models a match as two unordered alliances of three teams.

## Current Components

The current `SetTransformerModel` uses:

- `Z_base`, a durable team embedding table initialized from the V5.6 prior when supplied.
- `Z_event`, an event-local delta table with row `0` reserved as the null delta.
- A shared teammate self-attention block for each alliance.
- Shared cross-alliance attention so red can attend to blue and blue can attend to red.
- PMA pooling with a learned seed to reduce each alliance set to one vector.
- Separate V5/V5.7 match heads plus learning-to-rank value heads.

## Forward Pass Shape

The model receives red and blue base/event index tensors shaped `[batch, 3]`.

At a high level:

1. Look up each team's base embedding and event delta.
2. Add them as `Z_base[team_base_idx] + Z_event[team_event_idx]`.
3. Route explicit missing slots and random dropout slots to learned `Z_base[0]`
   plus null `Z_event[0]`.
4. Apply the same SAB to the three red team embeddings and three blue team embeddings.
5. Let each alliance representation attend to the opposing alliance through cross-attention.
6. Pool each alliance set into `z_red` and `z_blue` with PMA.
7. Build the match vector:

```text
diff = z_red - z_blue
z_match = [z_red, z_blue, diff, abs(diff), z_red * z_blue]
```

`diff` helps the model learn relative alliance strength. `abs(diff)` lets the model see mismatch magnitude independent of sign. The elementwise product captures interaction terms that are not represented by either alliance alone.

## Current Heads

The current model exposes:

- Continuous phase heads for red/blue auto and teleop points.
- Atomic count heads.
- Committed foul heads.
- Bonus and special binary heads that output raw logits.
- Anti-symmetric red-win logits.
- Ordinal endgame logits per slot.
- Judged-award logits per slot.
- Team and alliance value heads for rank, playoff, and selection sidecar losses.

## Permutation Tolerance

The architecture is designed around set-like alliance inputs. Do not add slot-specific parameters or order-dependent transforms unless a future task explicitly changes the modeling assumption and updates tests.
