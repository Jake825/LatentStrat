# Architecture

LatentStrat models a match as two unordered alliances of three teams.

## Current Components

The current `SetTransformerModel` uses:

- A team embedding table.
- A shared teammate self-attention block for each alliance.
- Shared cross-alliance attention so red can attend to blue and blue can attend to red.
- PMA pooling with a learned seed to reduce each alliance set to one vector.
- Separate continuous and binary prediction heads.

## Forward Pass Shape

The model receives red and blue team index tensors shaped `[batch, 3]`.

At a high level:

1. Look up each team's embedding.
2. Apply the same SAB to the three red team embeddings and three blue team embeddings.
3. Let each alliance representation attend to the opposing alliance through cross-attention.
4. Pool each alliance set into `z_red` and `z_blue` with PMA.
5. Build the match vector:

```text
diff = z_red - z_blue
z_match = [z_red, z_blue, diff, abs(diff), z_red * z_blue]
```

`diff` helps the model learn relative alliance strength. `abs(diff)` lets the model see mismatch magnitude independent of sign. The elementwise product captures interaction terms that are not represented by either alliance alone.

## Permutation Tolerance

The architecture is designed around set-like alliance inputs. Do not add slot-specific parameters or order-dependent transforms unless a future task explicitly changes the modeling assumption and updates tests.
