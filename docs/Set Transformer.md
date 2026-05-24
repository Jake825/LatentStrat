---
tags:
  - latentstrat
  - model-architecture
aliases:
  - "Set Transformer"
  - "set transformer"
related:
  - "[[model-structure]]"
  - "[[model-architecture-reference]]"
  - "[[season-training]]"
  - "[[student-primer]]"
---

# Set Transformer

A Set Transformer is an attention-based neural network designed for unordered sets. LatentStrat uses this idea because an FRC alliance is a set of three robots: swapping slot order should not fundamentally change the meaning of the alliance.

In LatentStrat, the Set Transformer contextualizes red and blue robot vectors, pools each alliance into one learned alliance representation, and then feeds task heads for match scores, win probability, endgame, awards, and sidecar ranking losses.

## Related

- [Model structure](model-structure.md): readable explanation of LatentStrat's Set Transformer trunk.
- [Model architecture reference](model-architecture-reference.md): exact tensor shapes and attention block sizes.
- [Season training](season-training.md): how match batches flow through the model.
- [Student primer](student-primer.md): beginner explanation of alliance interaction modeling.
