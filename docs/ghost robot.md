---
tags:
  - latentstrat
  - model-architecture
  - prior-training
aliases:
  - "ghost token"
  - "team 0"
  - "null robot"
  - "learned ghost robot"
related:
  - "[[prior-training]]"
  - "[[model-structure]]"
  - "[[student-primer]]"
  - "[[model-architecture-reference]]"
---

# Ghost Robot

The ghost robot is LatentStrat's learned representation for an empty or missing
robot slot. It is team `0`, stored in `Z_base[0]`, and it is trained as a real
prior row instead of being deleted with an attention mask.

This lets the match model always process three slots per alliance. If an
alliance is short a robot, or a training dropout intentionally blanks a slot, the slot routes to the ghost robot rather than disappearing from the [[Set Transformer]].

## Related

- [Prior training](prior-training.md): how team `0` is trained in the Day Zero
  prior.
- [Model structure](model-structure.md): how missing slots route through
  `Z_base[0]` and `Z_event[0]`.
- [Student primer](student-primer.md): plain-language explanation of why the
  ghost robot exists.
- [Model architecture reference](model-architecture-reference.md): exact tensor
  shape and row-index behavior.
