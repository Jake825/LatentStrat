# Training And Regularization

Use this reference for architecture-sensitive training behavior.

## Heads And Targets

The model has separate heads for continuous phase targets, win logits, endgame
ordinal logits, judged-award logits, V5.7 atomic counts, committed fouls, bonus
binary logits, special binary logits, and sidecar value tasks. Continuous
targets are evaluated with regression metrics such as RMSE and MAE. Binary
targets are evaluated as probabilities, so calibration-aware metrics matter.

Binary heads output raw logits. Use `BCEWithLogitsLoss` with masks for missing
target entries; do not add model-side sigmoid layers for training.

Coordinate metric interpretation with `$latentstrat-model-evaluation`.

## Optimizer Groups

Team embeddings and the rest of the model can use different optimizer settings. Preserve that structure when reviewing training changes because embedding behavior is central to the project.

## Active Embedding Regularization

The training path can regularize embeddings for active teams. Do not replace this with global regularization over all possible teams unless a future runtime task explicitly changes the objective and updates the validation story.

`Z_base` active-row regularization includes row `0` because the ghost robot is a
learned live base row. `Z_event` active-row regularization excludes row `0`
because event row `0` is the null delta.

## Split-Safe Normalization

Target normalization is fitted on the train split and applied to other splits. Keep this split boundary intact. Normalizing on all rows leaks validation/test information.

## TensorBoard Observability

Training script changes should preserve TensorBoard observability. LatentStrat training CLIs are expected to offer local TensorBoard logging by default, with an explicit opt-out for quiet batch and test runs.
