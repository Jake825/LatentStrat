# Training And Regularization

Use this reference for architecture-sensitive training behavior.

## Heads And Targets

The model has separate heads for continuous and binary targets. Continuous targets are evaluated with regression metrics such as RMSE and MAE. Binary targets are evaluated as probabilities, so calibration-aware metrics matter.

Coordinate metric interpretation with `$latentstrat-model-evaluation`.

## Optimizer Groups

Team embeddings and the rest of the model can use different optimizer settings. Preserve that structure when reviewing training changes because embedding behavior is central to the project.

## Active Embedding Regularization

The training path can regularize embeddings for active teams. Do not replace this with global regularization over all possible teams unless a future runtime task explicitly changes the objective and updates the validation story.

## Split-Safe Normalization

Target normalization is fitted on the train split and applied to other splits. Keep this split boundary intact. Normalizing on all rows leaks validation/test information.
