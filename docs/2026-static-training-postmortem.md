# 2026 Static Training Postmortem

This postmortem records the July 12, 2026 static robot-state experiments before LatentStrat attempts another architecture or training campaign. The source evidence is preserved in `artifacts/reference/2026-static-training-postmortem/`. The raw TensorBoard event files were summarized, hashed, and then intentionally deleted as part of the clean reset.

The central conclusion is narrow: the reliability experiment did not suffer numerical divergence, but its development gate correctly rejected the proposed fixed-duration protocol. Separately, the completed static reference produced systematically low out-of-time score forecasts. Those are related warnings, but they are not the same failure.

## Runs Performed

| Study | Objective | Work performed | Temporal boundary | Outcome |
|---|---|---|---|---|
| `2026-static-z-base` | Establish a simple one-seed static `Z_base` reference | Compared pre-2026 prior and random initialization on week 4; trained additive, teammate-set, and full-match selection/refit folds for weeks 6, 8, and 10 with an 11-epoch cap | Initialization used weeks 1-3 and week 4; each test fold selected on `T-1`, reinitialized, refit through `T-1`, then froze for `T` | Completed, non-promotable reference evidence |
| `2026-static-reliability` | Exercise the reliability workflow with a two-epoch smoke | Three architecture development runs, three confirmation runs, and three week-6 final runs, each for two epochs | Development week 4; final smoke trained through week 5 and predicted week 6 | Training completed, but final Parquet writing failed because integer neural seeds shared one column with string control/ensemble identifiers |
| `2026-static-reliability-smoke` | Confirm the corrected artifact, Streamlit, and TensorBoard path | Repeated the two-epoch development, confirmation, and week-6 architecture smoke | Same smoke boundary as above | Completed successfully; one seed and one test week, so no architecture conclusion |
| `2026-static-reliability-full` | Freeze one clipping threshold and one duration before a three-seed weeks 6/8/10 test matrix | Two-epoch timing pilot plus 3 architectures x 3 clipping thresholds x 40 development epochs | Only matches through week 3 trained the models; week 4 supplied development metrics | Clip 5 passed, but no common epoch from 15-40 satisfied every metric tolerance; the run stopped before confirmation or test training |

The first smoke failure was an engineering artifact-contract defect. It was fixed by normalizing the shared `seed` identifier before Parquet writing. It says nothing about model quality.

The full reliability outcome was a scientific gate rejection. All nine grid cells completed 40 epochs, their losses and metrics remained finite, and the timing budget passed. The test matrix did not start because the predeclared selection conditions were not met. Calling this numerical instability or a failed training process would be inaccurate.

## Development Metric Conflict

Clip 5 was the smallest threshold whose post-warmup clipping fraction stayed within the 10 percent rule across architectures. The remaining problem was duration selection.

| Architecture and epoch | Validation total loss | Score RMSE | Score-differential RMSE | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Additive, epoch 40 | `4.666` | `81.83` | `106.76` | `0.2080` | `0.6015` |
| Teammate-set, epoch 9 | `3.323` | `74.82` | `98.09` | `0.1756` | `0.5246` |
| Teammate-set, epoch 40 | `2.443` | `74.87` | `97.91` | `0.1993` | `0.7021` |
| Full-match, epoch 6 | `3.887` | `78.27` | `98.73` | `0.1769` | `0.5286` |
| Full-match, epoch 14 | `2.925` | `74.47` | `97.87` | `0.1999` | `0.6585` |
| Full-match, epoch 40 | `2.778` | `75.94` | `99.63` | `0.2227` | `0.9076` |

Additive improved gradually through the campaign. The interaction models learned useful-looking validation structure much earlier, but their win probabilities then became increasingly overconfident. Teammate-set probability metrics were best near epoch 9. Full-match probability metrics were best near epoch 6, while its score metrics improved until roughly epoch 14.

The heterogeneous validation objective continued to decrease after Brier score and log loss became materially worse. That objective combines many continuously and discretely supervised tasks with learned homoscedastic weights; it is not equivalent to score RMSE, Brier score, or log loss. The gate exposed that mismatch instead of allowing one convenient epoch to be chosen after seeing test results.

This is overfitting in the saved win-probability predictions relative to week 4, not an exploding-gradient failure. The evidence does not localize that behavior to the probability head, the shared representation, or both. Gradient norms remained finite, clipping was bounded, learning rates followed the configured schedule, and `Z_base` update distances plateaued normally.

## Why The Scores Were Low

The completed static reference, rather than the rejected reliability grid, provides the relevant out-of-time score predictions. Full-match showed the following aggregate behavior:

| Test week | Preceding training alliance mean | Actual alliance mean | Predicted alliance mean | Mean bias, predicted minus actual | Alliance-score RMSE |
|---:|---:|---:|---:|---:|---:|
| 6 | `150.9` | `180.68` | `122.35` | `-58.34` | `95.65` |
| 8 | `165.4` | `297.51` | `228.95` | `-68.55` | `113.19` |
| 10 | `178.4` | `434.04` | `367.18` | `-66.87` | `133.91` |

The held-out score distribution moved far beyond the aggregate preceding training distribution. Week 8 contained championship divisions and the finals field; week 10 was championship stress evidence. The event populations were much stronger than an average earlier-season event, and robots had also improved through the season.

The bias is not explained primarily by unseen teams. It remained negative for low-, medium-, and high-observation slices, including teams with many previous matches.

The total-score reconstruction is also not the main explanation. LatentStrat predicts auto and teleop phase points, then adds the training-only mean residual between official total and those phase targets. The observed total-minus-phase residual averaged `3.95`, `3.93`, and `7.92` points in weeks 6, 8, and 10. Those values are much smaller than the approximately 58-69 point forecast biases.

## Interpretation

### Confirmed

- The static model underpredicted out-of-time alliance scores in every evaluated week.
- The test score distribution shifted sharply relative to the preceding aggregate training distribution.
- A single static `Z_base` supplies no explicit week, event, or match-level adaptation.
- The main continuous heads supervise auto and teleop phases; total score and score differential are derived evaluation quantities rather than direct optimization targets.
- Interaction-model probability metrics overfit earlier than their score metrics.
- The heterogeneous training objective was not aligned with one common duration that preserved every headline metric.
- The predeclared development gate prevented unsupported three-seed test claims.

### Plausible contributors requiring controlled experiments

- Static team state cannot represent robot improvement or event-local configuration changes.
- Strong-event selection and later-season score inflation require context beyond historical average team state.
- Direct total-score or score-differential supervision may reduce shrinkage, but it could also duplicate phase targets or distort task balance.
- Winner and score objectives may need separate regularization, duration selection, or calibration treatment.
- Learned homoscedastic task weights may make the aggregate objective a poor checkpoint-selection signal even when they remain mathematically well behaved.

### Not supported by this evidence

- The model has too few parameters.
- SAB, PMA, or cross-alliance attention is inherently defective.
- The pre-2026 prior caused the score bias. On week-4 development, prior initialization beat random initialization on score-differential RMSE (`104.60` versus `128.33`), Brier score (`0.1824` versus `0.2530`), and log loss (`0.5428` versus `0.7273`).
- Teammate or opponent interactions improve formal out-of-time forecasting. The multi-seed test matrix never ran.
- Latent coordinates currently identify robot roles or causal compatibility effects.

## Evidence Bundle

The postmortem artifact contains:

- `run_inventory.csv`: every TensorBoard run directory present before cleanup.
- `recent_scalar_history.csv`: normalized key scalar histories for the static-reference and reliability studies.
- `development_key_epochs.csv`: all nine 40-epoch grid cells with comparable optimization and forecast metrics.
- `score_bias_slices.csv`: score bias and RMSE by model, week, event type, and observation slice.
- `tensorboard_tree_hashes.csv`: hashes and sizes for every deleted event file.
- `summary.json`: structured findings and validation counts.
- `manifest.json`: source and output hashes; `promotion_eligible=false`.

The exports reconcile with the saved development grid to numerical precision. They preserve all 145 selected TensorBoard scalar tags, 360 full-grid epoch rows, and the exact saved-prediction score biases. Raw TensorBoard replay was intentionally discarded after this export.

## Hold Point

LatentStrat is paused before another architecture experiment. No new event-state design, larger model, loss change, calibration rule, or checkpoint-selection policy should be promoted from this evidence alone.

The next planning session should first choose what the reference experiment is optimizing: calibrated winner probabilities, alliance scores, score differential, or an explicitly defined multi-objective rule. That decision must be made on development evidence before another formal test matrix is launched.

Statbotics is intentionally excluded from this postmortem while its API is unreliable. This does not weaken the development-gate conclusion because the rejected run never reached external test comparison.
