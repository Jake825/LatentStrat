# 2026 Static Championship Core Diagnostic

> Under a fixed 100-epoch, fixed-loss static `Z_base` model, does direct official-total supervision materially reduce Championship score bias without harming relative-strength or probability quality, and do teammate or opponent interaction modules add useful information beyond additive team strength?

## Executive Verdict

All six development and six refit runs completed. The tables below are fixed-100-epoch diagnostic evidence, not a best-achievable architecture ranking.

Direct official-total supervision was classified as `diagnostic-supported` for additive, `harmful` for teammate-set, `uncertain` for full-match.
Teammate interaction versus additive strength was `harmful` under phase-core, `harmful` under official-total-core; opponent interaction versus teammate-only was `uncertain` under phase-core, `uncertain` under official-total-core.
These labels answer the declared question only at epoch 100 under this static, one-seed contract.

| comparison                              | classification       |   score_rmse_relative_delta |   differential_rmse_relative_delta |   brier_delta |   log_loss_delta | leave_one_division_out_stable   |
|:----------------------------------------|:---------------------|----------------------------:|-----------------------------------:|--------------:|-----------------:|:--------------------------------|
| full-v-additive:official-total-core     | harmful              |                      0.1444 |                             0.1905 |        0.0692 |           0.8477 | True                            |
| full-v-additive:phase-core              | harmful              |                      0.1211 |                             0.0985 |        0.0642 |           0.7233 | True                            |
| full-v-teammate:official-total-core     | uncertain            |                      0.0007 |                             0.0013 |       -0.0049 |           0.3085 | False                           |
| full-v-teammate:phase-core              | uncertain            |                      0.0402 |                            -0.0021 |        0.0113 |           0.2221 | False                           |
| supervision:additive                    | diagnostic-supported |                     -0.0276 |                            -0.0197 |        0.0005 |           0.0062 | True                            |
| supervision:full-match                  | uncertain            |                     -0.0074 |                             0.0624 |        0.0056 |           0.1306 | False                           |
| supervision:teammate-set                | harmful              |                      0.0318 |                             0.0589 |        0.0218 |           0.0442 | True                            |
| teammate-v-additive:official-total-core | harmful              |                      0.1436 |                             0.1890 |        0.0741 |           0.5391 | True                            |
| teammate-v-additive:phase-core          | harmful              |                      0.0777 |                             0.1008 |        0.0529 |           0.5012 | True                            |

| supervision_mode    | architecture   |   alliance_score_mae |   alliance_score_rmse |   mean_bias |   relative_bias |   score_differential_mae |   score_differential_rmse |   winner_accuracy |   winner_brier |   winner_ece |   winner_log_loss |
|:--------------------|:---------------|---------------------:|----------------------:|------------:|----------------:|-------------------------:|--------------------------:|------------------:|---------------:|-------------:|------------------:|
| official-total-core | additive       |              99.5841 |              125.5032 |    -56.5942 |         -0.1302 |                 119.8242 |                  150.2029 |            0.7089 |         0.2206 |       0.1646 |            0.7963 |
| official-total-core | full-match     |             114.4335 |              143.6316 |    -57.0206 |         -0.1312 |                 143.4592 |                  178.8236 |            0.6709 |         0.2898 |       0.2657 |            1.6439 |
| official-total-core | teammate-set   |             115.0027 |              143.5245 |    -58.6227 |         -0.1349 |                 140.7082 |                  178.5920 |            0.6447 |         0.2946 |       0.2646 |            1.3354 |
| phase-core          | additive       |             102.0698 |              129.0688 |    -60.9063 |         -0.1401 |                 121.4525 |                  153.2157 |            0.7061 |         0.2200 |       0.1667 |            0.7900 |
| phase-core          | full-match     |             116.1010 |              144.7001 |    -66.9552 |         -0.1540 |                 135.1256 |                  168.3135 |            0.6664 |         0.2842 |       0.2676 |            1.5133 |
| phase-core          | teammate-set   |             111.3344 |              139.1029 |    -63.9010 |         -0.1470 |                 133.0434 |                  168.6651 |            0.6727 |         0.2729 |       0.2454 |            1.2912 |

## Data and Provenance

```json
{
  "championship_rows": 1119,
  "complete": true,
  "development_holdout_rows": 1642,
  "development_train_rows": 15373,
  "einstein_rows": 16,
  "elapsed_seconds": 9579.13143909999,
  "evidence_role": "reused-diagnostic-holdout",
  "failed_leaves": [],
  "filtering": {
    "excluded_missing_week": 0,
    "excluded_non_2026": 0,
    "excluded_non_official": 31,
    "excluded_unplayed": 0,
    "excluded_week_zero_or_nonstandard": 0,
    "included_rows": 18164,
    "input_rows": 18195
  },
  "final_train_rows": 17029,
  "frozen_final_contract_sha256": "e69f0cc457e2818cddd219f6e259b6a9b7edee426f0e6513fc8042ee35cf6df0",
  "promotion_eligible": false
}
```

## Development Runs

### official-total-core / additive

Completed 100 epochs. Final validation score RMSE was `107.308`, differential RMSE `117.110`, Brier `0.2048`, and log loss `0.7685`.
Training loss moved from `1.4444` to `0.4845`; mean `Z_base` update distance ended at `0.4094`.

- Best score_rmse occurred at epoch 9: `105.3616`; epoch 100: `107.3082`.
- Best score_differential_rmse occurred at epoch 13: `116.5721`; epoch 100: `117.1098`.
- Best winner_brier occurred at epoch 11: `0.1610`; epoch 100: `0.2048`.
- Best winner_log_loss occurred at epoch 12: `0.4860`; epoch 100: `0.7685`.

### official-total-core / full-match

Completed 100 epochs. Final validation score RMSE was `123.672`, differential RMSE `138.538`, Brier `0.2636`, and log loss `1.4522`.
Training loss moved from `0.9429` to `0.1229`; mean `Z_base` update distance ended at `0.2771`.

- Best score_rmse occurred at epoch 3: `107.1065`; epoch 100: `123.6717`.
- Best score_differential_rmse occurred at epoch 5: `121.1800`; epoch 100: `138.5384`.
- Best winner_brier occurred at epoch 3: `0.1580`; epoch 100: `0.2636`.
- Best winner_log_loss occurred at epoch 3: `0.4815`; epoch 100: `1.4522`.

### official-total-core / teammate-set

Completed 100 epochs. Final validation score RMSE was `120.854`, differential RMSE `135.499`, Brier `0.2489`, and log loss `1.2792`.
Training loss moved from `1.0134` to `0.1759`; mean `Z_base` update distance ended at `0.2701`.

- Best score_rmse occurred at epoch 5: `105.7649`; epoch 100: `120.8542`.
- Best score_differential_rmse occurred at epoch 4: `119.9640`; epoch 100: `135.4995`.
- Best winner_brier occurred at epoch 4: `0.1589`; epoch 100: `0.2489`.
- Best winner_log_loss occurred at epoch 4: `0.4809`; epoch 100: `1.2792`.

### phase-core / additive

Completed 100 epochs. Final validation score RMSE was `108.222`, differential RMSE `117.047`, Brier `0.2033`, and log loss `0.7644`.
Training loss moved from `1.4836` to `0.5426`; mean `Z_base` update distance ended at `0.4224`.

- Best score_rmse occurred at epoch 11: `106.4939`; epoch 100: `108.2217`.
- Best score_differential_rmse occurred at epoch 25: `116.7363`; epoch 100: `117.0474`.
- Best winner_brier occurred at epoch 10: `0.1590`; epoch 100: `0.2033`.
- Best winner_log_loss occurred at epoch 10: `0.4833`; epoch 100: `0.7644`.

### phase-core / full-match

Completed 100 epochs. Final validation score RMSE was `126.450`, differential RMSE `135.645`, Brier `0.2548`, and log loss `1.5253`.
Training loss moved from `1.0602` to `0.2055`; mean `Z_base` update distance ended at `0.2901`.

- Best score_rmse occurred at epoch 5: `110.9112`; epoch 100: `126.4504`.
- Best score_differential_rmse occurred at epoch 7: `120.3559`; epoch 100: `135.6450`.
- Best winner_brier occurred at epoch 3: `0.1603`; epoch 100: `0.2548`.
- Best winner_log_loss occurred at epoch 3: `0.4879`; epoch 100: `1.5253`.

### phase-core / teammate-set

Completed 100 epochs. Final validation score RMSE was `120.103`, differential RMSE `134.045`, Brier `0.2552`, and log loss `1.5108`.
Training loss moved from `1.1017` to `0.2503`; mean `Z_base` update distance ended at `0.2783`.

- Best score_rmse occurred at epoch 5: `106.6093`; epoch 100: `120.1027`.
- Best score_differential_rmse occurred at epoch 5: `121.5485`; epoch 100: `134.0453`.
- Best winner_brier occurred at epoch 4: `0.1589`; epoch 100: `0.2552`.
- Best winner_log_loss occurred at epoch 4: `0.4812`; epoch 100: `1.5108`.

## Final Refit Runs

### phase-core / additive

At epoch 100: score RMSE `129.069`, mean bias `-60.906`, relative bias `-14.013%`, differential RMSE `153.216`, Brier `0.2200`, and log loss `0.7900`.

### phase-core / teammate-set

At epoch 100: score RMSE `139.103`, mean bias `-63.901`, relative bias `-14.702%`, differential RMSE `168.665`, Brier `0.2729`, and log loss `1.2912`.

### phase-core / full-match

At epoch 100: score RMSE `144.700`, mean bias `-66.955`, relative bias `-15.404%`, differential RMSE `168.314`, Brier `0.2842`, and log loss `1.5133`.

### official-total-core / additive

At epoch 100: score RMSE `125.503`, mean bias `-56.594`, relative bias `-13.021%`, differential RMSE `150.203`, Brier `0.2206`, and log loss `0.7963`.

### official-total-core / teammate-set

At epoch 100: score RMSE `143.525`, mean bias `-58.623`, relative bias `-13.487%`, differential RMSE `178.592`, Brier `0.2946`, and log loss `1.3354`.

### official-total-core / full-match

At epoch 100: score RMSE `143.632`, mean bias `-57.021`, relative bias `-13.119%`, differential RMSE `178.824`, Brier `0.2898`, and log loss `1.6439`.

## What 100 Epochs Changed

This table separates continued loss reduction from the epoch of each development forecast optimum. An optimum before epoch 100 is evidence of fixed-horizon sensitivity, not proof that the architecture is intrinsically harmful.

| supervision_mode    | architecture   |   first_train_loss |   final_train_loss |   final_z_base_update |   best_score_epoch |   best_score |   final_score |   best_differential_epoch |   best_differential |   final_differential |   best_brier_epoch |   best_brier |   final_brier |   best_log_loss_epoch |   best_log_loss |   final_log_loss |
|:--------------------|:---------------|-------------------:|-------------------:|----------------------:|-------------------:|-------------:|--------------:|--------------------------:|--------------------:|---------------------:|-------------------:|-------------:|--------------:|----------------------:|----------------:|-----------------:|
| official-total-core | additive       |             1.4444 |             0.4845 |                0.4094 |                  9 |     105.3616 |      107.3082 |                        13 |            116.5721 |             117.1098 |                 11 |       0.1610 |        0.2048 |                    12 |          0.4860 |           0.7685 |
| official-total-core | full-match     |             0.9429 |             0.1229 |                0.2771 |                  3 |     107.1065 |      123.6717 |                         5 |            121.1800 |             138.5384 |                  3 |       0.1580 |        0.2636 |                     3 |          0.4815 |           1.4522 |
| official-total-core | teammate-set   |             1.0134 |             0.1759 |                0.2701 |                  5 |     105.7649 |      120.8542 |                         4 |            119.9640 |             135.4995 |                  4 |       0.1589 |        0.2489 |                     4 |          0.4809 |           1.2792 |
| phase-core          | additive       |             1.4836 |             0.5426 |                0.4224 |                 11 |     106.4939 |      108.2217 |                        25 |            116.7363 |             117.0474 |                 10 |       0.1590 |        0.2033 |                    10 |          0.4833 |           0.7644 |
| phase-core          | full-match     |             1.0602 |             0.2055 |                0.2901 |                  5 |     110.9112 |      126.4504 |                         7 |            120.3559 |             135.6450 |                  3 |       0.1603 |        0.2548 |                     3 |          0.4879 |           1.5253 |
| phase-core          | teammate-set   |             1.1017 |             0.2503 |                0.2783 |                  5 |     106.6093 |      120.1027 |                         5 |            121.5485 |             134.0453 |                  4 |       0.1589 |        0.2552 |                     4 |          0.4812 |           1.5108 |

### Direct answers about the long horizon

- Direct totals made mean bias less negative in all three architectures, but only the additive comparison met the full support rule: its bias changed from `-60.9` to `-56.6` points. The remaining bias is still material.
- Extra epochs mostly reduced training loss after forecast quality had stopped improving. Development score RMSE reached its optimum between epochs 3 and 11, never at epoch 100.
- Relative-strength estimates did not keep improving through epoch 100: score-differential RMSE reached its development optimum between epochs 4 and 25.
- Probability quality worsened after an earlier optimum. Brier and log-loss optima occurred between epochs 3 and 12, while every epoch-100 value was worse.
- `Z_base` changed measurably: mean update distance ended between `0.270` and `0.422` across development leaves. This quantifies movement, not semantic meaning or useful adaptation.
- The long horizon did not help the interaction architectures at the formal checkpoint. Teammate models were harmful versus additive strength in both supervision modes; opponent interaction was uncertain versus teammate-only and harmful versus additive. This is fixed-horizon sensitivity, not an intrinsic architecture ranking.

## Raw and Transferred Calibration

Platt and score-derived mappings were fitted on the sealed development holdout and transferred to separately initialized full pre-Championship refits. They are not guaranteed Championship calibration.

| supervision_mode    | architecture   | metric          |   raw_pred_red_win_probability |   transferred_platt_red_win_probability |   transferred_score_red_win_probability |
|:--------------------|:---------------|:----------------|-------------------------------:|----------------------------------------:|----------------------------------------:|
| official-total-core | additive       | winner_brier    |                        0.22056 |                                 0.18902 |                                 0.18358 |
| official-total-core | additive       | winner_ece      |                        0.16460 |                                 0.02881 |                                 0.07162 |
| official-total-core | additive       | winner_log_loss |                        0.79628 |                                 0.55658 |                                 0.54573 |
| official-total-core | full-match     | winner_brier    |                        0.28977 |                                 0.21263 |                                 0.21051 |
| official-total-core | full-match     | winner_ece      |                        0.26569 |                                 0.05760 |                                 0.06848 |
| official-total-core | full-match     | winner_log_loss |                        1.64393 |                                 0.61244 |                                 0.61892 |
| official-total-core | teammate-set   | winner_brier    |                        0.29465 |                                 0.21563 |                                 0.21176 |
| official-total-core | teammate-set   | winner_ece      |                        0.26464 |                                 0.05324 |                                 0.09277 |
| official-total-core | teammate-set   | winner_log_loss |                        1.33539 |                                 0.61745 |                                 0.62690 |
| phase-core          | additive       | winner_brier    |                        0.22002 |                                 0.18892 |                                 0.18623 |
| phase-core          | additive       | winner_ece      |                        0.16667 |                                 0.02849 |                                 0.07444 |
| phase-core          | additive       | winner_log_loss |                        0.79003 |                                 0.55616 |                                 0.55513 |
| phase-core          | full-match     | winner_brier    |                        0.28422 |                                 0.20210 |                                 0.19979 |
| phase-core          | full-match     | winner_ece      |                        0.26759 |                                 0.04295 |                                 0.06861 |
| phase-core          | full-match     | winner_log_loss |                        1.51330 |                                 0.58743 |                                 0.59341 |
| phase-core          | teammate-set   | winner_brier    |                        0.27289 |                                 0.20696 |                                 0.20121 |
| phase-core          | teammate-set   | winner_ece      |                        0.24543 |                                 0.03513 |                                 0.06164 |
| phase-core          | teammate-set   | winner_log_loss |                        1.29122 |                                 0.60132 |                                 0.59840 |

## Frozen Baselines and Historical Context

Mean, ridge, and rolling pRidge controls use the same non-DQ training rows and exact Championship keys. The historical static reference is descriptive only.

| supervision_mode                  | architecture        |   alliance_score_mae |   alliance_score_rmse |   mean_bias |   relative_bias |   score_differential_mae |   score_differential_rmse |   winner_accuracy |   winner_brier |   winner_ece |   winner_log_loss |
|:----------------------------------|:--------------------|---------------------:|----------------------:|------------:|----------------:|-------------------------:|--------------------------:|------------------:|---------------:|-------------:|------------------:|
| baseline                          | direct-total-pridge |             97.65573 |             123.60487 |   -54.93888 |        -0.12640 |                115.31310 |                 146.41961 |           0.73508 |        0.17856 |      0.05396 |           0.53206 |
| baseline                          | direct-total-ridge  |            102.63430 |             129.44265 |   -63.51699 |        -0.14613 |                118.44233 |                 149.81808 |           0.72242 |        0.18462 |      0.06607 |           0.55071 |
| baseline                          | mean-total          |            258.88539 |             296.65003 |  -254.71436 |        -0.58602 |                164.66697 |                 207.94797 |           0.52260 |        0.24951 |      0.00416 |           0.69216 |
| historical-uncontrolled-reference | additive            |            122.03039 |             150.75031 |   -95.21673 |        -0.21906 |                125.85736 |                 158.05947 |           0.68445 |        0.19970 |      0.11793 |           0.58131 |
| historical-uncontrolled-reference | full-match          |            106.48046 |             133.82217 |   -67.31605 |        -0.15487 |                126.73970 |                 157.00495 |           0.71248 |        0.18502 |      0.08490 |           0.55585 |
| historical-uncontrolled-reference | mean                |            260.23089 |             297.82222 |  -256.09806 |        -0.58920 |                164.59515 |                 207.85203 |           0.52260 |        0.24955 |      0.00786 |           0.69225 |
| historical-uncontrolled-reference | ridge               |            116.14230 |             147.83702 |   -65.49319 |        -0.15068 |                131.85780 |                 168.15291 |           0.69801 |        0.24088 |      0.20796 |           0.98124 |
| historical-uncontrolled-reference | rolling-pridge      |            112.40348 |             142.98910 |   -53.87237 |        -0.12394 |                131.92785 |                 168.09933 |           0.69982 |        0.23712 |      0.19263 |           0.88498 |
| historical-uncontrolled-reference | teammate-set        |            108.87114 |             137.00455 |   -73.48707 |        -0.16907 |                124.71143 |                 155.17222 |           0.66094 |        0.20235 |      0.12907 |           0.58281 |

## Bias and Availability Slices

| supervision_mode    | architecture   | scope              | metric              |      value |   count |
|:--------------------|:---------------|:-------------------|:--------------------|-----------:|--------:|
| official-total-core | additive       | DQ-only            | alliance_score_rmse |  153.57906 |      22 |
| official-total-core | additive       | DQ-only            | mean_bias           |  -21.14252 |      22 |
| official-total-core | additive       | DQ-only            | relative_bias       |   -0.05669 |      22 |
| official-total-core | additive       | surrogate-only     | alliance_score_rmse |   89.88193 |      12 |
| official-total-core | additive       | surrogate-only     | mean_bias           |  -42.61564 |      12 |
| official-total-core | additive       | surrogate-only     | relative_bias       |   -0.11014 |      12 |
| official-total-core | additive       | observation:high   | alliance_score_rmse |  124.57784 |    2134 |
| official-total-core | additive       | observation:high   | mean_bias           |  -54.74078 |    2134 |
| official-total-core | additive       | observation:high   | relative_bias       |   -0.12542 |    2134 |
| official-total-core | additive       | observation:low    | alliance_score_rmse |  121.69788 |       4 |
| official-total-core | additive       | observation:low    | mean_bias           |  -82.59586 |       4 |
| official-total-core | additive       | observation:low    | relative_bias       |   -0.21996 |       4 |
| official-total-core | additive       | observation:medium | alliance_score_rmse |  149.83625 |     100 |
| official-total-core | additive       | observation:medium | mean_bias           |  -87.30558 |     100 |
| official-total-core | additive       | observation:medium | relative_bias       |   -0.22683 |     100 |
| official-total-core | full-match     | DQ-only            | alliance_score_rmse |  175.53550 |      22 |
| official-total-core | full-match     | DQ-only            | mean_bias           |  -12.82586 |      22 |
| official-total-core | full-match     | DQ-only            | relative_bias       |   -0.03439 |      22 |
| official-total-core | full-match     | surrogate-only     | alliance_score_rmse |  102.60720 |      12 |
| official-total-core | full-match     | surrogate-only     | mean_bias           |  -47.07334 |      12 |
| official-total-core | full-match     | surrogate-only     | relative_bias       |   -0.12166 |      12 |
| official-total-core | full-match     | observation:high   | alliance_score_rmse |  142.38610 |    2134 |
| official-total-core | full-match     | observation:high   | mean_bias           |  -56.00170 |    2134 |
| official-total-core | full-match     | observation:high   | relative_bias       |   -0.12831 |    2134 |
| official-total-core | full-match     | observation:low    | alliance_score_rmse |  191.93078 |       4 |
| official-total-core | full-match     | observation:low    | mean_bias           |   18.35640 |       4 |
| official-total-core | full-match     | observation:low    | relative_bias       |    0.04889 |       4 |
| official-total-core | full-match     | observation:medium | alliance_score_rmse |  172.69588 |     100 |
| official-total-core | full-match     | observation:medium | mean_bias           |  -72.05593 |     100 |
| official-total-core | full-match     | observation:medium | relative_bias       |   -0.18721 |     100 |
| official-total-core | teammate-set   | DQ-only            | alliance_score_rmse |  175.47606 |      22 |
| official-total-core | teammate-set   | DQ-only            | mean_bias           |  -27.56809 |      22 |
| official-total-core | teammate-set   | DQ-only            | relative_bias       |   -0.07392 |      22 |
| official-total-core | teammate-set   | surrogate-only     | alliance_score_rmse |  122.61303 |      12 |
| official-total-core | teammate-set   | surrogate-only     | mean_bias           |  -54.25674 |      12 |
| official-total-core | teammate-set   | surrogate-only     | relative_bias       |   -0.14023 |      12 |
| official-total-core | teammate-set   | observation:high   | alliance_score_rmse |  143.09884 |    2134 |
| official-total-core | teammate-set   | observation:high   | mean_bias           |  -57.91606 |    2134 |
| official-total-core | teammate-set   | observation:high   | relative_bias       |   -0.13270 |    2134 |
| official-total-core | teammate-set   | observation:low    | alliance_score_rmse |  112.17007 |       4 |
| official-total-core | teammate-set   | observation:low    | mean_bias           |   49.65208 |       4 |
| official-total-core | teammate-set   | observation:low    | relative_bias       |    0.13223 |       4 |
| official-total-core | teammate-set   | observation:medium | alliance_score_rmse |  160.51770 |     100 |
| official-total-core | teammate-set   | observation:medium | mean_bias           |  -71.20109 |     100 |
| official-total-core | teammate-set   | observation:medium | relative_bias       |   -0.18499 |     100 |
| phase-core          | additive       | DQ-only            | alliance_score_rmse |  154.73500 |      22 |
| phase-core          | additive       | DQ-only            | mean_bias           |  -22.57318 |      22 |
| phase-core          | additive       | DQ-only            | relative_bias       |   -0.06053 |      22 |
| phase-core          | additive       | surrogate-only     | alliance_score_rmse |   92.83436 |      12 |
| phase-core          | additive       | surrogate-only     | mean_bias           |  -45.40957 |      12 |
| phase-core          | additive       | surrogate-only     | relative_bias       |   -0.11736 |      12 |
| phase-core          | additive       | observation:high   | alliance_score_rmse |  127.45450 |    2134 |
| phase-core          | additive       | observation:high   | mean_bias           |  -58.62676 |    2134 |
| phase-core          | additive       | observation:high   | relative_bias       |   -0.13432 |    2134 |
| phase-core          | additive       | observation:low    | alliance_score_rmse |  135.66972 |       4 |
| phase-core          | additive       | observation:low    | mean_bias           | -104.03118 |       4 |
| phase-core          | additive       | observation:low    | relative_bias       |   -0.27705 |       4 |
| phase-core          | additive       | observation:medium | alliance_score_rmse |  164.40324 |     100 |
| phase-core          | additive       | observation:medium | mean_bias           |  -99.39359 |     100 |
| phase-core          | additive       | observation:medium | relative_bias       |   -0.25823 |     100 |
| phase-core          | full-match     | DQ-only            | alliance_score_rmse |  182.99598 |      22 |
| phase-core          | full-match     | DQ-only            | mean_bias           |  -46.87036 |      22 |
| phase-core          | full-match     | DQ-only            | relative_bias       |   -0.12567 |      22 |
| phase-core          | full-match     | surrogate-only     | alliance_score_rmse |  120.09336 |      12 |
| phase-core          | full-match     | surrogate-only     | mean_bias           |  -67.53400 |      12 |
| phase-core          | full-match     | surrogate-only     | relative_bias       |   -0.17454 |      12 |
| phase-core          | full-match     | observation:high   | alliance_score_rmse |  143.91532 |    2134 |
| phase-core          | full-match     | observation:high   | mean_bias           |  -66.29282 |    2134 |
| phase-core          | full-match     | observation:high   | relative_bias       |   -0.15189 |    2134 |
| phase-core          | full-match     | observation:low    | alliance_score_rmse |  198.94465 |       4 |
| phase-core          | full-match     | observation:low    | mean_bias           |  -14.37369 |       4 |
| phase-core          | full-match     | observation:low    | relative_bias       |   -0.03828 |       4 |
| phase-core          | full-match     | observation:medium | alliance_score_rmse |  166.69309 |     100 |
| phase-core          | full-match     | observation:medium | mean_bias           |  -78.77428 |     100 |
| phase-core          | full-match     | observation:medium | relative_bias       |   -0.20466 |     100 |
| phase-core          | teammate-set   | DQ-only            | alliance_score_rmse |  186.03780 |      22 |
| phase-core          | teammate-set   | DQ-only            | mean_bias           |  -35.75765 |      22 |
| phase-core          | teammate-set   | DQ-only            | relative_bias       |   -0.09588 |      22 |
| phase-core          | teammate-set   | surrogate-only     | alliance_score_rmse |  105.56217 |      12 |
| phase-core          | teammate-set   | surrogate-only     | mean_bias           |  -30.26735 |      12 |
| phase-core          | teammate-set   | surrogate-only     | relative_bias       |   -0.07823 |      12 |
| phase-core          | teammate-set   | observation:high   | alliance_score_rmse |  139.02891 |    2134 |
| phase-core          | teammate-set   | observation:high   | mean_bias           |  -62.94775 |    2134 |
| phase-core          | teammate-set   | observation:high   | relative_bias       |   -0.14422 |    2134 |
| phase-core          | teammate-set   | observation:low    | alliance_score_rmse |  169.10763 |       4 |
| phase-core          | teammate-set   | observation:low    | mean_bias           |   -2.08931 |       4 |
| phase-core          | teammate-set   | observation:low    | relative_bias       |   -0.00556 |       4 |
| phase-core          | teammate-set   | observation:medium | alliance_score_rmse |  150.91816 |     100 |
| phase-core          | teammate-set   | observation:medium | mean_bias           |  -80.52409 |     100 |
| phase-core          | teammate-set   | observation:medium | relative_bias       |   -0.20921 |     100 |

## Paired Comparisons and Sensitivity

| comparison                              | left                             | right                            | metric                     |   delta_right_minus_left |   left_value |   right_value |   relative_delta |      ci_low |     ci_high |   probability_improvement |
|:----------------------------------------|:---------------------------------|:---------------------------------|:---------------------------|-------------------------:|-------------:|--------------:|-----------------:|------------:|------------:|--------------------------:|
| supervision:additive                    | phase-core:additive              | official-total-core:additive     | score_squared_error        |               -907.70168 |    129.06881 |     125.50321 |         -0.02763 | -1898.21786 |  -272.31073 |                   1.00000 |
| supervision:additive                    | phase-core:additive              | official-total-core:additive     | differential_squared_error |               -914.13349 |    153.21570 |     150.20292 |         -0.01966 | -2628.34222 |    60.56229 |                   0.92460 |
| supervision:additive                    | phase-core:additive              | official-total-core:additive     | winner_brier               |                  0.00054 |      0.22002 |       0.22056 |          0.00245 |    -0.00111 |     0.00193 |                   0.23580 |
| supervision:additive                    | phase-core:additive              | official-total-core:additive     | winner_log_loss            |                  0.00625 |      0.79003 |       0.79628 |          0.00791 |    -0.00760 |     0.01661 |                   0.16100 |
| supervision:teammate-set                | phase-core:teammate-set          | official-total-core:teammate-set | score_squared_error        |               1249.66008 |    139.10292 |     143.52450 |          0.03179 |   324.45115 |  2138.03602 |                   0.00340 |
| supervision:teammate-set                | phase-core:teammate-set          | official-total-core:teammate-set | differential_squared_error |               3447.19726 |    168.66507 |     178.59201 |          0.05886 |  1819.40503 |  4967.09876 |                   0.00000 |
| supervision:teammate-set                | phase-core:teammate-set          | official-total-core:teammate-set | winner_brier               |                  0.02176 |      0.27289 |       0.29465 |          0.07975 |     0.00463 |     0.03888 |                   0.00520 |
| supervision:teammate-set                | phase-core:teammate-set          | official-total-core:teammate-set | winner_log_loss            |                  0.04416 |      1.29122 |       1.33539 |          0.03420 |    -0.05776 |     0.16793 |                   0.21780 |
| supervision:full-match                  | phase-core:full-match            | official-total-core:full-match   | score_squared_error        |               -308.07593 |    144.70006 |     143.63158 |         -0.00738 | -1335.62335 |   667.18839 |                   0.71200 |
| supervision:full-match                  | phase-core:full-match            | official-total-core:full-match   | differential_squared_error |               3648.43924 |    168.31350 |     178.82358 |          0.06244 |   898.68869 |  6149.14463 |                   0.00340 |
| supervision:full-match                  | phase-core:full-match            | official-total-core:full-match   | winner_brier               |                  0.00555 |      0.28422 |       0.28977 |          0.01953 |    -0.02027 |     0.03065 |                   0.36360 |
| supervision:full-match                  | phase-core:full-match            | official-total-core:full-match   | winner_log_loss            |                  0.13063 |      1.51330 |       1.64393 |          0.08632 |    -0.06646 |     0.31262 |                   0.09800 |
| teammate-v-additive:phase-core          | phase-core:additive              | phase-core:teammate-set          | score_squared_error        |               2690.86665 |    129.06881 |     139.10292 |          0.07774 |   580.98764 |  4496.21834 |                   0.00600 |
| teammate-v-additive:phase-core          | phase-core:additive              | phase-core:teammate-set          | differential_squared_error |               4972.85768 |    153.21570 |     168.66507 |          0.10083 |  2286.18085 |  7069.37205 |                   0.00000 |
| teammate-v-additive:phase-core          | phase-core:additive              | phase-core:teammate-set          | winner_brier               |                  0.05287 |      0.22002 |       0.27289 |          0.24030 |     0.03307 |     0.07261 |                   0.00000 |
| teammate-v-additive:phase-core          | phase-core:additive              | phase-core:teammate-set          | winner_log_loss            |                  0.50119 |      0.79003 |       1.29122 |          0.63439 |     0.35093 |     0.64210 |                   0.00000 |
| full-v-teammate:phase-core              | phase-core:teammate-set          | phase-core:full-match            | score_squared_error        |               1588.48416 |    139.10292 |     144.70006 |          0.04024 |   619.48978 |  2468.68903 |                   0.00140 |
| full-v-teammate:phase-core              | phase-core:teammate-set          | phase-core:full-match            | differential_squared_error |               -118.47311 |    168.66507 |     168.31350 |         -0.00208 | -2560.07261 |  2401.69015 |                   0.53420 |
| full-v-teammate:phase-core              | phase-core:teammate-set          | phase-core:full-match            | winner_brier               |                  0.01133 |      0.27289 |       0.28422 |          0.04153 |    -0.01945 |     0.04168 |                   0.23360 |
| full-v-teammate:phase-core              | phase-core:teammate-set          | phase-core:full-match            | winner_log_loss            |                  0.22208 |      1.29122 |       1.51330 |          0.17199 |    -0.03058 |     0.48934 |                   0.04420 |
| full-v-additive:phase-core              | phase-core:additive              | phase-core:full-match            | score_squared_error        |               4279.35081 |    129.06881 |     144.70006 |          0.12111 |  1777.23797 |  6230.27484 |                   0.00020 |
| full-v-additive:phase-core              | phase-core:additive              | phase-core:full-match            | differential_squared_error |               4854.38457 |    153.21570 |     168.31350 |          0.09854 |  1331.45570 |  7564.09451 |                   0.00380 |
| full-v-additive:phase-core              | phase-core:additive              | phase-core:full-match            | winner_brier               |                  0.06420 |      0.22002 |       0.28422 |          0.29181 |     0.03813 |     0.08992 |                   0.00000 |
| full-v-additive:phase-core              | phase-core:additive              | phase-core:full-match            | winner_log_loss            |                  0.72327 |      0.79003 |       1.51330 |          0.91549 |     0.56574 |     0.88257 |                   0.00000 |
| teammate-v-additive:official-total-core | official-total-core:additive     | official-total-core:teammate-set | score_squared_error        |               4848.22840 |    125.50321 |     143.52450 |          0.14359 |  3776.38930 |  5866.43834 |                   0.00000 |
| teammate-v-additive:official-total-core | official-total-core:additive     | official-total-core:teammate-set | differential_squared_error |               9334.18842 |    150.20292 |     178.59201 |          0.18900 |  7340.78436 | 11071.63602 |                   0.00000 |
| teammate-v-additive:official-total-core | official-total-core:additive     | official-total-core:teammate-set | winner_brier               |                  0.07409 |      0.22056 |       0.29465 |          0.33594 |     0.05702 |     0.09287 |                   0.00000 |
| teammate-v-additive:official-total-core | official-total-core:additive     | official-total-core:teammate-set | winner_log_loss            |                  0.53910 |      0.79628 |       1.33539 |          0.67703 |     0.38478 |     0.68946 |                   0.00000 |
| full-v-teammate:official-total-core     | official-total-core:teammate-set | official-total-core:full-match   | score_squared_error        |                 30.74816 |    143.52450 |     143.63158 |          0.00075 |  -901.19506 |   984.46153 |                   0.46780 |
| full-v-teammate:official-total-core     | official-total-core:teammate-set | official-total-core:full-match   | differential_squared_error |                 82.76887 |    178.59201 |     178.82358 |          0.00130 | -2471.64599 |  2451.51994 |                   0.46620 |
| full-v-teammate:official-total-core     | official-total-core:teammate-set | official-total-core:full-match   | winner_brier               |                 -0.00488 |      0.29465 |       0.28977 |         -0.01656 |    -0.02325 |     0.01747 |                   0.69380 |
| full-v-teammate:official-total-core     | official-total-core:teammate-set | official-total-core:full-match   | winner_log_loss            |                  0.30855 |      1.33539 |       1.64393 |          0.23105 |     0.15622 |     0.48221 |                   0.00000 |
| full-v-additive:official-total-core     | official-total-core:additive     | official-total-core:full-match   | score_squared_error        |               4878.97656 |    125.50321 |     143.63158 |          0.14445 |  3569.00503 |  5892.81441 |                   0.00000 |
| full-v-additive:official-total-core     | official-total-core:additive     | official-total-core:full-match   | differential_squared_error |               9416.95729 |    150.20292 |     178.82358 |          0.19055 |  7365.68329 | 11436.60896 |                   0.00000 |
| full-v-additive:official-total-core     | official-total-core:additive     | official-total-core:full-match   | winner_brier               |                  0.06921 |      0.22056 |       0.28977 |          0.31382 |     0.05622 |     0.08153 |                   0.00000 |
| full-v-additive:official-total-core     | official-total-core:additive     | official-total-core:full-match   | winner_log_loss            |                  0.84765 |      0.79628 |       1.64393 |          1.06451 |     0.75647 |     0.93999 |                   0.00000 |

Every classification also passed through the saved leave-one-division-out veto. See `leave_one_division_out.csv` for all omitted-division rows.

## Findings, Explanations, and Open Questions

### Confirmed by this diagnostic

- All final comparisons use the same exact primary-clean Championship match set.
- The predeclared classifications above describe supervision and interaction effects at the fixed 100-epoch horizon.
- The development table records whether score, relative-strength, or probability metrics peaked before epoch 100 even while training loss continued downward.
- The recorded `Z_base` update distances quantify state movement without assigning semantic meaning to latent coordinates.

### Plausible explanations requiring another experiment

- Remaining score bias may reflect static state, target construction, distribution shift, or optimization. This matrix isolates supervision and interaction only.
- A transferred calibrator can shift after the full-data refit; raw and transferred probabilities must therefore be interpreted together.
- Any interaction benefit may depend on training duration because the architectures have different development optima.

### Unanswered

- Whether explicit week or event context reduces the remaining out-of-time bias.
- Whether the fixed-horizon effects reproduce across seeds.
- Each architecture's best achievable performance under a separately sealed selection protocol.
- Whether these forecast changes improve alliance-selection decisions or tournament outcomes.

## Interpretation Boundaries

- This study does not compare static and temporal state.
- Championship divisions are reused diagnostic evidence.
- One seed cannot establish optimization reliability.
- Earlier development optima are diagnostic only; Championship predictions use epoch 100 exclusively.

## Next Decision

Use official-total additive as the fixed static control for one next experiment. Compare it against one minimal week/event-context model while holding the exact split, fixed loss, initialization, and optimizer contract constant. Treat 100 epochs as the maximum trajectory, select duration only on a sealed non-Championship development block, refit for that frozen duration, and evaluate Championship once. Run at least three seeds before any promotion claim. Persistent bias motivates this test but does not prove a temporal model will solve it.
