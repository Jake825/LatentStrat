import numpy as np
import pandas as pd

from latentstrat.config import default_options
from latentstrat.data import Split, TargetStats, apply_target_stats, fit_target_stats
from latentstrat.evaluation import (
    _predict_batched,
    common_match_metrics_from_predictions,
    evaluate_model,
)
from latentstrat.model import init_model
from latentstrat.training import match_team_matrices


def _evaluation_table(row_count=7):
    rows = []
    for idx in range(row_count):
        red = [idx % 8 + 1, (idx + 1) % 8 + 1, (idx + 2) % 8 + 1]
        blue = [(idx + 3) % 8 + 1, (idx + 4) % 8 + 1, (idx + 5) % 8 + 1]
        rows.append(
            {
                "event_key": "2026eval",
                "match_key": f"2026eval_qm{idx + 1}",
                "red_team_1_key": f"frc{red[0]}",
                "red_team_2_key": f"frc{red[1]}",
                "red_team_3_key": f"frc{red[2]}",
                "blue_team_1_key": f"frc{blue[0]}",
                "blue_team_2_key": f"frc{blue[1]}",
                "blue_team_3_key": f"frc{blue[2]}",
                "red_team_1_idx": red[0],
                "red_team_2_idx": red[1],
                "red_team_3_idx": red[2],
                "blue_team_1_idx": blue[0],
                "blue_team_2_idx": blue[1],
                "blue_team_3_idx": blue[2],
                "red_total_score": 100 + idx,
                "blue_total_score": 90 + 2 * idx,
                "red_auto_pts": 20 + idx % 3,
                "red_teleop_pts": 80 + idx,
                "blue_auto_pts": 18 + idx % 3,
                "blue_teleop_pts": 72 + 2 * idx,
                "win_margin": 10 - idx,
                "red_foul_pts": idx % 4,
                "blue_foul_pts": (idx + 1) % 3,
                "fouls_drawn": (idx % 4) - ((idx + 1) % 3),
                "red_win": idx % 2 == 0,
                "sort_ordinal": idx + 1,
                "event_week": 0,
            }
        )
    table = pd.DataFrame(rows)
    for color in ("red", "blue"):
        for slot in (1, 2, 3):
            table[f"{color}_team_{slot}_base_idx"] = table[f"{color}_team_{slot}_idx"]
            table[f"{color}_team_{slot}_event_idx"] = table[f"{color}_team_{slot}_idx"]
            table[f"{color}_team_{slot}_missing_team_mask"] = False
            table[f"{color}_team_{slot}_endgame_status"] = "Level1"
    for column in [name for name in table.columns if name.endswith("_idx")]:
        table[column] = table[column].astype("Int64")
    return table


def _prepared_context(eval_batch_size=2):
    table = _evaluation_table()
    opts = default_options().model_copy(update={"eval_batch_size": eval_batch_size})
    validation_mask = np.zeros(len(table), dtype=bool)
    validation_mask[-2:] = True
    split = Split(
        train_mask=~validation_mask,
        validation_mask=validation_mask,
        test_mask=np.zeros(len(table), dtype=bool),
        policy="fixture",
    )
    stats = fit_target_stats(table, split.train_mask, opts)
    prepared = apply_target_stats(table, stats)
    model = init_model(9, opts.latent_dim, len(opts.target_map), len(opts.binary_targets), opts)
    return prepared, split, stats, model


def test_batched_evaluation_matches_large_batch_and_keeps_gradients_empty():
    prepared, split, stats, model = _prepared_context()
    small_batch_opts = default_options().model_copy(update={"eval_batch_size": 2})
    large_batch_opts = default_options().model_copy(update={"eval_batch_size": 128})
    for parameter in model.parameters():
        parameter.grad = None

    small = evaluate_model(model, prepared, split, stats, small_batch_opts)
    large = evaluate_model(model, prepared, split, stats, large_batch_opts)

    pd.testing.assert_frame_equal(
        small.continuous_metrics, large.continuous_metrics, check_exact=False, atol=1e-6
    )
    pd.testing.assert_frame_equal(
        small.binary_metrics, large.binary_metrics, check_exact=False, atol=1e-6
    )
    pd.testing.assert_frame_equal(
        small.endgame_metrics, large.endgame_metrics, check_exact=False, atol=1e-6
    )
    pd.testing.assert_frame_equal(
        small.award_metrics, large.award_metrics, check_exact=False, atol=1e-6
    )
    pd.testing.assert_frame_equal(
        small.zero_out_diagnostics,
        large.zero_out_diagnostics,
        check_exact=False,
        atol=2e-6,
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_predict_batched_preserves_rows_and_attention_shapes_across_chunks():
    prepared, _, _, model = _prepared_context()
    opts = default_options().model_copy(update={"eval_batch_size": 2})
    red, blue = match_team_matrices(prepared)

    cont_z, bin_logits, red_pma, blue_pma = _predict_batched(model, red, blue, opts)

    assert cont_z.shape == (len(prepared), len(opts.target_map))
    assert bin_logits.shape == (len(prepared), len(opts.binary_targets))
    assert red_pma.shape == (len(prepared), 1, 3)
    assert blue_pma.shape == (len(prepared), 1, 3)
    np.testing.assert_allclose(red_pma.sum(axis=2), np.ones((len(prepared), 1)), rtol=1e-6)


def _logit(probability: float) -> float:
    return float(np.log(probability / (1 - probability)))


def test_common_match_metrics_use_red_win_orientation_and_score_math():
    opts = default_options()
    table = pd.DataFrame(
        {
            "red_auto_pts": [20.0, 10.0],
            "red_teleop_pts": [80.0, 50.0],
            "blue_auto_pts": [15.0, 25.0],
            "blue_teleop_pts": [75.0, 75.0],
            "red_total_score": [105.0, 63.0],
            "blue_total_score": [92.0, 104.0],
            "red_win": [1.0, 0.0],
        }
    )
    split = Split(
        train_mask=np.array([False, False]),
        validation_mask=np.array([True, True]),
        test_mask=np.array([False, False]),
        policy="fixture",
    )
    target_stats = TargetStats(
        target_names=["red_auto_pts", "red_teleop_pts", "blue_auto_pts", "blue_teleop_pts"],
        mu=np.zeros(4),
        sigma=np.ones(4),
        is_constant=np.zeros(4, dtype=bool),
    )
    foul_names = [f"{color}_{target}" for color in ("red", "blue") for target in opts.foul_targets]
    v57_stats = TargetStats(
        target_names=foul_names,
        mu=np.zeros(len(foul_names)),
        sigma=np.ones(len(foul_names)),
        is_constant=np.zeros(len(foul_names), dtype=bool),
    )
    pred_cont = table[target_stats.target_names].to_numpy(dtype=float)
    pred_foul = np.zeros((len(table), len(foul_names)), dtype=float)
    pred_foul[:, foul_names.index("red_committed_foul_pts")] = [2.0, 4.0]
    pred_foul[:, foul_names.index("blue_committed_foul_pts")] = [5.0, 3.0]
    pred_bin = np.array([[_logit(0.99)], [_logit(0.01)]], dtype=float)

    metrics = common_match_metrics_from_predictions(
        table,
        split,
        target_stats,
        opts,
        pred_cont,
        pred_bin,
        v57_target_stats=v57_stats,
        pred_foul_z=pred_foul,
    )

    validation = metrics[metrics["split"] == "validation"].iloc[0]
    assert validation["match_accuracy"] == 1.0
    assert validation["win_brier"] < 0.001
    assert np.isfinite(validation["win_log_loss"])
    assert validation["next_match_phase_score_mse"] == 0.0
    assert validation["next_match_total_score_mse"] == 0.0
    assert validation["alliance_score_count"] == 4
    assert validation["win_correct_count"] == 2


def test_common_match_metrics_without_total_columns_report_nan_total_mse():
    opts = default_options()
    table = pd.DataFrame(
        {
            "red_auto_pts": [20.0],
            "red_teleop_pts": [80.0],
            "blue_auto_pts": [15.0],
            "blue_teleop_pts": [75.0],
            "red_win": [1.0],
        }
    )
    split = Split(
        train_mask=np.array([False]),
        validation_mask=np.array([True]),
        test_mask=np.array([False]),
        policy="fixture",
    )
    target_stats = TargetStats(
        target_names=["red_auto_pts", "red_teleop_pts", "blue_auto_pts", "blue_teleop_pts"],
        mu=np.zeros(4),
        sigma=np.ones(4),
        is_constant=np.zeros(4, dtype=bool),
    )

    metrics = common_match_metrics_from_predictions(
        table,
        split,
        target_stats,
        opts,
        table[target_stats.target_names].to_numpy(dtype=float),
        np.array([[_logit(0.99)]], dtype=float),
    )

    validation = metrics[metrics["split"] == "validation"].iloc[0]
    assert validation["next_match_phase_score_mse"] == 0.0
    assert np.isnan(validation["next_match_total_score_mse"])
