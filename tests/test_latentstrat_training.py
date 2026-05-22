import sys

import numpy as np
import pandas as pd
import torch

from latentstrat.config import default_options
from latentstrat.data import Split, apply_target_stats, fit_target_stats
from latentstrat.evaluation import evaluate_model
from latentstrat.model import init_model
from latentstrat.training import (
    MatchTensorDataset,
    model_loss,
    resolve_amp_enabled,
    resolve_compile_enabled,
    resolve_device,
    resolve_positive_weights,
    train_model,
)


def _training_table(row_count=8):
    rows = []
    for idx in range(row_count):
        red = [idx % 6 + 1, (idx + 1) % 6 + 1, (idx + 2) % 6 + 1]
        blue = [(idx + 3) % 6 + 1, (idx + 4) % 6 + 1, (idx + 5) % 6 + 1]
        rows.append(
            {
                "event_key": "2026test",
                "match_key": f"2026test_qm{idx + 1}",
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
                "blue_total_score": 95 + idx,
                "red_auto_pts": 20 + idx % 3,
                "red_teleop_pts": 80 + idx,
                "blue_auto_pts": 18 + idx % 3,
                "blue_teleop_pts": 77 + idx,
                "win_margin": 5,
                "red_foul_pts": 3,
                "blue_foul_pts": 1,
                "fouls_drawn": 2,
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


def _split(row_count):
    validation_mask = np.zeros(row_count, dtype=bool)
    validation_mask[-2:] = True
    return Split(
        train_mask=~validation_mask,
        validation_mask=validation_mask,
        test_mask=np.zeros(row_count, dtype=bool),
        policy="fixture",
    )


def test_match_tensor_dataset_converts_once_to_expected_dtypes():
    table = _training_table()
    opts = default_options()
    stats = fit_target_stats(table, np.ones(len(table), dtype=bool), opts)
    prepared = apply_target_stats(table, stats)

    dataset = MatchTensorDataset.from_table(prepared, opts)

    assert dataset.red_team_idx.dtype == torch.long
    assert dataset.cont_targets.dtype == torch.float32
    assert len(dataset) == len(table)


def test_train_model_uses_dataloader_and_returns_diagnostics():
    table = _training_table()
    opts = default_options().model_copy(
        update={"epochs": 2, "mini_batch_size": 3, "use_early_stopping": False}
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    prepared = apply_target_stats(table, stats)

    model, history, diagnostics = train_model(prepared, split, opts, verbose=False)

    assert len(history) == 2
    assert diagnostics.iterations >= 4
    assert diagnostics.device == next(model.parameters()).device.type
    assert diagnostics.amp_enabled is False
    assert diagnostics.compiled is False
    assert diagnostics.final_loss >= 0


def test_evaluation_runs_under_inference_mode_without_gradients():
    table = _training_table()
    opts = default_options().model_copy(update={"epochs": 1, "mini_batch_size": 4})
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    prepared = apply_target_stats(table, stats)
    model, _, _ = train_model(prepared, split, opts, verbose=False)
    for parameter in model.parameters():
        parameter.grad = None

    report = evaluate_model(model, prepared, split, stats, opts)

    assert len(report.continuous_metrics) == 8
    assert all(parameter.grad is None for parameter in model.parameters())


def test_positive_weights_handle_balanced_and_degenerate_labels():
    opts = default_options()

    balanced = resolve_positive_weights(np.array([[1.0], [0.0], [1.0], [0.0]]), opts)
    all_positive = resolve_positive_weights(np.array([[1.0], [1.0]]), opts)
    all_negative = resolve_positive_weights(np.array([[0.0], [0.0]]), opts)

    np.testing.assert_allclose(balanced, np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(all_positive, np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(all_negative, np.array([1.0], dtype=np.float32))


def test_award_loss_ignores_nan_targets():
    opts = default_options().model_copy(update={"team_dropout_rate": 0.0})
    model = init_model(8, opts.latent_dim, len(opts.target_map), len(opts.binary_targets), opts)
    red = torch.tensor([[1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[4, 5, 6]], dtype=torch.long)
    cont = torch.zeros((1, len(opts.target_map)))
    binary = torch.tensor([[1.0]])
    endgame = torch.zeros((1, 6), dtype=torch.long)
    awards = torch.full((1, 6, len(opts.award_targets)), float("nan"))

    loss, metrics, _ = model_loss(
        model,
        red,
        blue,
        cont,
        binary,
        opts,
        torch.tensor([1.0]),
        endgame_targets=endgame,
        award_targets=awards,
    )

    assert torch.isfinite(loss)
    assert metrics.award_loss == 0.0

    awards[0, 0, 0] = 1.0
    loss, metrics, _ = model_loss(
        model,
        red,
        blue,
        cont,
        binary,
        opts,
        torch.tensor([1.0]),
        endgame_targets=endgame,
        award_targets=awards,
    )

    assert torch.isfinite(loss)
    assert metrics.award_loss > 0


def test_resolve_device_returns_available_torch_device():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda", "mps"}


def test_acceleration_defaults_are_safe_by_device():
    opts = default_options()

    assert not resolve_amp_enabled(opts, torch.device("cpu"))
    assert resolve_amp_enabled(opts, torch.device("cuda"))
    assert not resolve_amp_enabled(opts, torch.device("mps"))

    assert not resolve_compile_enabled(opts, torch.device("cpu"))
    assert resolve_compile_enabled(opts, torch.device("cuda")) == (sys.platform != "win32")
    assert not resolve_compile_enabled(opts, torch.device("mps"))
