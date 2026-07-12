
import numpy as np
import pandas as pd
import pytest
import torch

from latentstrat.config import default_options
from latentstrat.data import Split, apply_target_stats, fit_target_stats
from latentstrat.evaluation import evaluate_model
from latentstrat.model import init_model
from latentstrat.training import (
    AlliancePairDataset,
    MatchTensorDataset,
    RankPairDataset,
    SelectionTripletDataset,
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
                "red_atomic_auto_count": float(idx),
                "blue_atomic_auto_count": float(idx + 1),
                "red_atomic_transition_count": np.nan if idx == 0 else float(idx),
                "blue_atomic_transition_count": float(idx),
                "red_atomic_shift1_count": float(idx),
                "blue_atomic_shift1_count": float(idx),
                "red_atomic_shift2_count": float(idx),
                "blue_atomic_shift2_count": float(idx),
                "red_atomic_shift3_count": float(idx),
                "blue_atomic_shift3_count": float(idx),
                "red_atomic_shift4_count": float(idx),
                "blue_atomic_shift4_count": float(idx),
                "red_atomic_endgame_count": float(idx),
                "blue_atomic_endgame_count": float(idx),
                "red_committed_foul_pts": float(idx % 3),
                "blue_committed_foul_pts": float((idx + 1) % 3),
                "red_committed_minor_foul_count": float(idx % 2),
                "blue_committed_minor_foul_count": float((idx + 1) % 2),
                "red_committed_major_foul_count": 0.0,
                "blue_committed_major_foul_count": 0.0,
                "red_bonus_energized": float(idx % 2 == 0),
                "blue_bonus_energized": float(idx % 2 == 1),
                "red_bonus_supercharged": np.nan,
                "blue_bonus_supercharged": np.nan,
                "red_bonus_traversal": 0.0,
                "blue_bonus_traversal": 1.0,
                "red_special_g206_penalty": 0.0,
                "blue_special_g206_penalty": 0.0,
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
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        prepared, fit_v57_target_stats(table, np.ones(len(table), dtype=bool), opts)
    )

    dataset = MatchTensorDataset.from_table(prepared, opts)

    assert dataset.red_team_idx.dtype == torch.long
    assert dataset.cont_targets.dtype == torch.float32
    assert dataset.v57_cont_targets.shape[1] == 2 * (
        len(opts.atomic_count_targets) + len(opts.foul_targets)
    )
    assert dataset.v57_bin_targets.shape[1] == 2 * (
        len(opts.bonus_binary_targets) + len(opts.special_binary_targets)
    )
    assert len(dataset) == len(table)


def test_train_model_uses_dataloader_and_returns_diagnostics():
    table = _training_table()
    opts = default_options().model_copy(
        update={"epochs": 2, "mini_batch_size": 3, "use_early_stopping": False}
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        apply_target_stats(table, stats), fit_v57_target_stats(table, split.train_mask, opts)
    )

    model, history, diagnostics = train_model(prepared, split, opts, verbose=False)

    assert len(history) == 2
    assert diagnostics.iterations >= 4
    assert diagnostics.device == next(model.parameters()).device.type
    assert diagnostics.amp_enabled is False
    assert diagnostics.compiled is False
    assert diagnostics.final_loss >= 0
    assert {"train_loss", "validation_loss"}.issubset(history.columns)
    assert "continuous_loss" in history.columns
    assert "continuous_precision" in history.columns
    assert "learning_rate" in history.columns


def test_train_model_can_freeze_embeddings_while_training_heads():
    table = _training_table()
    opts = default_options().model_copy(
        update={
            "epochs": 2,
            "mini_batch_size": 4,
            "use_early_stopping": False,
            "restore_best_validation_model": False,
            "learning_rate": 1e-2,
            "team_dropout_rate": 0.0,
        }
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        apply_target_stats(table, stats), fit_v57_target_stats(table, split.train_mask, opts)
    )
    model = init_model(
        7,
        opts.latent_dim,
        len(opts.continuous_targets),
        len(opts.binary_targets),
        opts,
        num_event_teams=7,
        num_endgame_classes=len(opts.endgame_class_order),
        num_awards=len(opts.award_targets),
    )
    base_before = model.Z_base.weight.detach().clone()
    event_before = model.Z_event.weight.detach().clone()
    head_before = model.cont_head.weight.detach().clone()

    trained, _, _ = train_model(
        prepared,
        split,
        opts,
        initial_model=model,
        freeze_team_embeddings=True,
        verbose=False,
    )

    assert torch.allclose(trained.Z_base.weight, base_before)
    assert torch.allclose(trained.Z_event.weight, event_before)
    assert not torch.allclose(trained.cont_head.weight, head_before)
    assert trained.Z_base.weight.requires_grad is False
    assert trained.Z_event.weight.requires_grad is False


def test_best_validation_restore_works_without_early_stopping():
    table = _training_table()
    opts = default_options().model_copy(
        update={"epochs": 2, "mini_batch_size": 4, "use_early_stopping": False}
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        apply_target_stats(table, stats), fit_v57_target_stats(table, split.train_mask, opts)
    )

    _, history, diagnostics = train_model(prepared, split, opts, verbose=False)

    assert len(history) == 2
    assert diagnostics.stopped_early is False
    assert diagnostics.best_epoch is not None
    assert diagnostics.restored_best_validation_model is True


def test_cosine_scheduler_records_decaying_learning_rate():
    table = _training_table()
    opts = default_options().model_copy(
        update={
            "epochs": 3,
            "mini_batch_size": 4,
            "use_early_stopping": False,
            "restore_best_validation_model": False,
            "learning_rate": 1e-3,
            "lr_eta_min": 1e-5,
        }
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        apply_target_stats(table, stats), fit_v57_target_stats(table, split.train_mask, opts)
    )

    _, history, _ = train_model(prepared, split, opts, verbose=False)

    assert history["learning_rate"].iloc[0] > history["learning_rate"].iloc[-1]


def test_loss_log_vars_are_clamped_after_training_step():
    table = _training_table()
    opts = default_options().model_copy(
        update={
            "epochs": 1,
            "mini_batch_size": 4,
            "use_early_stopping": False,
            "restore_best_validation_model": False,
            "loss_log_var_min": -5.0,
            "loss_log_var_max": 5.0,
        }
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        apply_target_stats(table, stats), fit_v57_target_stats(table, split.train_mask, opts)
    )
    model = init_model(
        7,
        opts.latent_dim,
        len(opts.continuous_targets),
        len(opts.binary_targets),
        opts,
    )
    with torch.no_grad():
        for idx, parameter in enumerate(model.loss_balancer.log_vars.values()):
            parameter.fill_(-50.0 if idx % 2 == 0 else 50.0)

    trained, _, _ = train_model(prepared, split, opts, initial_model=model, verbose=False)

    for parameter in trained.loss_balancer.log_vars.values():
        value = float(parameter.detach())
        assert value >= opts.loss_log_var_min
        assert value <= opts.loss_log_var_max
        assert torch.isfinite(torch.exp(-parameter))


class _FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))


def test_train_model_writes_tensorboard_task_scalars():
    table = _training_table()
    opts = default_options().model_copy(
        update={"epochs": 1, "mini_batch_size": 4, "use_early_stopping": False}
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        apply_target_stats(table, stats), fit_v57_target_stats(table, split.train_mask, opts)
    )
    writer = _FakeWriter()

    train_model(prepared, split, opts, verbose=False, tensorboard_writer=writer)

    tags = {tag for tag, _, _ in writer.scalars}
    assert "Loss/Train_Total" in tags
    assert "LossRaw/continuous" in tags
    assert "LossRaw/atomic" in tags
    assert "LogVar/continuous" in tags
    assert "Weights/continuous_precision" in tags
    assert "LR/base" in tags


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


def test_v57_masked_losses_ignore_nan_targets():
    opts = default_options().model_copy(update={"team_dropout_rate": 0.0})
    model = init_model(8, opts.latent_dim, len(opts.target_map), len(opts.binary_targets), opts)
    red = torch.tensor([[1, 2, 3]], dtype=torch.long)
    blue = torch.tensor([[4, 5, 6]], dtype=torch.long)
    cont = torch.zeros((1, len(opts.target_map)))
    binary = torch.tensor([[1.0]])
    v57_cont = torch.full(
        (1, 2 * (len(opts.atomic_count_targets) + len(opts.foul_targets))), float("nan")
    )
    v57_bin = torch.full(
        (1, 2 * (len(opts.bonus_binary_targets) + len(opts.special_binary_targets))),
        float("nan"),
    )

    loss, metrics, _ = model_loss(
        model,
        red,
        blue,
        cont,
        binary,
        opts,
        torch.tensor([1.0]),
        v57_cont_targets=v57_cont,
        v57_bin_targets=v57_bin,
    )

    assert torch.isfinite(loss)
    assert metrics.atomic_loss == 0.0
    assert metrics.bonus_loss == 0.0

    v57_cont[0, 0] = 1.0
    v57_bin[0, 0] = 1.0
    loss, metrics, _ = model_loss(
        model,
        red,
        blue,
        cont,
        binary,
        opts,
        torch.tensor([1.0]),
        v57_cont_targets=v57_cont,
        v57_bin_targets=v57_bin,
    )

    assert torch.isfinite(loss)
    assert metrics.atomic_loss > 0
    assert metrics.bonus_loss > 0


def test_sidecar_datasets_and_training_with_unequal_lengths():
    table = _training_table(10)
    opts = default_options().model_copy(
        update={"epochs": 1, "mini_batch_size": 3, "use_early_stopping": False}
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        apply_target_stats(table, stats), fit_v57_target_stats(table, split.train_mask, opts)
    )
    rankings = pd.DataFrame(
        {
            "event_key": ["2026test"] * 4,
            "team_key": ["frc1", "frc2", "frc3", "frc4"],
            "qual_rank": [1, 2, 3, 4],
            "team_base_idx": [1, 2, 3, 4],
            "team_event_idx": [1, 2, 3, 4],
        }
    )
    selections = pd.DataFrame(
        {
            "event_key": ["2026test"],
            "captain_base_idx": [1],
            "pick_base_idx": [3],
            "passed_over_base_idx": [2],
            "captain_event_idx": [1],
            "pick_event_idx": [3],
            "passed_over_event_idx": [2],
        }
    )
    playoffs = pd.DataFrame(
        {
            "event_key": ["2026test", "2026test"],
            "playoff_finish_order": [1, 2],
            "team_1_base_idx": [1, 4],
            "team_2_base_idx": [2, 5],
            "team_3_base_idx": [3, 6],
            "team_1_event_idx": [1, 4],
            "team_2_event_idx": [2, 5],
            "team_3_event_idx": [3, 6],
        }
    )

    assert len(RankPairDataset.from_table(rankings)) == 3
    assert len(SelectionTripletDataset.from_table(selections)) == 1
    assert len(AlliancePairDataset.from_table(playoffs)) == 1

    model, history, diagnostics = train_model(
        prepared,
        split,
        opts,
        verbose=False,
        sidecar_tables={"rankings": rankings, "selections": selections, "playoffs": playoffs},
    )

    assert len(history) == 1
    assert diagnostics.iterations > len(selections)
    assert model.team_value_head.linear.weight.grad is None


def test_resolve_device_returns_available_torch_device():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda", "mps"}


def test_acceleration_defaults_are_safe_by_device():
    opts = default_options()

    assert not resolve_amp_enabled(opts, torch.device("cpu"))
    assert not resolve_amp_enabled(opts, torch.device("cuda"))
    assert not resolve_amp_enabled(opts, torch.device("mps"))

    assert not resolve_compile_enabled(opts, torch.device("cpu"))
    assert not resolve_compile_enabled(opts, torch.device("cuda"))
    assert not resolve_compile_enabled(opts, torch.device("mps"))


def test_season_training_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch):
    table = _training_table()
    opts = default_options().model_copy(
        update={
            "epochs": 2,
            "mini_batch_size": 4,
            "use_early_stopping": False,
            "restore_best_validation_model": False,
            "team_dropout_rate": 0.0,
        }
    )
    split = _split(len(table))
    stats = fit_target_stats(table, split.train_mask, opts)
    from latentstrat.data import apply_v57_target_stats, fit_v57_target_stats

    prepared = apply_v57_target_stats(
        apply_target_stats(table, stats),
        fit_v57_target_stats(table, split.train_mask, opts),
    )
    uninterrupted, uninterrupted_history, _ = train_model(
        prepared, split, opts, verbose=False
    )

    resume_path = tmp_path / "latest.ckpt"
    from latentstrat.season import train as training_module

    original_save = training_module.save_resume_checkpoint

    def save_then_interrupt(path, payload):
        result = original_save(path, payload)
        if payload["completed_epoch"] == 1:
            raise RuntimeError("simulated interruption")
        return result

    monkeypatch.setattr(training_module, "save_resume_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_model(
            prepared,
            split,
            opts,
            verbose=False,
            resume_output=str(resume_path),
        )
    monkeypatch.setattr(training_module, "save_resume_checkpoint", original_save)
    resumed, resumed_history, _ = train_model(
        prepared,
        split,
        opts,
        verbose=False,
        resume_checkpoint=str(resume_path),
    )

    for name, tensor in uninterrupted.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[name]), name
    metric_columns = [
        column
        for column in uninterrupted_history.columns
        if column not in {"epoch_seconds", "samples_per_second"}
    ]
    pd.testing.assert_frame_equal(
        uninterrupted_history[metric_columns].reset_index(drop=True),
        resumed_history[metric_columns].reset_index(drop=True),
    )
