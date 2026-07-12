import pandas as pd
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typer.testing import CliRunner

from latentstrat.artifacts.checkpoints import load_season_checkpoint_model
from latentstrat.cli import app
from latentstrat.config import default_options
from latentstrat.pretraining.prior.train import apply_prior_checkpoint_to_model
from latentstrat.season.model import init_model
from latentstrat.season.reference_study import (
    nested_temporal_split,
    paired_event_bootstrap,
    reference_calibration,
    reference_metrics,
    refit_temporal_split,
    validate_reference_rows,
)
from latentstrat.season.train import (
    TASK_NAMES,
    _write_feature_tensorboard_epoch,
    _write_parameter_histograms,
    create_tensorboard_writer,
)


@pytest.mark.parametrize("architecture", ["additive", "teammate-set", "full-match"])
def test_static_reference_variants_are_permutation_tolerant_and_antisymmetric(architecture):
    opts = default_options().model_copy(
        update={
            "state_model": "static-z-base",
            "match_architecture": architecture,
            "team_dropout_rate": 0.0,
        }
    )
    model = init_model(8, 16, 4, 1, opts, num_event_teams=20)
    model.eval()
    assert not hasattr(model, "Z_event")
    assert not any(name.startswith("Z_event") for name in model.state_dict())
    red = torch.tensor([[1, 2, 3]])
    blue = torch.tensor([[4, 5, 6]])
    event_a = torch.tensor([[1, 2, 3]])
    event_b = torch.tensor([[9, 8, 7]])
    with torch.inference_mode():
        baseline = model(red, blue, red_event_idx=event_a, blue_event_idx=event_b)
        permuted = model(
            red[:, [2, 0, 1]],
            blue[:, [1, 2, 0]],
            red_event_idx=event_a,
            blue_event_idx=event_b,
        )
        changed_event = model(red, blue, red_event_idx=event_b, blue_event_idx=event_a)
        swapped = model(blue, red, red_event_idx=event_b, blue_event_idx=event_a)
    assert torch.allclose(baseline.cont_z, permuted.cont_z, atol=1e-5)
    assert torch.allclose(baseline.cont_z, changed_event.cont_z, atol=1e-7)
    assert torch.allclose(baseline.bin_logits, -swapped.bin_logits, atol=1e-6)
    assert baseline.auto_logits.shape == (1, 6, len(opts.auto_class_order) - 1)


def test_static_architecture_modules_are_absent_when_unused():
    additive = init_model(
        8,
        16,
        4,
        1,
        default_options().model_copy(
            update={"state_model": "static-z-base", "match_architecture": "additive"}
        ),
    )
    teammate = init_model(
        8,
        16,
        4,
        1,
        default_options().model_copy(
            update={"state_model": "static-z-base", "match_architecture": "teammate-set"}
        ),
    )
    assert not hasattr(additive, "sab")
    assert not hasattr(additive, "cross")
    assert not hasattr(additive, "pma")
    assert hasattr(teammate, "sab")
    assert hasattr(teammate, "pma")
    assert not hasattr(teammate, "cross")


def test_static_checkpoint_round_trip_is_strict(tmp_path):
    opts = default_options().model_copy(
        update={"state_model": "static-z-base", "match_architecture": "teammate-set"}
    )
    model = init_model(8, 16, 4, 1, opts)
    path = tmp_path / "fold_6.pt"
    torch.save(
        {
            "checkpoint_schema_version": 7,
            "model_state_dict": model.state_dict(),
            "options": opts.model_dump(mode="json"),
            "world_model": {"enabled": False},
        },
        path,
    )
    restored = load_season_checkpoint_model(path)
    assert restored.opts.state_model == "static-z-base"
    assert restored.opts.match_architecture == "teammate-set"
    assert set(restored.state_dict()) == set(model.state_dict())


def test_static_prior_handoff_uses_compact_2026_vocabulary(tmp_path):
    opts = default_options().model_copy(update={"state_model": "static-z-base"})
    model = init_model(4, 16, 4, 1, opts)
    prior = torch.arange(20 * 16, dtype=torch.float32).reshape(20, 16)
    path = tmp_path / "prior.pt"
    torch.save({"embedding_table": prior}, path)

    copied = apply_prior_checkpoint_to_model(
        model,
        path,
        latent_dim=16,
        team_index_map={"frc2": 1, "frc7": 2, "frc19": 3},
    )

    assert copied == 3
    assert torch.equal(model.Z_base.weight[0], prior[0])
    assert torch.equal(model.Z_base.weight[1], prior[2])
    assert torch.equal(model.Z_base.weight[2], prior[7])
    assert torch.equal(model.Z_base.weight[3], prior[19])


def test_reference_rows_and_nested_split_fail_closed():
    table = pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2025],
            "event_key": ["a", "b", "c", "d"],
            "match_key": ["a_qm1", "b_qm1", "c_qm1", "d_qm1"],
            "event_week": [3, 4, 4, 3],
            "red_score_raw": [10, 11, -1, 12],
            "blue_score_raw": [9, 10, -1, 13],
            "sort_ordinal": [1, 2, 3, 4],
        }
    )
    metadata = pd.DataFrame({"key": ["a", "b", "c", "d"], "event_type": [0, 1, 99, 0]})
    filtered, coverage = validate_reference_rows(table, metadata)
    assert filtered["match_key"].tolist() == ["a_qm1", "b_qm1"]
    assert coverage["included_rows"] == 2
    split = nested_temporal_split(filtered, train_max_week=3, holdout_week=4)
    assert split.train_mask.tolist() == [True, False]
    assert split.validation_mask.tolist() == [False, True]
    refit = refit_temporal_split(filtered, train_max_week=3)
    assert refit.train_mask.tolist() == [True, False]
    assert not refit.validation_mask.any()
    assert refit.test_mask.tolist() == [False, True]


def _predictions() -> pd.DataFrame:
    rows = []
    for week, event in ((6, "a"), (8, "b")):
        for match, actual_red, actual_blue in ((1, 100, 90), (2, 80, 80)):
            for model, offset in (("additive", 8), ("teammate-set", 3), ("full-match", 1)):
                rows.append(
                    {
                        "scope": "test",
                        "model": model,
                        "initialization": "random",
                        "fold_number": week,
                        "test_week": week,
                        "event_key": event,
                        "match_key": f"{event}_qm{match}",
                        "actual_red_total_score": actual_red,
                        "actual_blue_total_score": actual_blue,
                        "pred_red_total_score": actual_red + offset,
                        "pred_blue_total_score": actual_blue - offset,
                        "pred_red_win_probability": 0.8 if actual_red > actual_blue else 0.5,
                    }
                )
    return pd.DataFrame(rows)


def test_reference_metrics_calibration_and_cluster_bootstrap_are_deterministic():
    predictions = _predictions()
    metrics = reference_metrics(predictions)
    calibration = reference_calibration(predictions)
    first = paired_event_bootstrap(predictions, resamples=25, seed=2026)
    second = paired_event_bootstrap(predictions, resamples=25, seed=2026)
    assert {"scope", "model", "metric", "value", "count", "direction"}.issubset(metrics.columns)
    assert calibration.groupby(["scope", "model", "test_week"]).size().eq(10).all()
    pd.testing.assert_frame_equal(first, second)
    full_vs_additive = first[
        (first["candidate"] == "full-match")
        & (first["baseline"] == "additive")
        & (first["metric"] == "score_differential_squared_error")
    ]
    assert (full_vs_additive["delta_mean"] < 0).all()


def test_reference_tensorboard_contract_has_required_groups(tmp_path):
    opts = default_options().model_copy(
        update={"state_model": "static-z-base", "match_architecture": "full-match"}
    )
    model = init_model(8, 16, 4, 1, opts)
    writer = create_tensorboard_writer(str(tmp_path))
    task_values = {name: 1.0 for name in TASK_NAMES}
    row = {
        "epoch": 1,
        "train_loss": 2.0,
        "validation_loss": 2.5,
        "learning_rate_group_0": 1e-3,
        "epoch_seconds": 0.5,
        "samples_per_second": 10,
        "optimizer_steps": 2,
        "preclip_grad_norm_mean": 0.2,
        "preclip_grad_norm_max": 0.3,
        "clipping_fraction": 0.0,
        "nominal_parameter_count": model.parameter_count(),
        "trainable_parameter_count": model.parameter_count(),
        "gradient_receiving_parameter_count": model.parameter_count() - 1,
        "train_rows": 5,
        "validation_rows": 2,
        "train_events": 2,
        "validation_events": 1,
        "active_team_rows": 6,
        "z_base_norm_mean": 0.1,
        "z_base_norm_std": 0.01,
        "z_base_norm_p05": 0.08,
        "z_base_norm_p50": 0.1,
        "z_base_norm_p95": 0.12,
        "z_base_ghost_norm": 0.05,
        "z_base_update_norm_mean": 0.02,
    }
    _write_feature_tensorboard_epoch(
        writer,
        model=model,
        row=row,
        task_means=task_values,
        task_counts=task_values,
        validation_task_means=task_values,
        validation_task_counts=task_values,
        active_sidecars=set(),
    )
    _write_parameter_histograms(
        writer, model, torch.arange(1, 7), model.Z_base.weight.detach().clone(), step=1
    )
    writer.flush()
    writer.close()
    accumulator = EventAccumulator(str(tmp_path)).Reload()
    scalars = set(accumulator.Tags()["scalars"])
    histograms = set(accumulator.Tags()["histograms"])
    assert {
        "Loss/Train/Total",
        "Loss/Validation/Total",
        "Runtime/EpochSeconds",
        "Parameters/GradientReceiving",
        "State/ZBase/UpdateNormMean",
        "ActiveCount/Train/continuous",
    }.issubset(scalars)
    assert "State/ZBase/ActiveNorms" in histograms


def test_reference_cli_defaults_to_smoke_without_training(monkeypatch, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return type(
            "Result",
            (),
            {
                "output_dir": tmp_path / "out",
                "selected_initialization": "random",
                "tensorboard_logdir": None,
            },
        )()

    monkeypatch.setattr("latentstrat.cli.run_reference_study", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "season",
            "validate-reference-2026",
            "--features",
            str(tmp_path / "features.parquet"),
            "--events",
            str(tmp_path / "events.parquet"),
            "--prior-checkpoint",
            str(tmp_path / "prior.pt"),
            "--output",
            str(tmp_path / "out"),
            "--no-tensorboard",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["smoke"] is True
    assert captured["kwargs"]["epochs"] == 20
    assert "Reference study complete" in result.output
