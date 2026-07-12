from pathlib import Path

import pandas as pd
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from latentstrat.season.championship_study import classify_comparisons
from latentstrat.season.features import read_feature_table
from latentstrat.season.model import init_model
from latentstrat.season.static_core import (
    CHAMPIONSHIP_EVENTS,
    DEVELOPMENT_HOLDOUT_EVENTS,
    StaticStudySpec,
    _write_hparams_same_run,
    build_split_manifest,
    core_options,
)
from latentstrat.season.train import (
    _core_batch_task_weights,
    _write_feature_tensorboard_epoch,
    active_embedding_l2,
    create_tensorboard_writer,
)


def _spec(architecture: str = "full-match") -> StaticStudySpec:
    return StaticStudySpec(
        "official-total-core",
        architecture,
        100,
        2026,
        ("score", "win", "embedding"),
        ("train",),
        ("test",),
        1.0,
        1.0,
        1.0,
    )


def test_frozen_real_split_contract_when_features_are_available():
    root = Path(__file__).resolve().parents[1]
    features = root / "data/features/season/features_v58_2026.parquet"
    events = root / "data/features/season/events_2026.parquet"
    if not features.exists() or not events.exists():
        pytest.skip("Local 2026 evidence is not installed.")
    manifest = build_split_manifest(read_feature_table(features), events)
    counts = manifest.groupby("split_role")["match_key"].nunique()
    assert counts["development_train"] == 15_373
    assert counts["development_holdout"] == 1_642
    assert counts["final_train"] == 17_029
    assert counts["championship_test"] == 1_119
    assert counts["excluded_einstein"] == 16
    clean = manifest.groupby("split_role")["included_for_score"].sum()
    wins = manifest.groupby("split_role")["included_for_win"].sum()
    assert clean["development_train"] == 15_029
    assert wins["development_train"] == 14_986
    assert clean["development_holdout"] == 1_630
    assert wins["development_holdout"] == 1_625
    assert clean["final_train"] == 16_673
    assert wins["final_train"] == 16_625
    assert clean["championship_test"] == 1_108
    assert wins["championship_test"] == 1_106
    assert set(manifest.loc[manifest["split_role"].eq("development_holdout"), "event_key"]) == set(
        DEVELOPMENT_HOLDOUT_EVENTS
    )
    assert set(manifest.loc[manifest["split_role"].eq("championship_test"), "event_key"]) == set(
        CHAMPIONSHIP_EVENTS
    )


@pytest.mark.parametrize("architecture", ["additive", "teammate-set", "full-match"])
def test_core_model_has_only_active_modules_and_preserves_symmetry(architecture):
    options = core_options(_spec(architecture))
    model = init_model(12, 16, 2, 1, options)
    assert not model.loss_balancer.log_vars
    assert not any("log_vars" in name for name, _ in model.named_parameters())
    assert model.award_head is None
    assert model.endgame_head is None
    assert model.team_value_head is None
    red = torch.tensor([[1, 2, 3]])
    blue = torch.tensor([[4, 5, 6]])
    model.eval()
    with torch.inference_mode():
        original = model(red, blue)
        permuted = model(red[:, [2, 0, 1]], blue[:, [1, 2, 0]])
        swapped = model(blue, red)
    assert torch.allclose(original.cont_z, permuted.cont_z, atol=1e-6)
    assert torch.allclose(original.bin_logits, -swapped.bin_logits, atol=1e-6)


def test_core_activity_counts_and_normalized_embedding_regularization():
    batch = tuple(torch.zeros((3, 1)) for _ in range(16))
    batch = list(batch)
    batch[0] = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    batch[1] = torch.tensor([[4, 5, 6], [4, 5, 6], [1, 2, 3]])
    batch[6] = torch.tensor([[1.0, 2.0], [float("nan"), float("nan")], [3.0, 4.0]])
    batch[7] = torch.tensor([[1.0], [float("nan")], [0.0]])
    assert _core_batch_task_weights(tuple(batch)) == {
        "continuous": 2.0,
        "win": 2.0,
        "embedding": 3.0,
    }

    model = init_model(8, 16, 2, 1, core_options(_spec("additive")))
    with torch.no_grad():
        model.Z_base.weight.fill_(2.0)
    penalty = active_embedding_l2(
        model,
        torch.tensor([[1, 1, 2], [1, 2, 2]]),
        torch.tensor([[2, 3, 3], [3, 3, 3]]),
        1.0,
        normalized=True,
    )
    assert float(penalty.detach()) == pytest.approx(4.0)


def test_core_tensorboard_profile_uses_one_epoch_step(tmp_path):
    model = init_model(8, 16, 2, 1, core_options(_spec("additive")))
    writer = create_tensorboard_writer(str(tmp_path))
    row = {
        "epoch": 1,
        "train_loss": 2.0,
        "validation_loss": 2.1,
        "learning_rate_group_0": 1e-3,
        "preclip_grad_norm_mean": 0.2,
        "preclip_grad_norm_max": 0.3,
        "clipping_fraction": 0.0,
        "epoch_seconds": 1.0,
        "samples_per_second": 10.0,
        "z_base_norm_mean": 0.4,
        "z_base_update_norm_mean": 0.1,
        "train_score_rmse": 3.0,
        "train_score_mae": 2.0,
        "train_score_differential_rmse": 4.0,
        "train_score_differential_mae": 3.0,
        "train_winner_brier": 0.2,
        "train_winner_log_loss": 0.6,
        "train_winner_ece": 0.05,
        "validation_score_rmse": 3.5,
        "validation_score_mae": 2.5,
        "validation_score_differential_rmse": 4.5,
        "validation_score_differential_mae": 3.5,
        "validation_winner_brier": 0.21,
        "validation_winner_log_loss": 0.61,
        "validation_winner_ece": 0.06,
    }
    _write_feature_tensorboard_epoch(
        writer,
        model=model,
        row=row,
        task_means={"continuous": 1.0, "win": 0.5, "embedding": 0.1},
        task_counts={"continuous": 3, "win": 2, "embedding": 3},
        validation_task_means={"continuous": 1.1, "win": 0.6, "embedding": 0.1},
        validation_task_counts={"continuous": 3, "win": 2, "embedding": 3},
        active_sidecars=set(),
    )
    writer.close()
    accumulator = EventAccumulator(str(tmp_path)).Reload()
    scalars = accumulator.Tags()["scalars"]
    assert "Loss/Train" in scalars
    assert "LossComponent/Score" in scalars
    assert "Forecast/Validation/WinnerBrier" in scalars
    assert not any(tag.startswith("Loss/Train/") for tag in scalars)
    assert all(event.step == 1 for tag in scalars for event in accumulator.Scalars(tag))


def test_hparams_outcomes_use_windows_safe_metric_group(tmp_path):
    leaf = tmp_path / "development" / "official-total-core" / "additive"
    writer = create_tensorboard_writer(str(leaf))
    _write_hparams_same_run(
        writer,
        {
            "phase": "development",
            "supervision": "official-total-core",
            "architecture": "additive",
            "epochs": 100,
        },
        {"HParams/CompletedEpoch": 100.0, "HParams/FinalScoreRMSE": 42.5},
        step=100,
    )
    writer.close()
    metrics = EventAccumulator(str(leaf / "hparams_metrics")).Reload()
    assert metrics.Scalars("HParams/CompletedEpoch")[0].value == 100.0
    assert metrics.Scalars("HParams/FinalScoreRMSE")[0].value == 42.5
    assert len(list(leaf.glob("events.out.tfevents.*"))) == 1


def test_comparison_classification_applies_leave_one_division_veto():
    metrics = {
        "score_squared_error": (-0.05, 0.9),
        "differential_squared_error": (-0.01, 0.7),
        "winner_brier": (0.001, 0.6),
        "winner_log_loss": (0.002, 0.6),
    }
    comparisons = pd.DataFrame(
        [
            {
                "comparison": "supervision:additive",
                "metric": metric,
                "relative_delta": delta,
                "delta_right_minus_left": delta,
                "probability_improvement": probability,
            }
            for metric, (delta, probability) in metrics.items()
        ]
    )
    leave_one_out = pd.DataFrame(
        [
            {
                "comparison": "supervision:additive",
                "metric": "score_squared_error",
                "relative_delta": 0.03,
                "delta_right_minus_left": 1.0,
            }
        ]
    )
    result = classify_comparisons(comparisons, leave_one_out)
    assert result.loc[0, "classification"] == "uncertain"
