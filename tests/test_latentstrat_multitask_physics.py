from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from latentstrat.season.model import FixedTaskBalancer, init_model
from latentstrat.season.multitask_study import (
    MultitaskStudySpec,
    build_award_candidates,
    select_common_epochs,
    study_options,
)
from latentstrat.season.physics import (
    HUB_PERIODS,
    build_physics_target_table,
    compose_predicted_scores,
    hard_composed_scores,
    validate_hard_composer,
)
from latentstrat.season.train import model_loss


def _spec(arm: str = "physics-consistent", architecture: str = "full-match"):
    return MultitaskStudySpec(
        arm=arm,
        architecture=architecture,
        seed=2026,
        epochs=2,
        train_match_keys=("a",),
        evaluation_match_keys=("b",),
        score_logit_alpha=0.03,
        score_logit_sigma=2.0,
    )


def test_hard_composer_uses_atomic_tower_and_opponent_fouls():
    row = {"match_key": "2026test_qm1", "has_dq": False}
    for color, base in (("red", 1), ("blue", 2)):
        for period in HUB_PERIODS:
            row[f"{color}_hub_{period}_count"] = base
        for slot in (1, 2, 3):
            row[f"{color}_team_{slot}_auto_status"] = "Level1" if slot == 1 else "None"
            row[f"{color}_team_{slot}_endgame_status"] = f"Level{slot}"
        row[f"{color}_committed_minor_foul_count"] = base
        row[f"{color}_committed_major_foul_count"] = base
        row[f"{color}_adjust_points"] = 0
    table = pd.DataFrame([row])
    expected_red = 7 + 15 + 10 + 20 + 30 + 5 * 2 + 15 * 2
    expected_blue = 14 + 15 + 10 + 20 + 30 + 5 * 1 + 15 * 1
    table["red_official_score"] = expected_red
    table["blue_official_score"] = expected_blue
    np.testing.assert_array_equal(
        hard_composed_scores(table, include_adjustment=True),
        [[expected_red, expected_blue]],
    )
    assert validate_hard_composer(table)["max_absolute_error"] == 0


def test_differentiable_composer_is_symmetric_and_updates_every_branch():
    atomic = torch.randn(3, 14, requires_grad=True)
    foul = torch.randn(3, 4, requires_grad=True)
    auto = torch.randn(3, 6, 2, requires_grad=True)
    endgame = torch.randn(3, 6, 4, requires_grad=True)
    scores = compose_predicted_scores(atomic, foul, auto, endgame)
    swapped = compose_predicted_scores(
        torch.cat([atomic[:, 7:], atomic[:, :7]], dim=1),
        torch.cat([foul[:, 2:], foul[:, :2]], dim=1),
        torch.cat([auto[:, 3:], auto[:, :3]], dim=1),
        torch.cat([endgame[:, 3:], endgame[:, :3]], dim=1),
    )
    torch.testing.assert_close(swapped, scores.flip(1))
    scores.sum().backward()
    for tensor in (atomic, foul, auto, endgame):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) > 0


def test_physics_study_uses_fixed_balancer_and_bidirectional_consistency_gradients():
    opts = study_options(_spec())
    model = init_model(20, 16, 2, 1, opts, num_endgame_classes=4, num_awards=0)
    assert isinstance(model.loss_balancer, FixedTaskBalancer)
    assert len(model.loss_balancer.log_vars) == 0
    with torch.no_grad():
        model.physics_cont_mu.copy_(torch.tensor([200.0, 200.0]))
        model.physics_cont_sigma.copy_(torch.tensor([100.0, 100.0]))
        model.physics_atomic_sigma.fill_(25.0)
        model.physics_foul_sigma.fill_(2.0)
        model.physics_score_sigma.fill_(100.0)
    batch = 4
    red = torch.tensor([[1, 2, 3]] * batch)
    blue = torch.tensor([[4, 5, 6]] * batch)
    continuous = torch.zeros(batch, 2)
    winner = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    atomic_and_foul = torch.zeros(batch, 18)
    bonuses = torch.zeros(batch, 4)
    status_auto = torch.zeros(batch, 6, dtype=torch.long)
    status_endgame = torch.zeros(batch, 6, dtype=torch.long)
    total, metrics, _ = model_loss(
        model,
        red,
        blue,
        continuous,
        winner,
        opts,
        torch.ones(1),
        red_event_idx=torch.zeros_like(red),
        blue_event_idx=torch.zeros_like(blue),
        red_missing_mask=torch.zeros_like(red, dtype=torch.bool),
        blue_missing_mask=torch.zeros_like(blue, dtype=torch.bool),
        endgame_targets=status_endgame,
        auto_targets=status_auto,
        v57_cont_targets=atomic_and_foul,
        v57_bin_targets=bonuses,
        official_total_targets=torch.full((batch, 2), 200.0),
        dq_mask=torch.zeros(batch, dtype=torch.bool),
    )
    total.backward()
    assert metrics.score_consistency_loss >= 0
    assert metrics.winner_consistency_loss >= 0
    for parameter in (
        model.cont_head.weight,
        model.atomic_head.weight,
        model.foul_head.weight,
        model.bin_head.score.weight,
        model.Z_base.weight,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.parametrize(
    ("arm", "dense", "structured"),
    [
        ("primary-only", False, False),
        ("dense-breakdown", True, False),
        ("physics-consistent", True, False),
        ("full-structured", True, True),
    ],
)
def test_arm_models_instantiate_only_active_heads(arm, dense, structured):
    opts = study_options(_spec(arm=arm, architecture="additive"))
    model = init_model(20, 16, 2, 1, opts, num_endgame_classes=4, num_awards=4)
    assert (model.atomic_head is not None) is dense
    assert (model.foul_head is not None) is dense
    assert (model.bonus_head is not None) is dense
    assert (getattr(model, "auto_head", None) is not None) is dense
    assert (model.endgame_head is not None) is dense
    assert (model.award_head is not None) is structured
    assert (model.team_value_head is not None) is structured
    assert (model.alliance_value_head is not None) is structured
    parameter_names = {name for name, _ in model.named_parameters()}
    assert any(name.startswith("atomic_head") for name in parameter_names) is dense
    assert any(name.startswith("award_head") for name in parameter_names) is structured


def test_common_epoch_selection_uses_seed_mean_and_earliest_tie():
    rows = []
    for seed in (2026, 2027, 2028):
        for epoch, value in ((1, 2.0), (2, 1.0), (3, 1.0)):
            rows.append(
                {
                    "arm": "primary-only",
                    "architecture": "additive",
                    "seed": seed,
                    "epoch": epoch,
                    "validation_score_rmse": value,
                    "validation_score_differential_rmse": value,
                    "validation_winner_brier": value,
                    "validation_winner_log_loss": value,
                }
            )
    selected, curve = select_common_epochs(
        pd.DataFrame(rows),
        {"score_rmse": 1.0, "differential_rmse": 1.0, "brier": 1.0, "log_loss": 1.0},
    )
    assert selected.iloc[0]["selected_epoch"] == 2
    assert curve["epoch"].tolist() == [1, 2, 3]


def test_award_candidates_use_observed_winners_and_explicit_event_negatives():
    features = pd.DataFrame(
        {
            "event_key": ["2026test"],
            **{
                f"{color}_team_{slot}_key": [f"frc{slot + (0 if color == 'red' else 3)}"]
                for color in ("red", "blue")
                for slot in (1, 2, 3)
            },
            **{
                f"{color}_team_{slot}_award_{category}": [
                    1.0 if color == "red" and slot == 1 and category == "auto" else np.nan
                ]
                for color in ("red", "blue")
                for slot in (1, 2, 3)
                for category in ("auto", "design", "control", "excellence")
            },
        }
    )
    rankings = pd.DataFrame(
        {
            "event_key": ["2026test"] * 3,
            "team_key": ["frc1", "frc2", "frc3"],
        }
    )
    candidates = build_award_candidates(features, rankings, {"2026test"})
    assert candidates["award_auto"].tolist() == [1.0, 0.0, 0.0]
    assert candidates[["award_design", "award_control", "award_excellence"]].isna().all().all()


@pytest.mark.skipif(
    not Path("data/pretraining/match-breakdown/corpus.sqlite").exists(),
    reason="local cached corpus is unavailable",
)
def test_cached_2026_composer_audit_is_exact():
    table = build_physics_target_table("data/pretraining/match-breakdown/corpus.sqlite")
    audit = validate_hard_composer(table)
    assert audit["alliance_rows"] == 36_328
    assert audit["max_absolute_error"] == 0
    assert audit["adjustment_without_dq"] == 0
