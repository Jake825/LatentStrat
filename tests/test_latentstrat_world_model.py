import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from latentstrat.config import default_options
from latentstrat.experimental.frozen_targets import (
    TargetSpaceOptions,
    WorldModelOptions,
    build_award_target_space,
    build_pick_target_space,
    build_rank_target_space,
    load_world_model_bundle,
    paired_bootstrap_noninferiority,
    ranked_unselected_opportunities,
)
from latentstrat.model import init_model
from latentstrat.training import model_loss


def _target(enabled: bool, width: int) -> TargetSpaceOptions:
    return TargetSpaceOptions(enabled=enabled, width=width)


def _world_options(
    *,
    score: tuple[bool, int] = (False, 4),
    award: tuple[bool, int] = (False, 256),
    rank: tuple[bool, int] = (False, 4),
    pick: tuple[bool, int] = (False, 4),
) -> WorldModelOptions:
    return WorldModelOptions(
        score_embedding=_target(*score),
        award_embedding=_target(*award),
        rank_embedding=_target(*rank),
        pick_embedding=_target(*pick),
        target_epochs=2,
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"key": "2026week1", "event_type": 0, "event_completed": True},
            {"key": "2026week2", "event_type": 1, "event_completed": False},
            {"key": "2026offseason", "event_type": 99, "event_completed": True},
        ]
    )


def test_score_target_space_is_rejected_until_per_alliance_runtime_integration():
    try:
        _world_options(score=(True, 16)).active_spaces()
    except ValueError as exc:
        assert "score_embedding is offline-only" in str(exc)
    else:
        raise AssertionError("Expected deferred score integration rejection.")


class _FakeEmbeddings:
    def create(self, *, model: str, input: list[str], dimensions: int):
        data = []
        for text in input:
            seed = (sum(ord(char) for char in text) % 13) / 13
            data.append(
                SimpleNamespace(
                    embedding=[seed + (idx + 1) / dimensions for idx in range(dimensions)]
                )
            )
        return SimpleNamespace(data=data)


class _FakeOpenAIClient:
    embeddings = _FakeEmbeddings()


def test_award_prototypes_keep_normalized_raw_256d_geometry_without_pca(tmp_path):
    catalog = tmp_path / "awards.yaml"
    catalog.write_text(
        """
awards:
  - id: impact
    family: culture
    description: Honors sustained impact.
    variants: [Recognizes measurable outreach.]
  - id: quality
    family: engineering
    description: Honors robust machine quality.
    variants: [Recognizes a reliable robot.]
""".strip(),
        encoding="utf-8",
    )

    space = build_award_target_space(
        catalog,
        tmp_path,
        _world_options(award=(True, 256)),
        cache_path=tmp_path / "openai-cache.sqlite",
        client=_FakeOpenAIClient(),
    )

    columns = [column for column in space.embeddings if column.startswith("z_")]
    assert len(columns) == 256
    assert np.allclose(np.linalg.norm(space.embeddings[columns].to_numpy(), axis=1), 1.0)
    assert "pca" not in json.dumps(space.bundle).lower()
    payload = torch.load(space.root / "model.pt", map_location="cpu", weights_only=False)
    assert payload == {"kind": "normalized-openai-prototypes", "width": 256}


def test_rank_target_space_rejects_incomplete_event_labels(tmp_path):
    rankings = pd.DataFrame(
        [
            {
                "season": 2026,
                "event_key": "2026week1",
                "event_week": 1,
                "team_key": "frc1",
                "qual_rank": 1,
                "matches_played": 10,
                "record_wins": 8,
            },
            {
                "season": 2026,
                "event_key": "2026week2",
                "event_week": 2,
                "team_key": "frc2",
                "qual_rank": 2,
                "matches_played": 10,
                "record_wins": 7,
            },
        ]
    )

    space = build_rank_target_space(
        rankings,
        tmp_path,
        _world_options(rank=(True, 4)),
        event_metadata=_events(),
        fit_max_week=2,
    )

    assert list(space.embeddings["team_key"]) == ["frc1"]
    assert space.bundle["audit"]["excluded_incomplete_event_rows"] == 1


def _rankings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "event_key": "2026week1",
                "event_week": 1,
                "team_key": f"frc{team}",
                "qual_rank": team,
            }
            for team in range(1, 5)
        ]
    )


def _selections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "event_key": "2026week1",
                "event_week": 1,
                "alliance_number": 1,
                "pick_order": 1,
                "captain_team_key": "frc1",
                "pick_team_key": "frc3",
            }
        ]
    )


def test_pick_targets_use_offline_pool_but_runtime_predictor_is_fixed_width(tmp_path):
    opportunities = ranked_unselected_opportunities(_selections(), _rankings())
    assert set(opportunities["candidate_team_key"]) == {"frc2", "frc3", "frc4"}
    assert opportunities["picked"].sum() == 1

    space = build_pick_target_space(
        _selections(),
        _rankings(),
        tmp_path,
        _world_options(pick=(True, 4)),
        event_metadata=_events(),
        fit_max_week=1,
    )
    payload = torch.load(space.root / "model.pt", map_location="cpu", weights_only=False)
    assert payload["offline_context"] == "ranked-unselected-pool-mean"

    opts = default_options()
    model = init_model(
        5,
        opts.latent_dim,
        len(opts.target_map),
        len(opts.binary_targets),
        opts,
        num_event_teams=5,
        world_model_opts=_world_options(pick=(True, 4)),
    )
    prediction = model.predict_selection_embedding(
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
    )
    assert prediction.shape == (2, 4)


def test_award_prototype_loss_is_positive_unlabeled_masked():
    opts = default_options().model_copy(update={"team_dropout_rate": 0.0})
    world = _world_options(award=(True, 4))
    model = init_model(
        8,
        opts.latent_dim,
        len(opts.target_map),
        len(opts.binary_targets),
        opts,
        world_model_opts=world,
    )
    red = torch.tensor([[1, 2, 3]])
    blue = torch.tensor([[4, 5, 6]])
    cont = torch.zeros((1, len(opts.target_map)))
    binary = torch.ones((1, len(opts.binary_targets)))
    prototypes = torch.full((1, 6, 4), float("nan"))

    _, metrics, _ = model_loss(
        model,
        red,
        blue,
        cont,
        binary,
        opts,
        torch.ones(len(opts.binary_targets)),
        wm_award_targets=prototypes,
    )
    assert metrics.wm_award_loss == 0.0

    prototypes[0, 0] = torch.nn.functional.normalize(torch.ones(4), dim=0)
    _, metrics, _ = model_loss(
        model,
        red,
        blue,
        cont,
        binary,
        opts,
        torch.ones(len(opts.binary_targets)),
        wm_award_targets=prototypes,
    )
    assert metrics.wm_award_loss > 0


def test_world_model_bundle_rejects_missing_enabled_space(tmp_path):
    (tmp_path / "bundle.json").write_text(
        json.dumps(
            {
                "world_model": _world_options(rank=(True, 4)).model_dump(mode="json"),
                "spaces": {},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_world_model_bundle(tmp_path)
    except ValueError as exc:
        assert "missing enabled spaces: rank" in str(exc)
    else:
        raise AssertionError("Expected missing target bundle rejection.")


def test_paired_bootstrap_is_deterministic_and_uses_noninferiority_epsilons():
    baseline = pd.DataFrame(
        [
            {
                "fold_number": fold,
                "match_key": f"m{fold}_{match}",
                "win_brier": 0.20,
                "win_log_loss": 0.60,
                "pred_red_total_score": 101.0,
                "actual_red_total_score": 100.0,
                "pred_blue_total_score": 91.0,
                "actual_blue_total_score": 90.0,
            }
            for fold in (1, 2)
            for match in range(3)
        ]
    )
    candidate = baseline.copy()
    candidate["win_brier"] += 0.001
    candidate["win_log_loss"] += 0.005
    candidate["pred_red_total_score"] += 0.005

    first = paired_bootstrap_noninferiority(baseline, candidate, resamples=100)
    second = paired_bootstrap_noninferiority(baseline, candidate, resamples=100)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == {"brier", "log_loss", "total_score_mse"}
    assert first["passes_noninferiority"].all()
