from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from latentstrat.cli import app
from latentstrat.config import PriorOpts, default_options
from latentstrat.model import init_model
from latentstrat.pretrain_features import (
    CULTURE_TARGET_COLUMNS,
    GHOST_TEAM_NARRATIVE,
    NORM_EPA_OBSERVED_COLUMNS,
    NORM_EPA_TARGET_COLUMNS,
    KnownTeamRecord,
    OpenAIEmbeddingCache,
    add_culture_targets,
    add_norm_epa_trajectory_targets,
    build_future_rookie_narrative,
    build_gap_team_narrative,
    build_prior_feature_table,
    build_prior_narratives,
    build_rich_team_narrative,
    clean_narrative_text,
    compress_years,
    embed_narratives,
    extract_statbotics_norm_epa,
    projected_future_rookie_year,
    write_prior_feature_table,
)
from latentstrat.pretrain_loop import (
    apply_prior_checkpoint_to_model,
    default_tensorboard_run_name,
    load_prior_embedding_table,
    train_prior_file,
    train_prior_model,
)
from latentstrat.prior_inspection import (
    inspect_prior_checkpoint,
    write_prior_inspection_artifacts,
)
from latentstrat.prior_model import (
    TeamPriorDistiller,
    distillation_loss,
    prior_distillation_losses,
)


class FakeEmbeddings:
    def __init__(self, failures: int = 0) -> None:
        self.calls = 0
        self.failures = failures
        self.inputs: list[list[str]] = []

    def create(self, *, model: str, input: list[str], dimensions: int):
        self.calls += 1
        self.inputs.append(list(input))
        if self.calls <= self.failures:
            raise FakeRateLimitError()
        data = []
        for text in input:
            seed = (sum(ord(char) for char in text) % 17) / 17.0
            embedding = [seed + idx / dimensions for idx in range(dimensions)]
            data.append(SimpleNamespace(embedding=embedding))
        return SimpleNamespace(data=data)


class FakeOpenAIClient:
    def __init__(self, failures: int = 0) -> None:
        self.embeddings = FakeEmbeddings(failures=failures)


class FakeRateLimitError(Exception):
    status_code = 429


class FakeProvider:
    teams = [
        {"key": "frc1", "team_number": 1},
        {"key": "frc2", "team_number": 2},
        {"key": "frc11", "team_number": 11},
    ]

    def get_teams_page(self, page: int, simple: bool = True) -> list[dict]:
        return self.teams if page == 0 else []

    def get_team(self, team_key: str) -> dict:
        number = int(team_key.removeprefix("frc"))
        return {
            "key": team_key,
            "team_number": number,
            "nickname": f"Orbit {number}",
            "name": "NASA / Motorola / Example High School",
            "city": "Chicago",
            "state_prov": "IL",
            "country": "USA",
            "rookie_year": 2010 + number,
        }

    def get_team_years(self, team_key: str) -> list[int]:
        return {
            "frc1": [2024, 2025, 2026],
            "frc2": [2023, 2024],
            "frc11": [2026],
        }[team_key]

    def get_team_events(self, team_key: str) -> list[dict]:
        return [
            {"key": "2024cmpmi", "name": "Championship Division", "year": 2024, "event_type": 3},
            {"key": "2025mrcmp", "name": "Midwest Regional", "year": 2025, "event_type": 0},
            {"key": "2026ilch", "name": "Current Regional", "year": 2026, "event_type": 0},
            {"key": "2027ilch", "name": "Future Regional", "year": 2027, "event_type": 0},
        ]

    def get_team_awards(self, team_key: str) -> list[dict]:
        return [
            {
                "name": "Regional Winner",
                "award_type": 1,
                "event_key": "2024cmpmi",
                "year": 2024,
                "recipient_list": [{"team_key": team_key}],
            },
            {
                "name": "Autonomous Award",
                "award_type": 74,
                "event_key": "2025mrcmp",
                "year": 2025,
                "recipient_list": [{"team_key": team_key}],
            },
            {
                "name": "FIRST Impact Award",
                "award_type": 0,
                "event_key": "2026ilch",
                "year": 2026,
                "recipient_list": [{"team_key": team_key}],
            },
            {
                "name": "Quality Award",
                "award_type": 18,
                "event_key": "2027ilch",
                "year": 2027,
                "recipient_list": [{"team_key": team_key}],
            },
        ]


class FakeStatboticsProvider:
    rows = [
        {"team": 1, "year": 2022, "norm_epa": {"current": 0.10}},
        {"team": 1, "year": 2023, "norm_epa": {"current": 0.20}},
        {"team": 1, "year": 2024, "norm_epa": {"current": 0.30}},
        {"team": 1, "year": 2025, "norm_epa": {"current": 0.40}},
        {"team": 2, "year": 2025, "norm_epa_current": 0.80},
        {"team": 99, "year": 2026, "norm_epa": {"current": 9.90}},
    ]

    def get_team_years(self, **filters) -> list[dict]:
        year = int(filters.get("year", 0))
        offset = int(filters.get("offset", 0))
        if offset:
            return []
        return [row for row in self.rows if int(row["year"]) == year]


class FakeTensorBoardWriter:
    instances: list[FakeTensorBoardWriter] = []

    def __init__(self, log_dir) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.closed = False
        self.instances.append(self)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.texts.append((tag, str(text), int(step)))

    def close(self) -> None:
        self.closed = True


def _prior_opts(tmp_path=None, **updates) -> PriorOpts:
    defaults = {
        "max_team_number": 4,
        "future_start_number": 10,
        "future_baseline_team": 9,
        "future_growth_per_year": 2,
        "epochs": 3,
        "batch_size": 2,
        "cache_path": ":memory:" if tmp_path is None else str(tmp_path / "cache.sqlite"),
    }
    defaults.update(updates)
    return PriorOpts(**defaults)


def _prior_table(row_count: int = 3) -> pd.DataFrame:
    team_numbers = list(range(row_count + 1))
    return pd.DataFrame(
        {
            "team_number": team_numbers,
            "team_key": [f"frc{idx}" for idx in team_numbers],
            "target_season": [2026] * len(team_numbers),
            "archetype": ["ghost_token", *(["anchor"] * row_count)],
            "narrative": [
                GHOST_TEAM_NARRATIVE,
                *[f"Team frc{idx} history." for idx in range(1, row_count + 1)],
            ],
            "narrative_hash": [f"hash-{idx}" for idx in team_numbers],
            "embedding_model": ["text-embedding-3-small"] * len(team_numbers),
            "llm_dim": [256] * len(team_numbers),
            "openai_narrative_vector": [
                [float((idx + offset) % 11) / 11.0 for offset in range(256)]
                for idx in team_numbers
            ],
            "norm_epa_source_years": ["2022,2023,2024,2025"] * len(team_numbers),
            **{
                column: [
                    float("nan") if idx == 0 else float(idx + axis) / 10.0
                    for idx in team_numbers
                ]
                for axis, column in enumerate(NORM_EPA_TARGET_COLUMNS)
            },
            **{
                column: [idx != 0 for idx in team_numbers]
                for column in NORM_EPA_OBSERVED_COLUMNS
            },
            **{
                column: [0.0 if idx == 0 else float(idx + axis) for idx in team_numbers]
                for axis, column in enumerate(CULTURE_TARGET_COLUMNS)
            },
        }
    )


def _feature_table(row_count: int = 6) -> pd.DataFrame:
    rows = []
    for idx in range(row_count):
        rows.append(
            {
                "season": 2026,
                "event_key": "2026test",
                "match_key": f"2026test_qm{idx + 1}",
                "comp_level": "qm",
                "set_number": 1,
                "match_number": idx + 1,
                "red_team_1_key": "frc1",
                "red_team_2_key": "frc2",
                "red_team_3_key": "frc3",
                "blue_team_1_key": "frc4",
                "blue_team_2_key": "frc5",
                "blue_team_3_key": "frc6",
                "sort_ordinal": idx + 1,
                "sort_time": 1_780_000_000 + idx,
                "event_week": 0,
                "red_total_score": 100 + idx,
                "blue_total_score": 90 + idx,
                "red_auto_pts": 20 + idx,
                "red_teleop_pts": 80 + idx,
                "blue_auto_pts": 18 + idx,
                "blue_teleop_pts": 72 + idx,
                "red_foul_pts": 0,
                "blue_foul_pts": 0,
                "win_margin": 10,
                "red_win": True,
            }
        )
    table = pd.DataFrame(rows)
    for color in ("red", "blue"):
        for slot in (1, 2, 3):
            table[f"{color}_team_{slot}_endgame_status"] = "Level1"
    return table


def test_compress_years_groups_contiguous_ranges():
    assert compress_years([2001, 2002, 2003, 2005]) == "2001-2003, 2005"


def test_rich_narrative_preserves_signal_and_quarantines_target_season():
    provider = FakeProvider()
    narrative = build_rich_team_narrative(
        provider.get_team("frc1"),
        provider.get_team_years("frc1"),
        provider.get_team_awards("frc1"),
        provider.get_team_events("frc1"),
        target_season=2026,
        active_in_target=True,
    )

    assert "NASA / Motorola / Example High School" in narrative
    assert "2024-2025" in narrative
    assert "won official events" in narrative
    assert "FIRST World Championship" in narrative
    assert "Autonomous Award: 2025" in narrative
    assert "FIRST Impact Award" not in narrative
    assert "Quality Award" not in narrative
    assert "2027" not in narrative


def test_narrative_cleaning_removes_json_artifacts_and_null_literals():
    narrative = clean_narrative_text('{ "name": null, "nickname": None, "team": "frc1" }')

    assert "{" not in narrative
    assert "}" not in narrative
    assert '"' not in narrative
    assert "null" not in narrative.lower()
    assert "None" not in narrative


def test_gap_and_future_narratives_encode_dark_matter_context():
    gap = build_gap_team_narrative(
        4000,
        2012,
        [{"number": 3999, "name": "Shaker Robotics", "location": "Latham, NY"}],
    )
    opts = _prior_opts(future_baseline_team=10_900, future_growth_per_year=800)
    future = build_future_rookie_narrative(12_500, opts)

    assert "Team 3999 (Shaker Robotics from Latham, NY)" in gap
    assert projected_future_rookie_year(12_500, opts) == 2028
    assert "2031" not in future
    assert "2028 season" in future
    assert "WPILib" in future


def test_build_prior_narratives_assigns_all_v56_archetypes():
    known = {
        1: KnownTeamRecord(
            1,
            "frc1",
            FakeProvider().get_team("frc1"),
            [2025, 2026],
            [],
            [],
        ),
        2: KnownTeamRecord(
            2,
            "frc2",
            FakeProvider().get_team("frc2"),
            [2024],
            [],
            [],
        ),
    }
    opts = _prior_opts(max_team_number=12, future_start_number=10)

    table = build_prior_narratives(known, 2026, opts)

    assert len(table) == 13
    assert table["team_number"].iloc[0] == 0
    assert table["team_key"].iloc[0] == "frc0"
    assert table["archetype"].iloc[0] == "ghost_token"
    assert table["narrative"].iloc[0] == GHOST_TEAM_NARRATIVE
    assert set(table["archetype"]) == {
        "ghost_token",
        "anchor",
        "ghost",
        "sibling",
        "future",
    }


def test_build_prior_feature_table_emits_exact_requested_universe():
    opts = _prior_opts(max_team_number=12, future_start_number=10)
    table = build_prior_feature_table(
        FakeProvider(),
        target_season=2026,
        opts=opts,
        client=FakeOpenAIClient(),
        statbotics_provider=FakeStatboticsProvider(),
    )

    assert len(table) == 13
    assert table["team_number"].tolist() == list(range(13))
    assert set(table["archetype"]) == {
        "ghost_token",
        "anchor",
        "ghost",
        "sibling",
        "future",
    }
    assert table["openai_narrative_vector"].map(len).eq(256).all()
    assert table.loc[0, "narrative"] == GHOST_TEAM_NARRATIVE
    assert set(NORM_EPA_TARGET_COLUMNS).issubset(table.columns)
    assert set(NORM_EPA_OBSERVED_COLUMNS).issubset(table.columns)
    assert set(CULTURE_TARGET_COLUMNS).issubset(table.columns)
    assert table.loc[1, "norm_epa_t_minus_4"] == pytest.approx(0.10)
    assert table.loc[1, "norm_epa_t_minus_1"] == pytest.approx(0.40)
    assert table.loc[2, "norm_epa_t_minus_1"] == pytest.approx(0.80)
    assert not bool(table.loc[0, "norm_epa_observed_t_minus_1"])
    assert table["raw_rookie_year_delta"].notna().all()


def test_norm_epa_trajectory_uses_statbotics_normalized_epa_and_masks_missing():
    opts = _prior_opts(max_team_number=4)
    narratives = build_prior_narratives({}, target_season=2026, opts=opts)

    table = add_norm_epa_trajectory_targets(
        narratives,
        FakeStatboticsProvider(),
        target_season=2026,
        opts=opts,
    )

    assert table.loc[1, "norm_epa_t_minus_4"] == pytest.approx(0.10)
    assert table.loc[1, "norm_epa_t_minus_1"] == pytest.approx(0.40)
    assert table.loc[2, "norm_epa_t_minus_1"] == pytest.approx(0.80)
    assert pd.isna(table.loc[0, "norm_epa_t_minus_1"])
    assert not bool(table.loc[0, "norm_epa_observed_t_minus_1"])
    assert table["norm_epa_source_years"].eq("2022,2023,2024,2025").all()


def test_extract_norm_epa_does_not_use_raw_epa():
    row = {
        "epa": {"norm": 1505.0, "total_points": {"mean": 99.0}},
        "norm_epa": {"current": 1.25},
    }

    assert extract_statbotics_norm_epa(row) == pytest.approx(1505.0)


def test_norm_epa_trajectory_requires_observations():
    class EmptyStatboticsProvider:
        def get_team_years(self, **filters):
            return []

    opts = _prior_opts(max_team_number=4)
    narratives = build_prior_narratives({}, target_season=2026, opts=opts)

    with pytest.raises(ValueError, match="No Statbotics normalized EPA observations"):
        add_norm_epa_trajectory_targets(
            narratives,
            EmptyStatboticsProvider(),
            target_season=2026,
            opts=opts,
        )


def test_culture_targets_are_raw_and_quarantined():
    provider = FakeProvider()
    record = KnownTeamRecord(
        1,
        "frc1",
        provider.get_team("frc1"),
        provider.get_team_years("frc1"),
        provider.get_team_events("frc1"),
        provider.get_team_awards("frc1"),
    )
    table = build_prior_narratives({1: record}, target_season=2026, opts=_prior_opts())

    table = add_culture_targets(table, {1: record}, target_season=2026)

    assert table.loc[0, "raw_rookie_year_delta"] == 0.0
    assert table.loc[1, "raw_rookie_year_delta"] == 19.0
    assert table.loc[1, "raw_seasons_played"] == 2.0
    assert table.loc[1, "raw_total_award_count"] == 2.0
    assert table.loc[1, "raw_blue_banner_count"] == 1.0
    assert table.loc[1, "raw_championship_appearance_count"] == 1.0
    assert table.loc[1, "raw_championship_win_count"] == 1.0
    assert table.loc[1, "raw_technical_award_count"] == 1.0


def test_openai_embedding_cache_uses_stable_request_hash(tmp_path):
    cache = OpenAIEmbeddingCache(tmp_path / "cache.sqlite")
    opts = _prior_opts(tmp_path)
    client = FakeOpenAIClient()
    narratives = {1: "Same text prior."}

    first = embed_narratives(narratives, opts, client=client, cache=cache)
    second = embed_narratives(narratives, opts, client=client, cache=cache)

    assert client.embeddings.calls == 1
    assert first == second


def test_embedding_batching_preserves_outputs_and_deduplicates_narratives(tmp_path):
    cache = OpenAIEmbeddingCache(tmp_path / "cache.sqlite")
    opts = _prior_opts(tmp_path, embedding_batch_size=2)
    client = FakeOpenAIClient()
    narratives = {
        1: "Shared text.",
        2: "Shared text.",
        3: "Unique text A.",
        4: "Unique text B.",
    }

    embeddings = embed_narratives(narratives, opts, client=client, cache=cache)

    assert set(embeddings) == set(narratives)
    assert client.embeddings.calls == 2
    assert [len(batch) for batch in client.embeddings.inputs] == [2, 1]
    assert embeddings[1] == embeddings[2]


def test_embedding_retry_recovers_from_fake_429_and_caches_result(tmp_path):
    cache = OpenAIEmbeddingCache(tmp_path / "cache.sqlite")
    opts = _prior_opts(
        tmp_path,
        max_embedding_retries=2,
        embedding_retry_initial_delay=0.0,
        embedding_retry_max_delay=0.0,
        embedding_retry_jitter=0.0,
    )
    client = FakeOpenAIClient(failures=1)

    embeddings = embed_narratives({1: "Retry text."}, opts, client=client, cache=cache)
    cached = embed_narratives({1: "Retry text."}, opts, client=client, cache=cache)

    assert set(embeddings) == {1}
    assert cached == embeddings
    assert client.embeddings.calls == 2


def test_embedding_retry_raises_after_exhausting_retries(tmp_path):
    cache = OpenAIEmbeddingCache(tmp_path / "cache.sqlite")
    opts = _prior_opts(
        tmp_path,
        max_embedding_retries=1,
        embedding_retry_initial_delay=0.0,
        embedding_retry_max_delay=0.0,
        embedding_retry_jitter=0.0,
    )
    client = FakeOpenAIClient(failures=3)

    with pytest.raises(FakeRateLimitError):
        embed_narratives({1: "Retry text."}, opts, client=client, cache=cache)


def test_prior_distiller_forward_shape_and_ghost_row():
    opts = _prior_opts(max_team_number=4)
    model = TeamPriorDistiller(opts)

    prediction, pred_norm_epa, pred_culture, latent = model(torch.tensor([0, 1, 2]))

    assert prediction.shape == (3, opts.llm_dim)
    assert pred_norm_epa.shape == (3, len(NORM_EPA_TARGET_COLUMNS))
    assert pred_culture.shape == (3, len(CULTURE_TARGET_COLUMNS))
    assert latent.shape == (3, opts.latent_dim)
    assert model.team_embedding.padding_idx is None
    assert not torch.allclose(model.team_embedding.weight[0], torch.zeros(opts.latent_dim))


def test_prior_distiller_has_only_coordinate_map_and_decoder():
    model = TeamPriorDistiller(_prior_opts(max_team_number=4))

    assert set(dict(model.named_children())) == {
        "team_embedding",
        "decoder",
        "openai_head",
        "norm_epa_head",
        "culture_head",
    }
    assert isinstance(model.team_embedding, torch.nn.Embedding)
    assert isinstance(model.decoder, torch.nn.Sequential)
    assert isinstance(model.log_var_openai, torch.nn.Parameter)
    assert isinstance(model.log_var_epa, torch.nn.Parameter)
    assert isinstance(model.log_var_culture, torch.nn.Parameter)
    assert tuple(model.log_var_epa.shape) == ()
    assert tuple(model.log_var_culture.shape) == (7,)


def test_prior_distiller_ghost_row_receives_gradients():
    opts = _prior_opts(max_team_number=4)
    model = TeamPriorDistiller(opts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = model.team_embedding.weight[0].detach().clone()
    team_numbers = torch.tensor([0])
    target = torch.ones((1, opts.llm_dim))
    target_norm_epa = torch.ones((1, len(NORM_EPA_TARGET_COLUMNS)))
    target_culture = torch.ones((1, len(CULTURE_TARGET_COLUMNS)))

    prediction, pred_norm_epa, pred_culture, _ = model(team_numbers)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss = loss + torch.nn.functional.mse_loss(pred_norm_epa, target_norm_epa)
    loss = loss + torch.nn.functional.mse_loss(pred_culture, target_culture)
    loss.backward()
    assert model.team_embedding.weight.grad is not None
    assert not torch.allclose(
        model.team_embedding.weight.grad[0],
        torch.zeros_like(model.team_embedding.weight.grad[0]),
    )
    optimizer.step()

    assert not torch.allclose(model.team_embedding.weight[0], before)


def test_distiller_loss_includes_row_zero():
    opts = _prior_opts(max_team_number=4)
    model = TeamPriorDistiller(opts)
    target = torch.ones((1, opts.llm_dim))
    target_norm_epa = torch.ones((1, len(NORM_EPA_TARGET_COLUMNS)))
    norm_epa_observed = torch.ones((1, len(NORM_EPA_TARGET_COLUMNS)), dtype=torch.bool)
    target_culture = torch.ones((1, len(CULTURE_TARGET_COLUMNS)))

    loss = distillation_loss(
        model,
        torch.tensor([0]),
        target,
        target_norm_epa,
        norm_epa_observed,
        target_culture,
    )

    assert float(loss.detach()) > 0.0


def test_distiller_openai_loss_is_feature_summed():
    opts = _prior_opts(max_team_number=1)
    model = TeamPriorDistiller(opts)
    team_numbers = torch.tensor([1])
    target = torch.zeros((1, opts.llm_dim))
    target_norm_epa = torch.zeros((1, len(NORM_EPA_TARGET_COLUMNS)))
    norm_epa_observed = torch.zeros((1, len(NORM_EPA_TARGET_COLUMNS)), dtype=torch.bool)
    target_culture = torch.zeros((1, len(CULTURE_TARGET_COLUMNS)))

    with torch.no_grad():
        prediction, _, _, _ = model(team_numbers)
        target.copy_(prediction + 1.0)

    losses = prior_distillation_losses(
        model,
        team_numbers,
        target,
        target_norm_epa,
        norm_epa_observed,
        target_culture,
    )

    assert losses.openai_mse.detach().item() == pytest.approx(float(opts.llm_dim))


def test_distiller_epa_loss_is_feature_summed_trajectory():
    opts = _prior_opts(max_team_number=1)
    model = TeamPriorDistiller(opts)
    team_numbers = torch.tensor([1])
    target = torch.zeros((1, opts.llm_dim))
    target_norm_epa = torch.zeros((1, len(NORM_EPA_TARGET_COLUMNS)))
    norm_epa_observed = torch.ones((1, len(NORM_EPA_TARGET_COLUMNS)), dtype=torch.bool)
    target_culture = torch.zeros((1, len(CULTURE_TARGET_COLUMNS)))

    with torch.no_grad():
        _, pred_norm_epa, _, _ = model(team_numbers)
        target_norm_epa.copy_(pred_norm_epa + 1.0)

    losses = prior_distillation_losses(
        model,
        team_numbers,
        target,
        target_norm_epa,
        norm_epa_observed,
        target_culture,
    )

    assert losses.norm_epa_mse.detach().item() == pytest.approx(
        float(len(NORM_EPA_TARGET_COLUMNS))
    )


def test_distiller_epa_loss_is_safe_with_missing_trajectory():
    opts = _prior_opts(max_team_number=4)
    model = TeamPriorDistiller(opts)
    target = torch.ones((1, opts.llm_dim))
    target_norm_epa = torch.full((1, len(NORM_EPA_TARGET_COLUMNS)), float("nan"))
    norm_epa_observed = torch.zeros((1, len(NORM_EPA_TARGET_COLUMNS)), dtype=torch.bool)
    target_culture = torch.ones((1, len(CULTURE_TARGET_COLUMNS)))

    loss = distillation_loss(
        model,
        torch.tensor([1]),
        target,
        target_norm_epa,
        norm_epa_observed,
        target_culture,
    )

    assert torch.isfinite(loss)


def test_distiller_epa_trajectory_backpropagates_as_grouped_task():
    opts = _prior_opts(max_team_number=4)
    model = TeamPriorDistiller(opts)
    target = torch.zeros((1, opts.llm_dim))
    target_norm_epa = torch.tensor([[1500.0, float("nan"), 1505.0, 1510.0]])
    norm_epa_observed = torch.tensor([[True, False, True, True]])
    target_culture = torch.zeros((1, len(CULTURE_TARGET_COLUMNS)))

    loss = distillation_loss(
        model,
        torch.tensor([1]),
        target,
        target_norm_epa,
        norm_epa_observed,
        target_culture,
    )
    loss.backward()

    assert model.log_var_epa.grad is not None
    assert model.norm_epa_head.weight.grad is not None
    assert model.team_embedding.weight.grad is not None
    assert torch.isfinite(model.log_var_epa.grad)


def test_distiller_training_reduces_reconstruction_loss():
    table = _prior_table(4)
    opts = _prior_opts(max_team_number=4, epochs=20, batch_size=4, learning_rate=1e-2)

    _, history, _ = train_prior_model(table, opts, verbose=False)

    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]


def test_prior_distiller_checkpoint_contract(tmp_path):
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "pretrained_prior_2026.pt"
    write_prior_feature_table(_prior_table(4), features)
    opts = _prior_opts(tmp_path, max_team_number=4, epochs=2, batch_size=2)

    result = train_prior_file(features, output, opts, verbose=False)
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert output.exists()
    assert checkpoint["target_season"] == 2026
    assert checkpoint["embedding_table"].shape == (5, opts.latent_dim)
    assert not torch.allclose(
        checkpoint["embedding_table"][0],
        torch.zeros(opts.latent_dim),
    )
    assert checkpoint["max_team_number"] == 4
    assert checkpoint["feature_metadata"]["has_ghost_token"]
    assert checkpoint["feature_metadata"]["has_norm_epa_trajectory"]
    assert checkpoint["feature_metadata"]["has_culture_targets"]
    assert checkpoint["feature_metadata"]["norm_epa_source_years"] == [2022, 2023, 2024, 2025]
    assert checkpoint["feature_metadata"]["culture_target_names"] == list(CULTURE_TARGET_COLUMNS)
    assert checkpoint["prior_opts"]["llm_dim"] == 256
    assert "model_state_dict" not in checkpoint
    assert "decoder_state_dict" not in checkpoint
    assert result.embedding_table.shape == (5, opts.latent_dim)

    full_model_path = tmp_path / "full_model.pt"
    torch.save(
        {
            "embedding_table": result.embedding_table,
            "model_state_dict": result.model.state_dict(),
            "history": result.history.to_dict(orient="records"),
        },
        full_model_path,
    )
    assert output.stat().st_size < full_model_path.stat().st_size


def test_train_prior_file_writes_tensorboard_scalars_and_closes(tmp_path, monkeypatch):
    FakeTensorBoardWriter.instances.clear()
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "pretrained_prior_2026.pt"
    write_prior_feature_table(_prior_table(2), features)
    monkeypatch.setattr(
        "latentstrat.pretrain_loop.create_tensorboard_writer",
        lambda log_dir: FakeTensorBoardWriter(log_dir),
    )

    result = train_prior_file(
        features,
        output,
        _prior_opts(tmp_path, max_team_number=2, epochs=1, batch_size=2),
        verbose=False,
        tensorboard_logdir=tmp_path / "runs",
        tensorboard_run_name="test_run",
    )

    writer = FakeTensorBoardWriter.instances[-1]
    tags = {tag for tag, _, _ in writer.scalars}
    assert result.tensorboard_logdir == tmp_path / "runs" / "test_run"
    assert writer.closed
    assert writer.texts[0][0] == "Run/Context"
    assert {
        "Loss/Total",
        "Loss/OpenAI_MSE",
        "Loss/Norm_EPA_MSE",
        "Loss/Culture_MSE",
        "LogVar/OpenAI",
        "LogVar/EPA",
        "LogVar/Culture/raw_blue_banner_count",
        "Weights/OpenAI_Precision",
        "Weights/EPA_Precision",
        "Weights/Culture_Precision/raw_blue_banner_count",
    }.issubset(tags)
    assert "model_state_dict" not in result.checkpoint


def test_tensorboard_writer_closes_when_training_raises(tmp_path, monkeypatch):
    FakeTensorBoardWriter.instances.clear()
    features = tmp_path / "prior_features_v56.parquet"
    output = tmp_path / "pretrained_prior_2026.pt"
    write_prior_feature_table(_prior_table(2).drop(columns=["norm_epa_t_minus_1"]), features)
    monkeypatch.setattr(
        "latentstrat.pretrain_loop.create_tensorboard_writer",
        lambda log_dir: FakeTensorBoardWriter(log_dir),
    )

    with pytest.raises(KeyError, match="Missing: .*norm_epa_t_minus_1"):
        train_prior_file(
            features,
            output,
            _prior_opts(tmp_path, max_team_number=2, epochs=1, batch_size=2),
            verbose=False,
            tensorboard_logdir=tmp_path / "runs",
        )

    assert FakeTensorBoardWriter.instances[-1].closed


def test_default_tensorboard_run_name_uses_target_season():
    run_name = default_tensorboard_run_name(_prior_table(2), timestamp=123)

    assert run_name == "v564_prior_2026_123"


def test_prior_inspection_exports_tables_and_pngs(tmp_path):
    features = tmp_path / "prior_features.parquet"
    checkpoint_path = tmp_path / "pretrained_prior_2026.pt"
    output = tmp_path / "inspection"
    write_prior_feature_table(_prior_table(3), features)
    opts = _prior_opts(tmp_path, max_team_number=3, epochs=2, batch_size=3)
    train_prior_file(features, checkpoint_path, opts, verbose=False)

    inspection = inspect_prior_checkpoint(checkpoint_path, features, top_k=2)
    artifact_dir = write_prior_inspection_artifacts(inspection, output)

    assert len(inspection.latent_table) == 4
    assert {"pc1", "pc2", "pc3", "reconstruction_mse"}.issubset(
        inspection.latent_table.columns
    )
    assert inspection.latent_table["reconstruction_mse"].isna().all()
    assert len(inspection.nearest_neighbors) == 8
    assert inspection.sanity_checks["passed"].notna().all()
    for name in (
        "prior_latent_table.csv",
        "prior_nearest_neighbors.csv",
        "prior_training_history.csv",
        "prior_sanity_checks.csv",
        "prior_tsne_coordinates.csv",
        "prior_latent_stat_correlations.csv",
        "prior_pca_pc1_pc2.png",
        "prior_pca_pc1_pc3.png",
        "prior_tsne_team_galaxy.png",
        "prior_latent_stat_correlation_heatmap.png",
        "prior_embedding_norm_hist.png",
        "prior_training_loss.png",
    ):
        path = artifact_dir / name
        assert path.exists()
        assert path.stat().st_size > 0
    tsne = pd.read_csv(artifact_dir / "prior_tsne_coordinates.csv")
    assert {"team_key", "tsne_x", "tsne_y"}.issubset(tsne.columns)
    correlations = pd.read_csv(artifact_dir / "prior_latent_stat_correlations.csv")
    assert {"stat", "latent_dim", "correlation", "count"}.issubset(correlations.columns)


def test_prior_inspection_accepts_stripped_checkpoint_without_features(tmp_path):
    features = tmp_path / "prior_features.parquet"
    checkpoint_path = tmp_path / "pretrained_prior_2026.pt"
    write_prior_feature_table(_prior_table(3), features)
    train_prior_file(
        features,
        checkpoint_path,
        _prior_opts(tmp_path, max_team_number=3, epochs=1, batch_size=3),
        verbose=False,
    )

    inspection = inspect_prior_checkpoint(checkpoint_path, top_k=1)

    assert len(inspection.latent_table) == 4
    assert inspection.latent_table["team_key"].tolist() == ["frc0", "frc1", "frc2", "frc3"]
    assert inspection.latent_table["reconstruction_mse"].isna().all()


def test_day_zero_prior_handoff_copies_embedding_table(tmp_path):
    opts = default_options()
    model = init_model(5, opts.latent_dim, 4, 1, opts)
    checkpoint = tmp_path / "prior.pt"
    embedding_table = torch.zeros((5, opts.latent_dim))
    embedding_table[0] = 3.0
    embedding_table[1] = 1.0
    embedding_table[2] = 2.0
    torch.save({"embedding_table": embedding_table}, checkpoint)

    applied = apply_prior_checkpoint_to_model(model, checkpoint, latent_dim=opts.latent_dim)

    assert applied == 4
    assert torch.allclose(model.Z_base.weight[1], torch.ones(opts.latent_dim))
    assert torch.allclose(model.Z_base.weight[2], torch.full((opts.latent_dim,), 2.0))
    assert torch.allclose(model.Z_base.weight[0], torch.full((opts.latent_dim,), 3.0))


def test_default_prior_and_model_latent_dimension_is_sixteen():
    assert default_options().latent_dim == 16
    assert PriorOpts().latent_dim == 16


def test_load_prior_embedding_table_reads_stripped_checkpoint(tmp_path):
    checkpoint = tmp_path / "prior.pt"
    embedding_table = torch.zeros((3, 16))
    embedding_table[0] = 2.0
    embedding_table[1] = 1.0
    torch.save({"embedding_table": embedding_table}, checkpoint)

    loaded = load_prior_embedding_table(checkpoint)

    assert torch.allclose(loaded, embedding_table)


def test_build_prior_features_cli_uses_mocked_provider_and_openai(tmp_path, monkeypatch):
    output = tmp_path / "prior.parquet"
    fake_client = FakeOpenAIClient()
    monkeypatch.setattr("latentstrat.cli._provider", lambda: FakeProvider())
    monkeypatch.setattr("latentstrat.cli._statbotics_provider", lambda: FakeStatboticsProvider())
    monkeypatch.setattr("latentstrat.pretrain_features._openai_client", lambda: fake_client)

    result = CliRunner().invoke(
        app,
        [
            "build-prior-features",
            "--target-season",
            "2026",
            "--output",
            str(output),
            "--cache-path",
            str(tmp_path / "cache.sqlite"),
            "--max-team-number",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    table = pd.read_parquet(output, engine="pyarrow")
    assert len(table) == 13
    assert table["team_number"].tolist() == list(range(13))
    assert table["archetype"].iloc[0] == "ghost_token"
    assert set(NORM_EPA_TARGET_COLUMNS).issubset(table.columns)
    assert set(NORM_EPA_OBSERVED_COLUMNS).issubset(table.columns)
    assert set(CULTURE_TARGET_COLUMNS).issubset(table.columns)
    assert table.loc[table["team_number"] == 0, "raw_rookie_year_delta"].iloc[0] == 0.0


def test_train_prior_cli_writes_checkpoint(tmp_path):
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "prior_dir"
    write_prior_feature_table(_prior_table(2), features)

    result = CliRunner().invoke(
        app,
        [
            "train-prior",
            "--features",
            str(features),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--latent-dim",
            "8",
            "--batch-size",
            "2",
            "--max-team-number",
            "2",
            "--no-tensorboard",
        ],
    )

    assert result.exit_code == 0, result.output
    checkpoint_path = output / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["embedding_table"].shape == (3, 8)
    assert not torch.allclose(checkpoint["embedding_table"][0], torch.zeros(8))
    assert "model_state_dict" not in checkpoint


def test_train_prior_cli_enables_tensorboard_by_default(tmp_path, monkeypatch):
    FakeTensorBoardWriter.instances.clear()
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "prior_dir"
    write_prior_feature_table(_prior_table(2), features)
    monkeypatch.setattr(
        "latentstrat.pretrain_loop.create_tensorboard_writer",
        lambda log_dir: FakeTensorBoardWriter(log_dir),
    )

    result = CliRunner().invoke(
        app,
        [
            "train-prior",
            "--features",
            str(features),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-team-number",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "TensorBoard active: tensorboard --logdir=runs" in result.output
    assert FakeTensorBoardWriter.instances
    assert FakeTensorBoardWriter.instances[-1].closed
    assert str(FakeTensorBoardWriter.instances[-1].log_dir).startswith("runs")


def test_train_prior_cli_no_tensorboard_does_not_create_writer(tmp_path, monkeypatch):
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "prior_dir"
    write_prior_feature_table(_prior_table(2), features)

    def fail_create_writer(log_dir):
        raise AssertionError(f"Unexpected TensorBoard writer: {log_dir}")

    monkeypatch.setattr("latentstrat.pretrain_loop.create_tensorboard_writer", fail_create_writer)

    result = CliRunner().invoke(
        app,
        [
            "train-prior",
            "--features",
            str(features),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-team-number",
            "2",
            "--no-tensorboard",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "TensorBoard active" not in result.output


def test_train_prior_requires_v564_targets(tmp_path):
    features = tmp_path / "prior_features_v56.parquet"
    table = _prior_table(2).drop(columns=["norm_epa_t_minus_1"])
    write_prior_feature_table(table, features)

    result = CliRunner().invoke(
        app,
        [
            "train-prior",
            "--features",
            str(features),
            "--output",
            str(tmp_path / "prior.pt"),
            "--epochs",
            "1",
            "--max-team-number",
            "2",
            "--no-tensorboard",
        ],
    )

    assert result.exit_code != 0
    assert "Missing: ['norm_epa_t_minus_1']" in str(result.exception)


def test_train_prior_rejects_all_false_epa_masks(tmp_path):
    features = tmp_path / "prior_features_no_epa.parquet"
    table = _prior_table(2)
    for column in NORM_EPA_TARGET_COLUMNS:
        table[column] = float("nan")
    for column in NORM_EPA_OBSERVED_COLUMNS:
        table[column] = False
    write_prior_feature_table(table, features)

    result = CliRunner().invoke(
        app,
        [
            "train-prior",
            "--features",
            str(features),
            "--output",
            str(tmp_path / "prior.pt"),
            "--epochs",
            "1",
            "--max-team-number",
            "2",
            "--no-tensorboard",
        ],
    )

    assert result.exit_code != 0
    assert "zero observed normalized EPA trajectory values" in str(result.exception)


def test_inspect_prior_cli_writes_artifacts(tmp_path):
    features = tmp_path / "prior_features.parquet"
    checkpoint_path = tmp_path / "pretrained_prior.pt"
    output = tmp_path / "prior_inspection"
    write_prior_feature_table(_prior_table(3), features)
    train_prior_file(
        features,
        checkpoint_path,
        _prior_opts(tmp_path, max_team_number=3, epochs=1, batch_size=3),
        verbose=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "inspect-prior",
            "--checkpoint",
            str(checkpoint_path),
            "--features",
            str(features),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "prior_latent_table.csv").exists()
    assert (output / "prior_pca_pc1_pc2.png").stat().st_size > 0


def test_inspect_prior_cli_accepts_checkpoint_without_features(tmp_path):
    features = tmp_path / "prior_features.parquet"
    checkpoint_path = tmp_path / "pretrained_prior.pt"
    output = tmp_path / "prior_inspection"
    write_prior_feature_table(_prior_table(3), features)
    train_prior_file(
        features,
        checkpoint_path,
        _prior_opts(tmp_path, max_team_number=3, epochs=1, batch_size=3),
        verbose=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "inspect-prior",
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "prior_latent_table.csv").exists()
    table = pd.read_csv(output / "prior_latent_table.csv")
    assert table["team_key"].iloc[0] == "frc0"
    assert table["reconstruction_mse"].isna().all()


def test_train_features_cli_accepts_prior_checkpoint(tmp_path):
    features = tmp_path / "features.parquet"
    output = tmp_path / "artifacts"
    prior = tmp_path / "prior.pt"
    _feature_table().to_parquet(features, engine="pyarrow", index=False)
    embedding_table = torch.zeros((7, default_options().latent_dim))
    embedding_table[0] = 2.0
    embedding_table[1:] = 1.0
    torch.save({"embedding_table": embedding_table}, prior)

    result = CliRunner().invoke(
        app,
        [
            "train-features",
            str(features),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--prior-checkpoint",
            str(prior),
        ],
    )

    assert result.exit_code == 0, result.output
    checkpoint = torch.load(output / "v6_checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["prior_vectors_applied"] == 6
