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
    OpenAIEmbeddingCache,
    build_prior_feature_table,
    build_team_narrative,
    clean_narrative_text,
    embed_narratives,
    write_prior_feature_table,
)
from latentstrat.pretrain_loop import (
    apply_prior_checkpoint_to_model,
    train_prior_file,
)
from latentstrat.prior_inspection import (
    inspect_prior_checkpoint,
    write_prior_inspection_artifacts,
)
from latentstrat.prior_model import UnifiedPriorAutoencoder, reconstruction_loss


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
    def get_team(self, team_key: str) -> dict:
        return {
            "key": team_key,
            "nickname": "Orbit",
            "city": "Chicago",
            "state_prov": "IL",
            "country": "USA",
            "rookie_year": 2010,
        }

    def get_team_events(self, team_key: str) -> list[dict]:
        return [
            {"key": "2025mrcmp", "name": "Midwest Regional", "year": 2025},
            {"key": "2026ilch", "name": "Current Regional", "year": 2026},
            {"key": "2027ilch", "name": "Future Regional", "year": 2027},
        ]

    def get_team_awards(self, team_key: str) -> list[dict]:
        return [
            {
                "name": "Autonomous Award",
                "event_key": "2025mrcmp",
                "year": 2025,
                "recipient_list": [{"team_key": team_key}],
            },
            {
                "name": "FIRST Impact Award",
                "event_key": "2026ilch",
                "year": 2026,
                "recipient_list": [{"team_key": team_key}],
            },
            {
                "name": "Quality Award",
                "event_key": "2027ilch",
                "year": 2027,
                "recipient_list": [{"team_key": team_key}],
            },
        ]


def _prior_table(row_count: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_key": [f"frc{idx + 1}" for idx in range(row_count)],
            "target_season": [2026] * row_count,
            "narrative": [f"Team frc{idx + 1} history." for idx in range(row_count)],
            "narrative_hash": [f"hash-{idx}" for idx in range(row_count)],
            "embedding_model": ["text-embedding-3-small"] * row_count,
            "llm_dim": [256] * row_count,
            "openai_narrative_vector": [
                [float((idx + offset) % 11) / 11.0 for offset in range(256)]
                for idx in range(row_count)
            ],
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


def test_temporal_quarantine_excludes_target_and_future_records_from_narrative():
    provider = FakeProvider()

    narrative = build_team_narrative(
        "frc111",
        provider.get_team("frc111"),
        provider.get_team_events("frc111"),
        provider.get_team_awards("frc111"),
        target_season=2026,
    )

    assert "2025 Midwest Regional" in narrative
    assert "Autonomous Award" in narrative
    assert "2026" not in narrative
    assert "FIRST Impact Award" not in narrative
    assert "2027" not in narrative
    assert "Quality Award" not in narrative


def test_narrative_cleaning_removes_json_artifacts_and_null_literals():
    narrative = clean_narrative_text('{ "name": null, "nickname": None, "team": "frc1" }')

    assert "{" not in narrative
    assert "}" not in narrative
    assert '"' not in narrative
    assert "null" not in narrative.lower()
    assert "None" not in narrative


def test_historical_awards_are_injected_next_to_event_text():
    table = build_prior_feature_table(
        ["frc111"],
        FakeProvider(),
        target_season=2026,
        opts=PriorOpts(cache_path=":memory:"),
        client=FakeOpenAIClient(),
    )

    narrative = table.loc[0, "narrative"]
    assert "Played 2025 Midwest Regional. Won Autonomous Award." in narrative


def test_openai_embedding_cache_uses_stable_request_hash(tmp_path):
    cache = OpenAIEmbeddingCache(tmp_path / "cache.sqlite")
    opts = PriorOpts(cache_path=str(tmp_path / "cache.sqlite"))
    client = FakeOpenAIClient()
    narratives = {"frc1": "Same text prior."}

    first = embed_narratives(narratives, opts, client=client, cache=cache)
    second = embed_narratives(narratives, opts, client=client, cache=cache)

    assert client.embeddings.calls == 1
    assert first == second


def test_embedding_batching_preserves_outputs_and_deduplicates_narratives(tmp_path):
    cache = OpenAIEmbeddingCache(tmp_path / "cache.sqlite")
    opts = PriorOpts(
        cache_path=str(tmp_path / "cache.sqlite"),
        embedding_batch_size=2,
    )
    client = FakeOpenAIClient()
    narratives = {
        "frc1": "Shared text.",
        "frc2": "Shared text.",
        "frc3": "Unique text A.",
        "frc4": "Unique text B.",
    }

    embeddings = embed_narratives(narratives, opts, client=client, cache=cache)

    assert set(embeddings) == set(narratives)
    assert client.embeddings.calls == 2
    assert [len(batch) for batch in client.embeddings.inputs] == [2, 1]
    assert embeddings["frc1"] == embeddings["frc2"]


def test_embedding_retry_recovers_from_fake_429_and_caches_result(tmp_path):
    cache = OpenAIEmbeddingCache(tmp_path / "cache.sqlite")
    opts = PriorOpts(
        cache_path=str(tmp_path / "cache.sqlite"),
        max_embedding_retries=2,
        embedding_retry_initial_delay=0.0,
        embedding_retry_max_delay=0.0,
        embedding_retry_jitter=0.0,
    )
    client = FakeOpenAIClient(failures=1)

    embeddings = embed_narratives({"frc1": "Retry text."}, opts, client=client, cache=cache)
    cached = embed_narratives({"frc1": "Retry text."}, opts, client=client, cache=cache)

    assert set(embeddings) == {"frc1"}
    assert cached == embeddings
    assert client.embeddings.calls == 2


def test_embedding_retry_raises_after_exhausting_retries(tmp_path):
    cache = OpenAIEmbeddingCache(tmp_path / "cache.sqlite")
    opts = PriorOpts(
        cache_path=str(tmp_path / "cache.sqlite"),
        max_embedding_retries=1,
        embedding_retry_initial_delay=0.0,
        embedding_retry_max_delay=0.0,
        embedding_retry_jitter=0.0,
    )
    client = FakeOpenAIClient(failures=3)

    with pytest.raises(FakeRateLimitError):
        embed_narratives({"frc1": "Retry text."}, opts, client=client, cache=cache)


def _calibrated_identity_prior_model(dropout_rate: float) -> UnifiedPriorAutoencoder:
    model = UnifiedPriorAutoencoder(PriorOpts(dropout_rate=dropout_rate))
    gain = torch.nn.functional.gelu(torch.tensor(5.0)).item()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        eye = torch.eye(16)
        model.encoder[0].weight[:16, :16].copy_(5.0 * eye)
        model.encoder[2].weight[:16, :16].copy_(eye / gain)
        model.decoder[0].weight[:16, :16].copy_(5.0 * eye)
        model.decoder[2].weight[:16, :16].copy_(eye / gain)
    return model


def test_autoencoder_input_dropout_makes_reconstruction_task_harder():
    x = torch.zeros((8, 256), dtype=torch.float32)
    x[:, :16] = 1.0
    no_dropout = _calibrated_identity_prior_model(0.0)
    with_dropout = _calibrated_identity_prior_model(0.3)
    no_dropout.train()
    with_dropout.train()

    torch.manual_seed(123)
    easy_loss = reconstruction_loss(no_dropout, x)
    torch.manual_seed(123)
    corrupted_loss = reconstruction_loss(with_dropout, x)

    assert easy_loss < corrupted_loss


def test_prior_autoencoder_checkpoint_contract(tmp_path):
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "pretrained_prior_2026.pt"
    write_prior_feature_table(_prior_table(2), features)
    opts = PriorOpts(epochs=2, batch_size=2, cache_path=str(tmp_path / "cache.sqlite"))

    result = train_prior_file(features, output, opts, verbose=False)
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)

    assert output.exists()
    assert checkpoint["target_season"] == 2026
    assert set(checkpoint["team_vectors"]) == {"frc1", "frc2"}
    assert checkpoint["team_vectors"]["frc1"].shape == (16,)
    assert checkpoint["prior_opts"]["llm_dim"] == 256
    assert set(result.team_vectors) == {"frc1", "frc2"}


def test_prior_inspection_exports_tables_and_pngs(tmp_path):
    features = tmp_path / "prior_features.parquet"
    checkpoint_path = tmp_path / "pretrained_prior_2026.pt"
    output = tmp_path / "inspection"
    write_prior_feature_table(_prior_table(3), features)
    opts = PriorOpts(epochs=2, batch_size=3, cache_path=str(tmp_path / "cache.sqlite"))
    train_prior_file(features, checkpoint_path, opts, verbose=False)

    inspection = inspect_prior_checkpoint(checkpoint_path, features, top_k=2)
    artifact_dir = write_prior_inspection_artifacts(inspection, output)

    assert len(inspection.latent_table) == 3
    assert {"pc1", "pc2", "pc3", "reconstruction_mse"}.issubset(
        inspection.latent_table.columns
    )
    assert len(inspection.nearest_neighbors) == 6
    assert inspection.sanity_checks["passed"].notna().all()
    for name in (
        "prior_latent_table.csv",
        "prior_nearest_neighbors.csv",
        "prior_training_history.csv",
        "prior_sanity_checks.csv",
        "prior_pca_pc1_pc2.png",
        "prior_pca_pc1_pc3.png",
        "prior_embedding_norm_hist.png",
        "prior_training_loss.png",
    ):
        path = artifact_dir / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_day_zero_prior_handoff_overwrites_matching_base_rows_only(tmp_path):
    opts = default_options()
    model = init_model(4, opts.latent_dim, 4, 1, opts)
    null_before = model.Z_base.weight[0].detach().clone()
    rookie_before = model.Z_base.weight[2].detach().clone()
    checkpoint = tmp_path / "prior.pt"
    torch.save({"team_vectors": {"frc1": torch.ones(opts.latent_dim)}}, checkpoint)

    applied = apply_prior_checkpoint_to_model(
        model,
        checkpoint,
        {"frc1": 1, "frc2": 2},
        latent_dim=opts.latent_dim,
    )

    assert applied == 1
    assert torch.allclose(model.Z_base.weight[1], torch.ones(opts.latent_dim))
    assert torch.allclose(model.Z_base.weight[0], null_before)
    assert torch.allclose(model.Z_base.weight[2], rookie_before)


def test_build_prior_features_cli_uses_mocked_provider_and_openai(tmp_path, monkeypatch):
    teams_from = tmp_path / "features.parquet"
    output = tmp_path / "prior.parquet"
    _feature_table(2).to_parquet(teams_from, engine="pyarrow", index=False)
    fake_client = FakeOpenAIClient()
    monkeypatch.setattr("latentstrat.cli._provider", lambda: FakeProvider())
    monkeypatch.setattr("latentstrat.pretrain_features._openai_client", lambda: fake_client)

    result = CliRunner().invoke(
        app,
        [
            "build-prior-features",
            "--target-season",
            "2026",
            "--teams-from",
            str(teams_from),
            "--output",
            str(output),
            "--cache-path",
            str(tmp_path / "cache.sqlite"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    table = pd.read_parquet(output, engine="pyarrow")
    assert set(table["team_key"]) == {"frc1", "frc2", "frc3", "frc4", "frc5", "frc6"}


def test_train_prior_cli_writes_checkpoint(tmp_path):
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "pretrained_prior.pt"
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
            "--batch-size",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert set(checkpoint["team_vectors"]) == {"frc1", "frc2"}


def test_inspect_prior_cli_writes_artifacts(tmp_path):
    features = tmp_path / "prior_features.parquet"
    checkpoint_path = tmp_path / "pretrained_prior.pt"
    output = tmp_path / "prior_inspection"
    write_prior_feature_table(_prior_table(3), features)
    train_prior_file(
        features,
        checkpoint_path,
        PriorOpts(epochs=1, batch_size=3, cache_path=str(tmp_path / "cache.sqlite")),
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


def test_train_features_cli_accepts_prior_checkpoint(tmp_path):
    features = tmp_path / "features.parquet"
    output = tmp_path / "artifacts"
    prior = tmp_path / "prior.pt"
    _feature_table().to_parquet(features, engine="pyarrow", index=False)
    torch.save({"team_vectors": {"frc1": torch.ones(default_options().latent_dim)}}, prior)

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
    checkpoint = torch.load(output / "v5_checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["prior_vectors_applied"] == 1
