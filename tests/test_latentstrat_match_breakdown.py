from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import torch
from typer.testing import CliRunner

from frc.providers.tba_provider import TbaJsonResponse
from latentstrat.cli import app
from latentstrat.world_model.match_breakdown.corpus import (
    MatchBreakdownCorpus,
    OpenApiMetadata,
    sync_match_breakdowns,
)
from latentstrat.world_model.match_breakdown.inspection import (
    EXPECTED_REPORT_FILES,
    LATENT_COLUMNS,
    MatchBreakdownInspectionOptions,
    add_component_percentiles,
    cross_season_neighbor_network,
    extreme_network_seeds,
    holdout_embeddings,
    load_match_breakdown_model,
    select_component_sources,
    shared_pca_coordinates,
    stratified_projection_sample,
    write_match_breakdown_inspection,
)
from latentstrat.world_model.match_breakdown.loss import grouped_type_aware_loss
from latentstrat.world_model.match_breakdown.model import MatchBreakdownAutoencoder
from latentstrat.world_model.match_breakdown.parse import (
    AllianceBreakdownRow,
    extract_alliance_rows,
    flatten_breakdown,
)
from latentstrat.world_model.match_breakdown.train import (
    MatchBreakdownTrainingOptions,
    _season_tensors,
    _split_indices,
    train_match_breakdown_encoder,
)
from latentstrat.world_model.match_breakdown.vectorize import (
    EncodedField,
    discover_season_schema,
    union_schema_audit,
    vectorize_flat_row,
)


def _match(
    season: int,
    number: int,
    *,
    event_key: str | None = None,
    breakdown: dict | None = None,
    score: int = 10,
) -> dict:
    event_key = event_key or f"{season}test"
    return {
        "key": f"{event_key}_qm{number}",
        "event_key": event_key,
        "comp_level": "qm",
        "set_number": 1,
        "match_number": number,
        "winning_alliance": "red",
        "alliances": {
            "red": {"score": score},
            "blue": {"score": score - 1},
        },
        "score_breakdown": breakdown,
    }


def _breakdown_2025(scale: int) -> dict:
    return {
        "red": {
            "autoPoints": scale,
            "foulPoints": 0,
            "endGameStatus": "Parked",
            "bonusAchieved": True,
        },
        "blue": {
            "autoPoints": scale - 1,
            "foulPoints": 1,
            "endGameStatus": "None",
            "bonusAchieved": False,
        },
    }


def _breakdown_2026(scale: int) -> dict:
    return {
        "red": {
            "totalPoints": scale,
            "adjustPoints": 0,
            "hubScore": {"autoCount": 0, "teleopCount": scale},
            "minorFoulCount": 0,
            "traversalAchieved": True,
        },
        "blue": {
            "totalPoints": scale - 2,
            "adjustPoints": 0,
            "hubScore": {"autoCount": 1, "teleopCount": scale - 3},
            "minorFoulCount": 1,
            "traversalAchieved": False,
        },
    }


class _FakeProvider:
    def __init__(self) -> None:
        self.calls = []
        self.responses = {
            "events/2026/simple": [
                {"key": "2026regional", "year": 2026, "event_type": 0},
                {"key": "2026foc", "year": 2026, "event_type": 6},
                {"key": "2026remote", "year": 2026, "event_type": 7},
                {"key": "2026offseason", "year": 2026, "event_type": 99},
            ],
            "event/2026regional/matches": [
                _match(2026, 1, event_key="2026regional", breakdown=_breakdown_2026(20)),
                _match(2026, 2, event_key="2026regional", breakdown=None, score=-1),
            ],
            "event/2026foc/matches": [],
            "event/2026remote/matches": [],
        }

    def get_json_response(self, path: str, *, etag=None, refresh=False):
        self.calls.append((path, etag, refresh))
        url = f"https://www.thebluealliance.com/api/v3/{path}"
        if refresh and etag:
            return TbaJsonResponse(url, 304, None, etag, False)
        return TbaJsonResponse(url, 200, self.responses[path], f"etag:{path}", False)


def test_flatten_extract_and_missing_breakdown_behavior():
    flat = flatten_breakdown({"z": [], "a": {"b": [1, {"c": True}]}})
    assert flat == {"a.b[0]": 1, "a.b[1].c": True, "z": []}
    assert len(extract_alliance_rows(_match(2026, 1, breakdown=_breakdown_2026(20)))) == 2
    assert extract_alliance_rows(_match(2026, 2, breakdown=None)) == []


def test_corpus_sync_filters_types_preserves_null_rows_and_refreshes_with_etags(tmp_path):
    corpus_path = tmp_path / "match_breakdowns.sqlite"
    provider = _FakeProvider()
    metadata = OpenApiMetadata(version="3.15.0", swagger_sha256="swagger-hash")
    first = sync_match_breakdowns(
        provider, corpus_path, start_season=2026, end_season=2026, openapi_metadata=metadata
    )
    assert first.status == "complete"
    assert first.counts["events_eligible"] == 1
    assert first.counts["events_excluded_type"] == 3
    assert first.counts["matches_upserted"] == 2

    corpus = MatchBreakdownCorpus(corpus_path)
    rows = corpus.raw_matches(start_season=2026, end_season=2026)
    assert len(rows) == 2
    assert rows[1]["score_breakdown"] is None
    with sqlite3.connect(corpus_path) as connection:
        connection.row_factory = sqlite3.Row
        stored = connection.execute(
            "select response_etag, payload_hash from source_responses "
            "where source_url like '%events/2026/simple'"
        ).fetchone()
        null_match = connection.execute(
            "select played, has_breakdown from matches where match_key = '2026regional_qm2'"
        ).fetchone()
        run = connection.execute(
            "select tba_api_version, swagger_sha256 from ingestion_runs"
        ).fetchone()
    assert stored["response_etag"] == "etag:events/2026/simple"
    assert stored["payload_hash"]
    assert tuple(null_match) == (0, 0)
    assert tuple(run) == ("3.15.0", "swagger-hash")

    refreshed = sync_match_breakdowns(
        provider,
        corpus_path,
        start_season=2026,
        end_season=2026,
        refresh=True,
        openapi_metadata=metadata,
    )
    assert refreshed.status == "complete"
    assert any(etag == "etag:events/2026/simple" for _, etag, _ in provider.calls)
    assert len(corpus.raw_matches(start_season=2026, end_season=2026)) == 2


def test_corpus_sync_optionally_includes_foc_and_remote(tmp_path):
    result = sync_match_breakdowns(
        _FakeProvider(),
        tmp_path / "match_breakdowns.sqlite",
        start_season=2026,
        end_season=2026,
        include_foc=True,
        include_remote=True,
        openapi_metadata=OpenApiMetadata(version="3.15.0", swagger_sha256="hash"),
    )
    assert result.event_types == (0, 1, 2, 3, 4, 5, 6, 7)
    assert result.counts["events_eligible"] == 3
    assert MatchBreakdownTrainingOptions().event_types == (0, 1, 2, 3, 4, 5)
    assert MatchBreakdownTrainingOptions(include_foc=True, include_remote=True).event_types == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )


def test_per_season_schema_is_typed_stable_and_not_union_padded():
    row_a = AllianceBreakdownRow(
        "2025test_qm1:red",
        2025,
        "2025test",
        "2025test_qm1",
        "qm",
        1,
        1,
        "red",
        10,
        9,
        "red",
        {"autoPoints": 0, "endGameStatus": "Parked", "bonusAchieved": True},
    )
    row_b = AllianceBreakdownRow(
        "2025test_qm2:red",
        2025,
        "2025test",
        "2025test_qm2",
        "qm",
        1,
        2,
        "red",
        12,
        11,
        "red",
        {"autoPoints": 1, "endGameStatus": "None"},
    )
    schema = discover_season_schema([row_a, row_b], 2025)
    assert discover_season_schema([row_b, row_a], 2025) == schema
    assert [field.name for field in schema.encoded_fields if field.kind == "categorical"] == [
        "endGameStatus=None",
        "endGameStatus=Parked",
    ]
    values, mask = vectorize_flat_row(row_a.flat_breakdown, schema.encoded_fields)
    auto_index = [field.name for field in schema.encoded_fields].index("autoPoints")
    assert values[auto_index] == 0
    assert mask[auto_index] == 1
    values, mask = vectorize_flat_row({}, schema.encoded_fields)
    category_indices = [
        index for index, field in enumerate(schema.encoded_fields) if field.kind == "categorical"
    ]
    assert not mask[category_indices].any()
    assert not values[category_indices].any()

    schema_2026 = discover_season_schema(
        extract_alliance_rows(_match(2026, 1, breakdown=_breakdown_2026(20))), 2026
    )
    audit = union_schema_audit({2025: schema, 2026: schema_2026})
    assert audit["training_uses_union_padding"] is False
    assert audit["season_widths"]["2025"] != audit["season_widths"]["2026"]


def test_vectorizer_rejects_non_boolean_values_for_boolean_fields():
    fields = [EncodedField("enabled", "bonus", "boolean", "enabled")]
    try:
        vectorize_flat_row({"enabled": 1}, fields)
    except ValueError as exc:
        assert "Expected boolean" in str(exc)
    else:
        raise AssertionError("Expected strict boolean rejection.")


def test_shared_bottleneck_routes_distinct_season_widths_and_loss_is_differentiable():
    model = MatchBreakdownAutoencoder({2025: 3, 2026: 5})
    values = torch.zeros((2, 3))
    mask = torch.ones((2, 3))
    reconstruction, latent = model(2025, values, mask)
    assert reconstruction.shape == (2, 3)
    assert latent.shape == (2, 16)
    fields = [
        EncodedField(f"value{index}", "auto", "numeric", f"value{index}")
        for index in range(3)
    ]
    loss = grouped_type_aware_loss(reconstruction, values, torch.zeros_like(mask), fields)
    loss.backward()
    assert loss.item() == 0


def test_training_exports_eval_and_all_data_artifacts_without_normalizers(tmp_path):
    corpus_path = tmp_path / "match_breakdowns.sqlite"
    corpus = MatchBreakdownCorpus(corpus_path)
    fetched_at = "2026-01-01T00:00:00+00:00"
    for season, builder in ((2025, _breakdown_2025), (2026, _breakdown_2026)):
        corpus.upsert_event(
            {"key": f"{season}test", "year": season, "event_type": 0},
            source_url=f"https://example.test/events/{season}/simple",
            response_etag=f"etag-events-{season}",
            fetched_at=fetched_at,
        )
        for number in range(1, 4):
            corpus.upsert_match(
                _match(season, number, breakdown=builder(20 + number)),
                source_url=f"https://example.test/event/{season}test/matches",
                response_etag=f"etag-{season}",
                fetched_at=fetched_at,
            )
    options = MatchBreakdownTrainingOptions(
        start_season=2025,
        end_season=2026,
        epochs=1,
        seasons_per_step=2,
        rows_per_season=2,
        seed=2026,
    )
    result = train_match_breakdown_encoder(
        corpus_path,
        tmp_path / "artifacts",
        tmp_path / "features.parquet",
        options,
    )
    for name in (
        "eval_model.pt",
        "model.pt",
        "embeddings.parquet",
        "schema.json",
        "union_schema_audit.json",
        "bundle.json",
        "validation_report.json",
        "training_history.csv",
    ):
        assert (result.output_dir / name).exists()
    assert not list(result.output_dir.glob("*norm*"))
    assert result.bundle["promotion_eligible"] is False
    assert result.bundle["provenance"] == {
        "validation_metrics": "eval_model.pt",
        "exported_embeddings": "model.pt",
    }
    assert result.bundle["season_widths"]["2025"] != result.bundle["season_widths"]["2026"]
    audit = json.loads((result.output_dir / "union_schema_audit.json").read_text())
    assert audit["training_uses_union_padding"] is False
    widths = torch.load(result.output_dir / "model.pt", weights_only=False)["season_widths"]
    assert len(widths) == 2


def test_match_grouped_split_keeps_alliances_together():
    rows = []
    for number in range(1, 5):
        rows.extend(extract_alliance_rows(_match(2026, number, breakdown=_breakdown_2026(20))))
    tensors = _season_tensors(rows)
    train, validation = _split_indices(tensors, fraction=0.25, seed=2026)
    data = tensors[2026]
    train_keys = {data.rows[index].match_key for index in train[2026]}
    validation_keys = {data.rows[index].match_key for index in validation[2026]}
    assert train_keys.isdisjoint(validation_keys)
    assert len(validation[2026]) == 2


def _train_toy_artifact(tmp_path):
    corpus_path = tmp_path / "match_breakdowns.sqlite"
    corpus = MatchBreakdownCorpus(corpus_path)
    fetched_at = "2026-01-01T00:00:00+00:00"
    for season, builder in ((2025, _breakdown_2025), (2026, _breakdown_2026)):
        corpus.upsert_event(
            {"key": f"{season}test", "year": season, "event_type": 0},
            source_url=f"https://example.test/events/{season}/simple",
            response_etag=f"etag-events-{season}",
            fetched_at=fetched_at,
        )
        for number in range(1, 7):
            corpus.upsert_match(
                _match(season, number, breakdown=builder(20 + number)),
                source_url=f"https://example.test/event/{season}test/matches",
                response_etag=f"etag-{season}",
                fetched_at=fetched_at,
            )
    return train_match_breakdown_encoder(
        corpus_path,
        tmp_path / "artifacts",
        tmp_path / "features.parquet",
        MatchBreakdownTrainingOptions(
            start_season=2025,
            end_season=2026,
            epochs=1,
            seasons_per_step=2,
            rows_per_season=2,
            seed=2026,
            validation_fraction=0.25,
        ),
    )


def _latent_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    latents = table.pop("latent").to_list()
    for index, column in enumerate(LATENT_COLUMNS):
        table[column] = [float(values[index]) for values in latents]
    return table


def test_component_sources_use_one_safe_aggregate_and_preserve_nan_for_missing_season():
    raw = pd.DataFrame(
        [
            {
                "row_id": "2025test_qm1:red",
                "season": 2025,
                "flat_breakdown_json": json.dumps(
                    {
                        "totalAutoPoints": 10,
                        "autoPoints": 999,
                        "endGameTowerPoints": 3,
                        "foulPoints": 0,
                    }
                ),
            },
            {
                "row_id": "2025test_qm2:red",
                "season": 2025,
                "flat_breakdown_json": json.dumps(
                    {
                        "totalAutoPoints": 20,
                        "autoPoints": 999,
                        "endGameTowerPoints": 7,
                        "foulPoints": 1,
                    }
                ),
            },
            {
                "row_id": "2015test_qm1:red",
                "season": 2015,
                "flat_breakdown_json": json.dumps({"auto_points": 4, "foul_points": 2}),
            },
        ]
    )
    geometry = raw[["row_id", "season"]].assign(alliance_score=[1, 2, 3])
    selected = select_component_sources(raw)
    assert selected["2025"]["auto"] == "totalAutoPoints"
    assert selected["2025"]["endgame"] == "endGameTowerPoints"
    assert selected["2015"]["endgame"] is None
    enriched = add_component_percentiles(geometry, raw, selected)
    assert enriched.loc[enriched["season"] == 2025, "auto_value"].tolist() == [10.0, 20.0]
    assert enriched.loc[enriched["season"] == 2015, "endgame_percentile"].isna().all()


def test_projection_sampling_is_deterministic_and_balanced_by_season_and_decile():
    rows = []
    for season in (2025, 2026):
        for index in range(30):
            rows.append(
                {
                    "row_id": f"{season}test_qm{index}:red",
                    "season": season,
                    "score_percentile": (index + 1) / 30,
                }
            )
    table = pd.DataFrame(rows)
    first = stratified_projection_sample(table, max_rows=20, per_season_limit=10, seed=2026)
    second = stratified_projection_sample(table, max_rows=20, per_season_limit=10, seed=2026)
    pd.testing.assert_frame_equal(first, second)
    assert first["season"].value_counts().to_dict() == {2025: 10, 2026: 10}
    assert first.groupby("season")["score_decile"].nunique().min() >= 8


def test_shared_pca_uses_production_frame_after_orthogonal_holdout_alignment():
    rng = np.random.default_rng(2026)
    matrix = rng.normal(size=(10, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    production = _latent_table(
        [
            {"row_id": f"row-{index}", "season": 2026, "latent": values}
            for index, values in enumerate(matrix)
        ]
    )
    holdout = _latent_table(
        [
            {"row_id": f"row-{index}", "season": 2026, "latent": values @ q}
            for index, values in enumerate(matrix[:6])
        ]
    )
    production_out, holdout_out, explained, alignment = shared_pca_coordinates(
        production, holdout
    )
    assert alignment["kind"] == "orthogonal-procrustes"
    assert {"pc1", "pc2", "pc3"}.issubset(production_out)
    assert len(explained) == min(matrix.shape)
    np.testing.assert_allclose(
        holdout_out[["pc1", "pc2", "pc3"]],
        production_out.loc[:5, ["pc1", "pc2", "pc3"]],
        atol=1e-10,
    )


def test_cross_season_network_edges_respect_limits_and_route_across_seasons():
    rows = []
    for season in (2024, 2025, 2026):
        for index in range(6):
            latent = np.zeros(16)
            latent[0] = season - 2024 + index / 100
            rows.append(
                {
                    "row_id": f"{season}test_qm{index}:red",
                    "season": season,
                    "score_percentile": 1.0 if index < 2 else 0.2,
                    "auto_percentile": 1.0 if index == 0 else 0.2,
                    "endgame_percentile": 1.0 if index == 1 else 0.2,
                    "fouls_percentile": 1.0 if index == 0 else 0.2,
                    "latent": latent,
                }
            )
    production = _latent_table(rows)
    seeds = extreme_network_seeds(production, node_limit=5, seed=2026)
    _, edges = cross_season_neighbor_network(production, seeds, neighbors_per_node=2)
    assert len(seeds) == 5
    assert len(edges) == 10
    assert (edges["source_season"] != edges["target_season"]).all()


def test_inspection_reloads_models_reconstructs_holdout_and_writes_static_report(tmp_path):
    result = _train_toy_artifact(tmp_path)
    before = (result.output_dir / "bundle.json").read_bytes()
    _, eval_checkpoint = load_match_breakdown_model(result.output_dir / "eval_model.pt")
    _, production_checkpoint = load_match_breakdown_model(result.output_dir / "model.pt")
    assert eval_checkpoint["provenance"] == "holdout-evaluation-only"
    assert production_checkpoint["provenance"] == "all-data-embeddings"
    raw = pd.read_parquet(result.alliance_features_path)
    holdout, validation = holdout_embeddings(raw, result.output_dir / "eval_model.pt")
    assert len(holdout) == sum(len(indices) for indices in validation.values())
    assert holdout.groupby("match_key").size().eq(2).all()

    inspection = write_match_breakdown_inspection(
        result.output_dir,
        result.output_dir / "inspection",
        MatchBreakdownInspectionOptions(
            tsne_max_rows=20,
            production_tsne_per_season=10,
            holdout_tsne_per_season=10,
            network_node_limit=8,
            neighbors_per_node=1,
            parallel_rows_per_slice=4,
            tsne_iterations=250,
        ),
    )
    for name in (*EXPECTED_REPORT_FILES, "inspection_manifest.json"):
        assert (inspection.output_dir / name).exists()
    assert (result.output_dir / "bundle.json").read_bytes() == before
    assert inspection.manifest["geometry"]["production"]["rows"] == 24
    assert inspection.manifest["geometry"]["holdout"]["rows"] == 8
    assert set(inspection.manifest["generated_files"]) == set(EXPECTED_REPORT_FILES)
    profiles = pd.read_csv(inspection.output_dir / "latent_archetype_profiles.csv")
    assert "archetype" in profiles


def test_inspection_cli_executes_against_tiny_artifact(tmp_path):
    result = _train_toy_artifact(tmp_path)
    output = result.output_dir / "cli-inspection"
    runner = CliRunner()
    response = runner.invoke(
        app,
        [
            "inspect-match-breakdown-encoder",
            "--artifact-dir",
            str(result.output_dir),
            "--output",
            str(output),
            "--tsne-max-rows",
            "12",
            "--network-node-limit",
            "6",
            "--neighbors-per-node",
            "1",
        ],
    )
    assert response.exit_code == 0, response.output
    assert "production_rows=24 holdout_rows=8" in response.output
    assert (output / "inspection_manifest.json").exists()
