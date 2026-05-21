import numpy as np
import pandas as pd
import pytest

from frc.datastore import FRCDataStore
from frc.models import Match, MatchAlliance
from latentstrat.config import default_options
from latentstrat.data import (
    apply_target_stats,
    build_season_alliance_table,
    build_season_match_table,
    fit_target_stats,
    make_split,
    make_team_index_map,
)


def _breakdown(score):
    return {
        "totalPoints": score,
        "totalAutoPoints": 20,
        "totalTeleopPoints": score - 20,
        "totalTowerPoints": 35,
        "autoTowerPoints": 10,
        "endGameTowerPoints": 25,
        "foulPoints": 3,
        "majorFoulCount": 0,
        "minorFoulCount": 1,
    }


def _match(match_key, red_score=120, blue_score=100, breakdown=True):
    red = {"team_keys": ["frc1", "frc2", "frc3"], "score": red_score, "surrogate_team_keys": ["frc3"], "dq_team_keys": ["frc2"]}
    blue = {"team_keys": ["frc4", "frc5", "frc6"], "score": blue_score, "surrogate_team_keys": [], "dq_team_keys": []}
    data = {
        "key": match_key,
        "event_key": "2026test",
        "comp_level": "qm",
        "set_number": 1,
        "match_number": int(match_key.rsplit("qm", 1)[1]),
        "time": 1760000000,
        "actual_time": np.nan,
        "post_result_time": np.nan,
        "alliances": {"red": red, "blue": blue},
    }
    if breakdown:
        data["score_breakdown"] = {"red": _breakdown(red_score), "blue": _breakdown(blue_score)}
    else:
        data["score_breakdown"] = {}
    return Match(
        match_key=match_key,
        red_alliance=MatchAlliance.model_validate(red),
        blue_alliance=MatchAlliance.model_validate(blue),
        tba_data=data,
        event_key="2026test",
    )


def test_build_tables_filters_and_maps_zero_based_indices():
    opts = default_options()
    store = FRCDataStore()
    store.add_match([_match("2026test_qm1"), _match("2026test_qm2", -1, -1), _match("2026test_qm3", breakdown=False)])
    metadata = pd.DataFrame([{"key": "2026test", "week": 1}])

    alliance = build_season_alliance_table(store, opts, event_metadata=metadata)
    match_table = build_season_match_table(store, opts, event_metadata=metadata)
    match_table, team_map = make_team_index_map(match_table)

    assert len(alliance) == 2
    assert len(match_table) == 1
    assert match_table["red_total_score"].iloc[0] == 120
    assert match_table["blue_total_score"].iloc[0] == 100
    assert match_table["win_margin"].iloc[0] == 20
    assert bool(match_table["red_win"].iloc[0])
    assert team_map["frc1"] == 0
    assert int(match_table["red_team_1_idx"].iloc[0]) == 0
    assert str(match_table["red_team_1_idx"].dtype) == "Int64"


def test_missing_required_breakdown_field_errors():
    opts = default_options()
    bad = _match("2026test_qm1")
    del bad.tba_data["score_breakdown"]["red"]["totalAutoPoints"]
    store = FRCDataStore()
    store.add_match(bad)

    with pytest.raises(KeyError):
        build_season_alliance_table(store, opts)


def test_split_and_target_stats_do_not_leak_validation_rows():
    opts = default_options()
    rows = []
    for idx in range(12):
        rows.append(
            {
                "event_key": "2026test",
                "sort_ordinal": idx + 1,
                "red_total_score": 10 + idx,
                "blue_total_score": 20 + idx,
                "win_margin": -10,
                "fouls_drawn": 0,
            }
        )
    table = pd.DataFrame(rows)
    split = make_split(table, opts, policy="chronological-holdout", validation_fraction=0.25)
    before = fit_target_stats(table, split.train_mask, opts)
    modified = table.copy()
    modified.loc[split.validation_mask, "red_total_score"] = 1e9
    after = fit_target_stats(modified, split.train_mask, opts)

    assert not np.any(split.train_mask & split.validation_mask)
    np.testing.assert_allclose(after.mu, before.mu)
    np.testing.assert_allclose(after.sigma, before.sigma)


def test_apply_target_stats_adds_z_columns():
    opts = default_options()
    table = pd.DataFrame(
        {
            "sort_ordinal": [1, 2, 3],
            "red_total_score": [1.0, 3.0, 5.0],
            "blue_total_score": [2.0, 4.0, 6.0],
            "win_margin": [-1.0, -1.0, -1.0],
            "fouls_drawn": [0.0, 0.0, 0.0],
        }
    )
    split = make_split(table, opts, validation_fraction=0.34)
    stats = fit_target_stats(table, split.train_mask, opts)
    out = apply_target_stats(table, stats)
    assert "red_total_score_z" in out.columns
