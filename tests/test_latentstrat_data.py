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
    canonical_event_week_table,
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
        "autoTowerRobot1": "Level1",
        "autoTowerRobot2": "None",
        "autoTowerRobot3": "Level2",
        "endGameTowerRobot1": "Level1",
        "endGameTowerRobot2": "Level2",
        "endGameTowerRobot3": "Level3",
        "foulPoints": 3,
        "majorFoulCount": 0,
        "minorFoulCount": 1,
        "energizedAchieved": True,
        "superchargedAchieved": False,
        "traversalAchieved": True,
        "g206Penalty": False,
        "hubScore": {
            "autoCount": 2,
            "transitionCount": 3,
            "shift1Count": 4,
            "shift2Count": 5,
            "shift3Count": 6,
            "shift4Count": 7,
            "endgameCount": 8,
        },
    }


def _match(match_key, red_score=120, blue_score=100, breakdown=True):
    red = {
        "team_keys": ["frc1", "frc2", "frc3"],
        "score": red_score,
        "surrogate_team_keys": ["frc3"],
        "dq_team_keys": ["frc2"],
    }
    blue = {
        "team_keys": ["frc4", "frc5", "frc6"],
        "score": blue_score,
        "surrogate_team_keys": [],
        "dq_team_keys": [],
    }
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
    store.add_match(
        [
            _match("2026test_qm1"),
            _match("2026test_qm2", -1, -1),
            _match("2026test_qm3", breakdown=False),
        ]
    )
    metadata = pd.DataFrame([{"key": "2026test", "week": 1}])

    alliance = build_season_alliance_table(store, opts, event_metadata=metadata)
    match_table = build_season_match_table(store, opts, event_metadata=metadata)
    match_table, team_map = make_team_index_map(match_table)

    assert len(alliance) == 4
    assert len(match_table) == 2
    assert match_table["red_total_score"].iloc[0] == 120
    assert match_table["blue_total_score"].iloc[0] == 100
    assert match_table["win_margin"].iloc[0] == 20
    assert bool(match_table["red_win"].iloc[0])
    assert match_table["red_auto_pts"].iloc[0] == 20
    assert match_table["blue_teleop_pts"].iloc[0] == 80
    assert match_table["red_team_3_endgame_status"].iloc[0] == "Level3"
    assert match_table["red_atomic_auto_count"].iloc[0] == 2
    assert match_table["blue_atomic_shift4_count"].iloc[0] == 7
    assert match_table["red_bonus_energized"].iloc[0] == 1.0
    assert match_table["red_bonus_supercharged"].iloc[0] == 0.0
    assert match_table["red_committed_foul_pts"].iloc[0] == 3
    assert np.isnan(match_table["red_atomic_auto_count"].iloc[1])
    assert team_map["frc1"] == 1
    assert int(match_table["red_team_1_base_idx"].iloc[0]) == 1
    assert int(match_table["red_team_1_event_idx"].iloc[0]) == 1
    assert not bool(match_table["red_team_1_missing_team_mask"].iloc[0])
    assert str(match_table["red_team_1_idx"].dtype) == "Int64"


def test_canonical_event_week_table_bundles_week_zero_and_infers_missing():
    metadata = pd.DataFrame(
        {
            "key": ["2026w0", "2026missing", "2026w2"],
            "week": [0, np.nan, 2],
            "start_date": ["2026-03-01", "2026-03-08", "2026-03-15"],
        }
    )

    weeks = canonical_event_week_table(metadata)

    by_key = weeks.set_index("event_key")
    assert int(by_key.loc["2026w0", "event_week"]) == 1
    assert int(by_key.loc["2026missing", "event_week"]) == 2
    assert int(by_key.loc["2026w2", "event_week"]) == 3
    assert int(by_key.loc["2026w0", "raw_event_week"]) == 0


def test_v56_team_indexing_uses_numeric_team_number():
    table = pd.DataFrame(
        {
            "event_key": ["2026test"],
            "red_team_1_key": ["frc2290"],
            "red_team_2_key": [""],
            "red_team_3_key": ["frc6"],
            "blue_team_1_key": ["frc4"],
            "blue_team_2_key": ["frc5"],
            "blue_team_3_key": ["frc1"],
        }
    )

    indexed, team_map = make_team_index_map(table)

    assert team_map["frc2290"] == 2290
    assert int(indexed["red_team_1_base_idx"].iloc[0]) == 2290
    assert int(indexed["red_team_2_base_idx"].iloc[0]) == 0
    assert bool(indexed["red_team_2_missing_team_mask"].iloc[0])


def test_v56_team_indexing_errors_above_prior_bounds():
    table = pd.DataFrame(
        {
            "event_key": ["2026test"],
            "red_team_1_key": ["frc12501"],
            "red_team_2_key": ["frc2"],
            "red_team_3_key": ["frc3"],
            "blue_team_1_key": ["frc4"],
            "blue_team_2_key": ["frc5"],
            "blue_team_3_key": ["frc6"],
        }
    )

    with pytest.raises(ValueError, match="Rebuild the V5.6 prior"):
        make_team_index_map(table)


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


def _event_comp_table(groups: list[tuple[str, str, int]]) -> pd.DataFrame:
    rows = []
    ordinal = 1
    for event_key, comp_level, count in groups:
        for idx in range(count):
            rows.append(
                {
                    "event_key": event_key,
                    "match_key": f"{event_key}_{comp_level}{idx + 1}",
                    "comp_level": comp_level,
                    "sort_ordinal": ordinal,
                    "red_total_score": 10 + ordinal,
                    "blue_total_score": 20 + ordinal,
                    "win_margin": -10,
                    "fouls_drawn": 0,
                }
            )
            ordinal += 1
    return pd.DataFrame(rows)


def test_stratified_event_comp_split_balances_events_and_match_types():
    table = _event_comp_table(
        [
            ("2026a", "qm", 4),
            ("2026a", "sf", 2),
            ("2026b", "qm", 4),
            ("2026b", "f", 2),
        ]
    )

    split = make_split(
        table,
        default_options(),
        policy="stratified-event-comp",
        validation_fraction=0.5,
        random_seed=7,
    )

    validation = table[split.validation_mask].copy()
    validation["comp_bucket"] = np.where(validation["comp_level"] == "qm", "qm", "elim")
    assert not np.any(split.train_mask & split.validation_mask)
    assert set(validation["event_key"]) == {"2026a", "2026b"}
    assert set(validation["comp_bucket"]) == {"qm", "elim"}
    for _, rows in table.groupby(["event_key", table["comp_level"].eq("qm")], sort=True):
        val_count = int(np.sum(split.validation_mask[rows.index]))
        assert 0 < val_count < len(rows)


def test_stratified_event_comp_split_keeps_singletons_in_training():
    table = _event_comp_table([("2026a", "qm", 3), ("2026b", "sf", 1)])

    split = make_split(
        table,
        default_options(),
        policy="stratified-event-comp",
        validation_fraction=0.5,
        random_seed=1,
    )

    singleton_index = table.index[table["event_key"] == "2026b"][0]
    assert split.train_mask[singleton_index]
    assert not split.validation_mask[singleton_index]
    assert np.any(split.validation_mask)


def test_stratified_event_comp_split_is_seed_deterministic():
    table = _event_comp_table([("2026a", "qm", 8), ("2026a", "sf", 6)])

    first = make_split(table, policy="stratified-event-comp", random_seed=2026)
    second = make_split(table, policy="stratified-event-comp", random_seed=2026)

    np.testing.assert_array_equal(first.validation_mask, second.validation_mask)


def test_stratified_event_comp_split_falls_back_when_no_strata_can_split():
    table = _event_comp_table([("2026a", "qm", 1), ("2026b", "sf", 1)])

    split = make_split(
        table,
        default_options(),
        policy="stratified-event-comp",
        validation_fraction=0.5,
        random_seed=1,
    )

    assert split.policy == "stratified-event-comp"
    assert split.fallback_reason == "no_stratifiable_event_comp_strata"
    assert split.validation_mask.tolist() == [False, True]


def test_apply_target_stats_adds_z_columns():
    opts = default_options()
    table = pd.DataFrame(
        {
            "sort_ordinal": [1, 2, 3],
            "red_auto_pts": [1.0, 3.0, 5.0],
            "red_teleop_pts": [2.0, 4.0, 6.0],
            "blue_auto_pts": [0.0, 1.0, 2.0],
            "blue_teleop_pts": [3.0, 4.0, 5.0],
        }
    )
    split = make_split(table, opts, validation_fraction=0.34)
    stats = fit_target_stats(table, split.train_mask, opts)
    out = apply_target_stats(table, stats)
    assert "red_auto_pts_z" in out.columns
