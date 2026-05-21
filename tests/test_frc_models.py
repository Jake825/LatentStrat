from frc.models import TbaMatch


def test_tba_match_ignores_unknown_fields_and_preserves_raw_breakdown():
    match = TbaMatch.model_validate(
        {
            "key": "2026test_qm1",
            "event_key": "2026test",
            "comp_level": "qm",
            "set_number": 1,
            "match_number": 1,
            "alliances": {
                "red": {"team_keys": ["frc1", "frc2", "frc3"], "score": 120},
                "blue": {"team_keys": ["frc4", "frc5", "frc6"], "score": 100},
            },
            "score_breakdown": {
                "red": {"totalPoints": 120, "foulPoints": 3, "newMidseasonField": 99},
                "blue": {"totalPoints": 100, "foulPoints": 1},
            },
            "unexpected": {"nested": True},
        }
    )

    assert match.key == "2026test_qm1"
    assert match.alliance("red").team_keys == ["frc1", "frc2", "frc3"]
    assert match.breakdown("red")["newMidseasonField"] == 99
