import json

import pandas as pd

from latentstrat.baseline_manifest import REQUIRED_ARCHIVE_FILES, write_baseline_manifest


def test_baseline_manifest_captures_hashes_metrics_and_regeneration_command(tmp_path):
    archive = tmp_path / "artifacts" / "baselines" / "v5.8"
    for relative in REQUIRED_ARCHIVE_FILES:
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (archive / "commit.txt").write_text("abc123\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"fold_number": 1, "val_win_brier": 0.21, "val_win_log_loss": 0.62},
            {"fold_number": "AVERAGE", "val_win_brier": 0.20, "val_win_log_loss": 0.60},
        ]
    ).to_csv(archive / "walk-forward" / "walk_forward_metrics.csv", index=False)
    checkpoint = archive / "walk-forward" / "fold_checkpoints" / "fold_01.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "baselines" / "v5.8-baseline.json"

    write_baseline_manifest(
        archive,
        output,
        tag="v5.8-baseline",
        commit_sha="abc123",
        regeneration_commands=["latentstrat validate-walk-forward --epochs 50"],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["baseline_ref"] == "v5.8-baseline"
    assert payload["commit_sha"] == "abc123"
    assert payload["summary_metrics"]["val_win_brier"] == 0.2
    assert payload["regeneration_commands"] == [
        "latentstrat validate-walk-forward --epochs 50"
    ]
    assert payload["file_sha256"]["prior_checkpoint.pt"]
    assert payload["file_sha256"]["walk-forward/fold_checkpoints/fold_01.pt"]
