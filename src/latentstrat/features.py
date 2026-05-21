"""Parquet feature-store helpers for LatentStrat."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlmodel import Session, SQLModel, select

from frc.datastore import FRCDataStore
from frc.importers import TBAImporter
from frc.providers.tba_provider import TbaProvider
from frc.scouting import (
    EventScouting,
    MatchAllianceScouting,
    MatchScouting,
    TeamEventScouting,
    TeamMatchScouting,
    TeamScouting,
    create_scouting_engine,
)
from latentstrat.baselines import fit_baselines
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.data import (
    apply_target_stats,
    build_season_match_table,
    fit_target_stats,
    make_split,
    make_team_index_map,
)
from latentstrat.evaluation import EvaluationReport, evaluate_model
from latentstrat.training import TrainingDiagnostics, train_model

PARQUET_ENGINE = "pyarrow"


@dataclass
class FeatureTrainingResult:
    table: pd.DataFrame
    prepared: pd.DataFrame
    team_index_map: dict[str, int]
    history: pd.DataFrame
    diagnostics: TrainingDiagnostics
    report: EvaluationReport


def _validate_feature_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Feature table is empty.")
    return table.reset_index(drop=True)


def _scouting_table(session: Session, model: type[SQLModel]) -> pd.DataFrame:
    rows = session.exec(select(model)).all()
    columns = list(model.model_fields)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([row.model_dump() for row in rows], columns=columns)


def _prefixed(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return table.add_prefix(prefix) if not table.empty else table.add_prefix(prefix)


def _merge_prefixed(
    table: pd.DataFrame,
    right: pd.DataFrame,
    *,
    prefix: str,
    left_on: list[str],
    right_on: list[str],
) -> pd.DataFrame:
    if right.empty:
        return table
    prefixed = _prefixed(right, prefix)
    merged = table.merge(prefixed, how="left", left_on=left_on, right_on=right_on)
    return merged.drop(columns=right_on, errors="ignore")


def merge_scouting_features(table: pd.DataFrame, scouting_db_path: str | Path) -> pd.DataFrame:
    path = Path(scouting_db_path)
    if not path.exists():
        return table.copy()

    engine = create_scouting_engine(path)
    with Session(engine) as session:
        team_scouting = _scouting_table(session, TeamScouting)
        event_scouting = _scouting_table(session, EventScouting)
        match_scouting = _scouting_table(session, MatchScouting)
        team_event_scouting = _scouting_table(session, TeamEventScouting)
        match_alliance_scouting = _scouting_table(session, MatchAllianceScouting)
        team_match_scouting = _scouting_table(session, TeamMatchScouting)

    out = table.copy()
    out = _merge_prefixed(
        out,
        event_scouting,
        prefix="event_scout_",
        left_on=["event_key"],
        right_on=["event_scout_event_key"],
    )
    out = _merge_prefixed(
        out,
        match_scouting,
        prefix="match_scout_",
        left_on=["match_key"],
        right_on=["match_scout_match_key"],
    )
    for color in ("red", "blue"):
        alliance = match_alliance_scouting[
            match_alliance_scouting.get("alliance_color", pd.Series(dtype=str)) == color
        ]
        out = _merge_prefixed(
            out,
            alliance,
            prefix=f"{color}_alliance_scout_",
            left_on=["match_key"],
            right_on=[f"{color}_alliance_scout_match_key"],
        )

    for color in ("red", "blue"):
        for slot in (1, 2, 3):
            slot_prefix = f"{color}_team_{slot}"
            team_col = f"{slot_prefix}_key"
            out = _merge_prefixed(
                out,
                team_scouting,
                prefix=f"{slot_prefix}_pit_",
                left_on=[team_col],
                right_on=[f"{slot_prefix}_pit_team_key"],
            )
            out = _merge_prefixed(
                out,
                team_event_scouting,
                prefix=f"{slot_prefix}_event_scout_",
                left_on=[team_col, "event_key"],
                right_on=[
                    f"{slot_prefix}_event_scout_team_key",
                    f"{slot_prefix}_event_scout_event_key",
                ],
            )
            team_match = team_match_scouting[
                team_match_scouting.get("alliance_color", pd.Series(dtype=str)) == color
            ]
            out = _merge_prefixed(
                out,
                team_match,
                prefix=f"{slot_prefix}_match_scout_",
                left_on=[team_col, "match_key"],
                right_on=[
                    f"{slot_prefix}_match_scout_team_key",
                    f"{slot_prefix}_match_scout_match_key",
                ],
            )
    return out


def build_event_feature_table(
    event_key: str,
    opts: LatentStratOptions | None = None,
    *,
    provider: TbaProvider | None = None,
    scouting_db_path: str | Path | None = None,
) -> pd.DataFrame:
    opts = opts or default_options()
    provider = provider or TbaProvider()
    store = FRCDataStore()
    event = TBAImporter.import_event(provider, store, event_key)
    metadata = pd.DataFrame([event.model_dump()])
    table = _validate_feature_table(build_season_match_table(store, opts, event_metadata=metadata))
    if scouting_db_path is not None:
        table = merge_scouting_features(table, scouting_db_path)
    return table


def build_season_feature_table(
    season: int,
    opts: LatentStratOptions | None = None,
    *,
    provider: TbaProvider | None = None,
    event_limit: int | None = None,
    scouting_db_path: str | Path | None = None,
) -> pd.DataFrame:
    opts = (opts or default_options()).model_copy(update={"season": season})
    provider = provider or TbaProvider()
    store = FRCDataStore()
    events = TBAImporter.import_season(provider, store, season, event_limit=event_limit)
    table = _validate_feature_table(build_season_match_table(store, opts, event_metadata=events))
    if scouting_db_path is not None:
        table = merge_scouting_features(table, scouting_db_path)
    return table


def write_feature_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    table = _validate_feature_table(table)
    idx_columns = [column for column in table.columns if column.endswith("_idx")]
    z_columns = [column for column in table.columns if column.endswith("_z")]
    if idx_columns or z_columns:
        table = table.drop(columns=idx_columns + z_columns)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, engine=PARQUET_ENGINE, index=False)
    return path


def read_feature_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {path}")
    table = pd.read_parquet(path, engine=PARQUET_ENGINE)
    return _validate_feature_table(table)


def train_feature_table(
    table: pd.DataFrame,
    opts: LatentStratOptions | None = None,
    *,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> FeatureTrainingResult:
    opts = opts or default_options()
    indexed, team_index_map = make_team_index_map(_validate_feature_table(table))
    split = make_split(indexed, opts, policy=opts.split_policy)
    target_stats = fit_target_stats(indexed, split.train_mask, opts)
    prepared = apply_target_stats(indexed, target_stats)
    baselines = fit_baselines(prepared, split, opts)
    model, history, diagnostics = train_model(prepared, split, opts, verbose=verbose)
    report = evaluate_model(model, prepared, split, target_stats, opts, baselines)
    result = FeatureTrainingResult(
        table=indexed,
        prepared=prepared,
        team_index_map=team_index_map,
        history=history,
        diagnostics=diagnostics,
        report=report,
    )
    if output_dir is not None:
        write_feature_training_artifacts(result, output_dir)
    return result


def train_feature_file(
    input_path: str | Path,
    opts: LatentStratOptions | None = None,
    *,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> FeatureTrainingResult:
    return train_feature_table(
        read_feature_table(input_path), opts, output_dir=output_dir, verbose=verbose
    )


def write_feature_training_artifacts(result: FeatureTrainingResult, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.prepared.to_csv(out / "feature_match_table.csv", index=False)
    result.history.to_csv(out / "feature_history.csv", index=False)
    result.report.continuous_metrics.to_csv(out / "feature_continuous_metrics.csv", index=False)
    result.report.binary_metrics.to_csv(out / "feature_binary_metrics.csv", index=False)
    result.report.set_attention.to_csv(out / "feature_set_attention.csv", index=False)
    result.report.zero_out_diagnostics.to_csv(
        out / "feature_zero_out_diagnostics.csv", index=False
    )
    return out
