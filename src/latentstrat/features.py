"""Parquet feature-store helpers for LatentStrat."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlmodel import Session, SQLModel, select

from frc.datastore import FRCDataStore
from frc.importers import TBAImporter
from frc.models import TbaAward
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
    apply_v57_target_stats,
    build_season_match_table,
    fit_target_stats,
    fit_v57_target_stats,
    make_split,
    make_v5_team_index_maps,
    team_number_from_key,
)
from latentstrat.evaluation import EvaluationReport, evaluate_model
from latentstrat.model import SetTransformerModel, init_model
from latentstrat.pretrain_loop import apply_prior_checkpoint_to_model, load_prior_embedding_table
from latentstrat.training import TrainingDiagnostics, train_model

PARQUET_ENGINE = "pyarrow"
NAN = float("nan")

AWARD_ONTOLOGY = {
    "Impact": torch.tensor([1.0, 1.0, NAN, NAN, NAN, NAN, NAN]),
    "EI": torch.tensor([0.0, 1.0, NAN, NAN, NAN, NAN, NAN]),
    "Auto": torch.tensor([NAN, NAN, 1.0, NAN, NAN, NAN, NAN]),
    "Quality": torch.tensor([NAN, NAN, NAN, 1.0, NAN, NAN, NAN]),
    "Design": torch.tensor([NAN, NAN, NAN, NAN, 1.0, NAN, NAN]),
    "Control": torch.tensor([NAN, NAN, NAN, NAN, NAN, 1.0, NAN]),
    "Excellence": torch.tensor([NAN, NAN, NAN, NAN, NAN, NAN, 1.0]),
}

AWARD_CATEGORY_BY_TYPE = {
    0: "Impact",
    9: "EI",
    17: "Design",
    18: "Quality",
    21: "Excellence",
    29: "Control",
    74: "Auto",
}

AWARD_NAME_PATTERNS = (
    ("impact", "Impact"),
    ("chairman", "Impact"),
    ("engineering inspiration", "EI"),
    ("autonomous", "Auto"),
    ("quality", "Quality"),
    ("industrial design", "Design"),
    ("innovation in control", "Control"),
    ("excellence in engineering", "Excellence"),
    ("engineering excellence", "Excellence"),
)


@dataclass
class FeatureTrainingResult:
    table: pd.DataFrame
    prepared: pd.DataFrame
    team_index_map: dict[str, int]
    team_event_index_map: dict[str, int]
    target_stats: object
    v57_target_stats: object | None
    model: object
    history: pd.DataFrame
    diagnostics: TrainingDiagnostics
    report: EvaluationReport
    prior_checkpoint: str | None = None
    prior_vectors_applied: int = 0
    sidecar_tables: dict[str, pd.DataFrame] | None = None


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


def _award_category(award: TbaAward) -> str | None:
    if award.award_type in AWARD_CATEGORY_BY_TYPE:
        return AWARD_CATEGORY_BY_TYPE[award.award_type]
    name = award.name.lower()
    for pattern, category in AWARD_NAME_PATTERNS:
        if pattern in name:
            return category
    return None


def _award_winners_by_event(awards: dict[str, list[TbaAward]]) -> dict[str, dict[str, set[str]]]:
    by_event: dict[str, dict[str, set[str]]] = {}
    for event_key, event_awards in awards.items():
        event_targets = by_event.setdefault(str(event_key), {})
        for award in event_awards:
            category = _award_category(award)
            if category is None:
                continue
            winners = event_targets.setdefault(category, set())
            for recipient in award.recipient_list:
                if recipient.team_key:
                    winners.add(str(recipient.team_key))
    return by_event


def add_award_features(
    table: pd.DataFrame,
    awards: dict[str, list[TbaAward]],
    opts: LatentStratOptions | None = None,
) -> pd.DataFrame:
    """Attach NaN-masked judged-award ontology targets at team-slot grain."""

    opts = opts or default_options()
    if table.empty:
        return table.copy()
    out = table.copy()
    slot_prefixes = [
        f"{color}_team_{slot}" for color in ("red", "blue") for slot in (1, 2, 3)
    ]
    for prefix in slot_prefixes:
        for axis in opts.award_targets:
            out[f"{prefix}_award_{axis}"] = np.nan

    winners_by_event = _award_winners_by_event(awards)
    if not winners_by_event:
        return out

    event_order = (
        out.groupby("event_key", sort=False)["sort_ordinal"].min().sort_values().index.astype(str)
    )
    impact_winners_seen: set[str] = set()
    ontology = {
        name: value.detach().cpu().numpy().astype(float)
        for name, value in AWARD_ONTOLOGY.items()
    }
    for event_key in event_order:
        event_winners = winners_by_event.get(str(event_key), {})
        if not event_winners:
            continue
        event_mask = out["event_key"].astype(str) == str(event_key)
        event_impact_winners = event_winners.get("Impact", set())
        for category, winners in event_winners.items():
            for team_key in winners:
                vector = ontology[category].copy()
                if category == "EI" and team_key in impact_winners_seen:
                    vector[0] = np.nan
                if category == "EI" and team_key in event_impact_winners:
                    vector[0] = np.nan
                for prefix in slot_prefixes:
                    team_mask = event_mask & (out[f"{prefix}_key"].astype(str) == team_key)
                    if not np.any(team_mask):
                        continue
                    for axis_idx, axis in enumerate(opts.award_targets):
                        if np.isfinite(vector[axis_idx]):
                            out.loc[team_mask, f"{prefix}_award_{axis}"] = vector[axis_idx]
        impact_winners_seen.update(event_impact_winners)
    return out


def _flatten_ranking_row(event_key: str, row: dict) -> dict:
    flat = {
        "season": int(str(event_key)[:4]),
        "event_key": event_key,
        "team_key": str(row.get("team_key", "")),
        "qual_rank": row.get("rank"),
        "matches_played": row.get("matches_played"),
    }
    for idx, stat in enumerate(row.get("sort_orders") or [], start=1):
        if isinstance(stat, dict):
            name = stat.get("name") or stat.get("display_name") or f"sort_order_{idx}"
            value = stat.get("value")
        else:
            name = f"sort_order_{idx}"
            value = stat
        clean = str(name).strip().lower().replace(" ", "_")
        flat[f"ranking_stat_{clean}"] = value
    record = row.get("record") or {}
    for key, value in record.items():
        flat[f"record_{key}"] = value
    return flat


def build_rankings_sidecar(event_keys: list[str], provider: TbaProvider) -> pd.DataFrame:
    rows = []
    for event_key in event_keys:
        payload = provider.get_event_rankings(event_key) or {}
        for row in payload.get("rankings") or []:
            rows.append(_flatten_ranking_row(event_key, row))
    return pd.DataFrame(
        rows,
        columns=["season", "event_key", "team_key", "qual_rank", "matches_played"]
        if not rows
        else None,
    )


def _ranking_maps(rankings: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    if rankings.empty:
        return maps
    for event_key, event_rows in rankings.dropna(subset=["qual_rank"]).groupby("event_key"):
        maps[str(event_key)] = {
            str(row["team_key"]): int(row["qual_rank"]) for _, row in event_rows.iterrows()
        }
    return maps


def _passed_over_team(
    rank_map: dict[str, int], selected: set[str], pick_team_key: str
) -> str:
    pick_rank = rank_map.get(pick_team_key)
    if pick_rank is None:
        return ""
    candidates = [
        (rank, team_key)
        for team_key, rank in rank_map.items()
        if team_key not in selected and rank < pick_rank
    ]
    if not candidates:
        return ""
    return sorted(candidates)[0][1]


def build_selection_sidecar(
    event_keys: list[str], provider: TbaProvider, rankings: pd.DataFrame
) -> pd.DataFrame:
    rank_maps = _ranking_maps(rankings)
    rows = []
    for event_key in event_keys:
        selected: set[str] = set()
        rank_map = rank_maps.get(str(event_key), {})
        for alliance_idx, alliance in enumerate(provider.get_event_alliances(event_key), start=1):
            picks = [str(team) for team in alliance.get("picks") or [] if team]
            if not picks:
                continue
            captain = picks[0]
            selected.add(captain)
            for pick_order, pick in enumerate(picks[1:], start=1):
                passed = _passed_over_team(rank_map, selected, pick)
                rows.append(
                    {
                        "season": int(str(event_key)[:4]),
                        "event_key": event_key,
                        "alliance_number": alliance_idx,
                        "captain_team_key": captain,
                        "pick_team_key": pick,
                        "pick_order": pick_order,
                        "passed_over_team_key": passed,
                    }
                )
                selected.add(pick)
    return pd.DataFrame(
        rows,
        columns=[
            "season",
            "event_key",
            "alliance_number",
            "captain_team_key",
            "pick_team_key",
            "pick_order",
            "passed_over_team_key",
        ],
    )


def _playoff_finish_order(status: dict | None) -> int:
    status = status or {}
    if str(status.get("status", "")).lower() == "won":
        return 1
    level = str(status.get("level", "")).lower()
    return {"f": 2, "sf": 3, "qf": 4, "ef": 5}.get(level, 99)


def build_playoff_sidecar(event_keys: list[str], provider: TbaProvider) -> pd.DataFrame:
    rows = []
    for event_key in event_keys:
        for alliance_idx, alliance in enumerate(provider.get_event_alliances(event_key), start=1):
            picks = [str(team) for team in alliance.get("picks") or [] if team]
            padded = picks[:3] + [""] * max(0, 3 - len(picks))
            status = alliance.get("status") or {}
            rows.append(
                {
                    "season": int(str(event_key)[:4]),
                    "event_key": event_key,
                    "alliance_number": alliance_idx,
                    "team_1_key": padded[0],
                    "team_2_key": padded[1],
                    "team_3_key": padded[2],
                    "playoff_finish_order": _playoff_finish_order(status),
                    "status": str(status.get("status", "")),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "season",
            "event_key",
            "alliance_number",
            "team_1_key",
            "team_2_key",
            "team_3_key",
            "playoff_finish_order",
            "status",
        ],
    )


def build_event_sidecars(event_keys: list[str], provider: TbaProvider) -> dict[str, pd.DataFrame]:
    rankings = build_rankings_sidecar(event_keys, provider)
    return {
        "rankings": rankings,
        "selections": build_selection_sidecar(event_keys, provider, rankings),
        "playoffs": build_playoff_sidecar(event_keys, provider),
    }


def write_sidecar_tables(
    sidecars: dict[str, pd.DataFrame], output_dir: str | Path, season: int
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in sidecars.items():
        path = out / f"{name}_{season}.parquet"
        table.to_parquet(path, engine=PARQUET_ENGINE, index=False)
        paths[name] = path
    return paths


def _base_idx_for_key(team_key: str, team_index_map: dict[str, int]) -> int:
    if not team_key or str(team_key).lower() == "nan":
        return 0
    return int(team_index_map.get(team_key, team_number_from_key(team_key)))


def index_sidecar_tables(
    sidecars: dict[str, pd.DataFrame] | None,
    team_index_map: dict[str, int],
    team_event_index_map: dict[str, int],
) -> dict[str, pd.DataFrame] | None:
    if not sidecars:
        return None
    indexed: dict[str, pd.DataFrame] = {}
    for name, table in sidecars.items():
        out = table.copy()
        if out.empty:
            indexed[name] = out
            continue
        if name == "rankings" and "team_key" in out.columns:
            out["team_base_idx"] = [
                _base_idx_for_key(str(key), team_index_map) for key in out["team_key"]
            ]
            out["team_event_idx"] = [
                team_event_index_map.get(f"{event_key}::{team_key}", 0)
                for event_key, team_key in zip(out["event_key"], out["team_key"], strict=True)
            ]
        elif name == "selections":
            for prefix, column in (
                ("captain", "captain_team_key"),
                ("pick", "pick_team_key"),
                ("passed_over", "passed_over_team_key"),
            ):
                out[f"{prefix}_base_idx"] = [
                    _base_idx_for_key(str(key), team_index_map) if str(key) else 0
                    for key in out[column]
                ]
                out[f"{prefix}_event_idx"] = [
                    team_event_index_map.get(f"{event_key}::{team_key}", 0)
                    for event_key, team_key in zip(out["event_key"], out[column], strict=True)
                ]
        elif name == "playoffs":
            for slot in (1, 2, 3):
                column = f"team_{slot}_key"
                out[f"team_{slot}_base_idx"] = [
                    _base_idx_for_key(str(key), team_index_map) if str(key) else 0
                    for key in out[column]
                ]
                out[f"team_{slot}_event_idx"] = [
                    team_event_index_map.get(f"{event_key}::{team_key}", 0)
                    for event_key, team_key in zip(out["event_key"], out[column], strict=True)
                ]
        indexed[name] = out
    return indexed


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
    table = add_award_features(table, store.awards, opts)
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
    table = add_award_features(table, store.awards, opts)
    if scouting_db_path is not None:
        table = merge_scouting_features(table, scouting_db_path)
    return table


def build_feature_sidecars(
    event_keys: list[str], provider: TbaProvider, season: int, output_dir: str | Path | None = None
) -> dict[str, pd.DataFrame]:
    sidecars = build_event_sidecars(event_keys, provider)
    if output_dir is not None:
        write_sidecar_tables(sidecars, output_dir, season)
    return sidecars


def write_feature_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    table = _validate_feature_table(table)
    idx_columns = [
        column
        for column in table.columns
        if column.endswith("_idx") or column.endswith("_base_idx") or column.endswith("_event_idx")
    ]
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


def load_v5_checkpoint_model(checkpoint_path: str | Path) -> SetTransformerModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    opts = LatentStratOptions.model_validate(
        checkpoint.get("options", default_options().model_dump())
    )
    state = checkpoint["model_state_dict"]
    base_weight = state["Z_base.weight"]
    event_weight = state["Z_event.weight"]
    model = init_model(
        base_weight.shape[0],
        base_weight.shape[1],
        len(opts.target_map),
        len(opts.binary_targets),
        opts,
        num_event_teams=event_weight.shape[0],
        num_endgame_classes=len(opts.endgame_class_order),
        num_awards=len(opts.award_targets),
    )
    model.load_state_dict(state, strict=False)
    return model


def train_feature_table(
    table: pd.DataFrame,
    opts: LatentStratOptions | None = None,
    *,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    venue_mode: bool = False,
    venue_event_key: str | None = None,
    initial_model: SetTransformerModel | None = None,
    prior_checkpoint: str | Path | None = None,
    sidecar_tables: dict[str, pd.DataFrame] | None = None,
) -> FeatureTrainingResult:
    opts = opts or default_options()
    indexed, team_index_map, team_event_index_map = make_v5_team_index_maps(
        _validate_feature_table(table)
    )
    if venue_mode and venue_event_key is not None:
        indexed = indexed[indexed["event_key"].astype(str) == str(venue_event_key)].reset_index(
            drop=True
        )
    split = make_split(indexed, opts, policy=opts.split_policy)
    target_stats = fit_target_stats(indexed, split.train_mask, opts)
    v57_target_stats = fit_v57_target_stats(indexed, split.train_mask, opts)
    prepared = apply_v57_target_stats(apply_target_stats(indexed, target_stats), v57_target_stats)
    baselines = fit_baselines(prepared, split, opts)
    indexed_sidecars = index_sidecar_tables(sidecar_tables, team_index_map, team_event_index_map)
    prior_vectors_applied = 0
    training_model = initial_model
    if prior_checkpoint is not None:
        prior_table = load_prior_embedding_table(prior_checkpoint)
        if training_model is None:
            training_model = init_model(
                int(prior_table.shape[0]),
                opts.latent_dim,
                len(opts.target_map),
                len(opts.binary_targets),
                opts,
                num_event_teams=len(team_event_index_map) + 1,
                num_endgame_classes=len(opts.endgame_class_order),
                num_awards=len(opts.award_targets),
            )
        prior_vectors_applied = apply_prior_checkpoint_to_model(
            training_model,
            prior_checkpoint,
            latent_dim=opts.latent_dim,
        )
    model, history, diagnostics = train_model(
        prepared,
        split,
        opts,
        verbose=verbose,
        venue_mode=venue_mode,
        initial_model=training_model,
        sidecar_tables=indexed_sidecars,
    )
    report = evaluate_model(model, prepared, split, target_stats, opts, baselines)
    result = FeatureTrainingResult(
        table=indexed,
        prepared=prepared,
        team_index_map=team_index_map,
        team_event_index_map=team_event_index_map,
        target_stats=target_stats,
        v57_target_stats=v57_target_stats,
        model=model,
        history=history,
        diagnostics=diagnostics,
        report=report,
        prior_checkpoint=str(prior_checkpoint) if prior_checkpoint is not None else None,
        prior_vectors_applied=prior_vectors_applied,
        sidecar_tables=indexed_sidecars,
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
    venue_mode: bool = False,
    venue_event_key: str | None = None,
    initial_model: SetTransformerModel | None = None,
    prior_checkpoint: str | Path | None = None,
    sidecar_tables: dict[str, pd.DataFrame] | None = None,
) -> FeatureTrainingResult:
    return train_feature_table(
        read_feature_table(input_path),
        opts,
        output_dir=output_dir,
        verbose=verbose,
        venue_mode=venue_mode,
        venue_event_key=venue_event_key,
        initial_model=initial_model,
        prior_checkpoint=prior_checkpoint,
        sidecar_tables=sidecar_tables,
    )


def write_feature_training_artifacts(result: FeatureTrainingResult, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.prepared.to_csv(out / "feature_match_table.csv", index=False)
    result.history.to_csv(out / "feature_history.csv", index=False)
    result.report.continuous_metrics.to_csv(out / "feature_continuous_metrics.csv", index=False)
    result.report.binary_metrics.to_csv(out / "feature_binary_metrics.csv", index=False)
    if hasattr(result.report, "endgame_metrics"):
        result.report.endgame_metrics.to_csv(out / "feature_endgame_metrics.csv", index=False)
    if hasattr(result.report, "award_metrics"):
        result.report.award_metrics.to_csv(out / "feature_award_metrics.csv", index=False)
    if result.sidecar_tables:
        for name, table in result.sidecar_tables.items():
            table.to_csv(out / f"feature_{name}_sidecar.csv", index=False)
    result.report.set_attention.to_csv(out / "feature_set_attention.csv", index=False)
    result.report.zero_out_diagnostics.to_csv(
        out / "feature_zero_out_diagnostics.csv", index=False
    )
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "options": result.model.opts.model_dump(mode="json"),
            "target_stats": {
                "target_names": result.target_stats.target_names,
                "mu": result.target_stats.mu.tolist(),
                "sigma": result.target_stats.sigma.tolist(),
                "is_constant": result.target_stats.is_constant.tolist(),
            },
            "v57_target_stats": {
                "target_names": result.v57_target_stats.target_names,
                "mu": result.v57_target_stats.mu.tolist(),
                "sigma": result.v57_target_stats.sigma.tolist(),
                "is_constant": result.v57_target_stats.is_constant.tolist(),
            }
            if result.v57_target_stats is not None
            else None,
            "team_base_index_map": result.team_index_map,
            "team_event_index_map": result.team_event_index_map,
            "prior_checkpoint": result.prior_checkpoint,
            "prior_vectors_applied": result.prior_vectors_applied,
        },
        out / "v5_checkpoint.pt",
    )
    return out
