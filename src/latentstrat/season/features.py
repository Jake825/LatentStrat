"""Parquet feature-store helpers for LatentStrat."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
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
from latentstrat.artifacts import write_artifact_manifest
from latentstrat.artifacts.checkpoints import (
    load_season_checkpoint_model as load_season_checkpoint_model,
)
from latentstrat.baselines import fit_baselines
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.experimental.frozen_targets import (
    LoadedWorldModelBundle,
    WorldModelOptions,
    attach_world_model_targets,
    load_world_model_bundle,
)
from latentstrat.pretraining.prior.train import (
    apply_prior_checkpoint_to_model,
    load_prior_embedding_table,
)
from latentstrat.season.data import (
    Split,
    apply_target_stats,
    apply_v57_target_stats,
    build_season_match_table,
    fit_target_stats,
    fit_v57_target_stats,
    make_split,
    make_v5_team_index_maps,
    team_number_from_key,
)
from latentstrat.season.evaluate import EvaluationReport, evaluate_model
from latentstrat.season.model import SetTransformerModel, init_model
from latentstrat.season.plots import write_feature_artifact_plots
from latentstrat.season.train import TrainingDiagnostics, train_model
from latentstrat.visualizations import write_feature_phase_visuals

PARQUET_ENGINE = "pyarrow"
NAN = float("nan")
AWARD_CATALOG_PATH = Path("data/catalogs/awards.yaml")


@lru_cache(maxsize=1)
def load_award_catalog(path: str | Path = AWARD_CATALOG_PATH) -> tuple[dict, ...]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    awards = payload.get("awards", [])
    if not isinstance(awards, list):
        raise ValueError("Award catalog must define an awards list.")
    return tuple(dict(award) for award in awards)


def _supervised_award_catalog() -> tuple[dict, ...]:
    return tuple(
        award for award in load_award_catalog() if award.get("supervised_category") is not None
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
    tensorboard_logdir: str | None = None
    split_policy: str | None = None
    split_fallback_reason: str | None = None
    frozen_embedding_tables: tuple[str, ...] = ()
    world_model_bundle: str | None = None
    world_model_options: WorldModelOptions | None = None


def _validate_feature_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Feature table is empty.")
    out = table.reset_index(drop=True).copy()
    required = {"red_total_score", "blue_total_score", "red_win"}
    if required.issubset(out.columns):
        red = pd.to_numeric(out["red_total_score"], errors="coerce")
        blue = pd.to_numeric(out["blue_total_score"], errors="coerce")
        out["red_win"] = pd.to_numeric(out["red_win"], errors="coerce").astype(float)
        out.loc[red.eq(blue) & red.notna(), "red_win"] = np.nan
    return out


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
    catalog = _supervised_award_catalog()
    for entry in catalog:
        if award.award_type in set(entry.get("tba_types", [])):
            return str(entry["supervised_category"])
    name = award.name.lower()
    for entry in catalog:
        if any(str(pattern).lower() in name for pattern in entry.get("name_patterns", [])):
            return str(entry["supervised_category"])
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
    slot_prefixes = [f"{color}_team_{slot}" for color in ("red", "blue") for slot in (1, 2, 3)]
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
    ontology = {}
    for entry in _supervised_award_catalog():
        values = entry.get("supervised_targets", {})
        ontology[str(entry["supervised_category"])] = np.asarray(
            [float(values.get(axis, NAN)) for axis in opts.award_targets],
            dtype=float,
        )
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


def _passed_over_team(rank_map: dict[str, int], selected: set[str], pick_team_key: str) -> str:
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


def event_week_table_from_feature_table(table: pd.DataFrame) -> pd.DataFrame:
    if "event_key" not in table.columns or "event_week" not in table.columns:
        return pd.DataFrame(columns=["event_key", "raw_event_week", "event_week"])
    columns = ["event_key", "event_week"]
    if "raw_event_week" in table.columns:
        columns.insert(1, "raw_event_week")
    event_weeks = table[columns].drop_duplicates(subset=["event_key"]).copy()
    if "raw_event_week" not in event_weeks.columns:
        event_weeks["raw_event_week"] = np.nan
    return event_weeks[["event_key", "raw_event_week", "event_week"]]


def enrich_sidecars_with_event_weeks(
    sidecars: dict[str, pd.DataFrame], event_weeks: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    if event_weeks is None or event_weeks.empty:
        return sidecars
    week_table = event_weeks[["event_key", "raw_event_week", "event_week"]].drop_duplicates(
        subset=["event_key"]
    )
    enriched = {}
    for name, table in sidecars.items():
        if table.empty or "event_key" not in table.columns:
            enriched[name] = table.copy()
            continue
        base = table.drop(columns=["raw_event_week", "event_week"], errors="ignore")
        enriched[name] = base.merge(week_table, on="event_key", how="left")
    return enriched


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
        elif name == "world_rank" and "team_key" in out.columns:
            out["team_base_idx"] = [
                _base_idx_for_key(str(key), team_index_map) for key in out["team_key"]
            ]
            out["team_event_idx"] = [
                team_event_index_map.get(f"{event_key}::{team_key}", 0)
                for event_key, team_key in zip(out["event_key"], out["team_key"], strict=True)
            ]
        elif name == "world_pick":
            for prefix, column in (
                ("captain", "captain_team_key"),
                ("candidate", "candidate_team_key"),
            ):
                out[f"{prefix}_base_idx"] = [
                    _base_idx_for_key(str(key), team_index_map) if str(key) else 0
                    for key in out[column]
                ]
                out[f"{prefix}_event_idx"] = [
                    team_event_index_map.get(f"{event_key}::{team_key}", 0)
                    for event_key, team_key in zip(out["event_key"], out[column], strict=True)
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
    event_keys: list[str],
    provider: TbaProvider,
    season: int,
    output_dir: str | Path | None = None,
    event_weeks: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    sidecars = enrich_sidecars_with_event_weeks(
        build_event_sidecars(event_keys, provider), event_weeks
    )
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
    freeze_team_embeddings: bool = False,
    sidecar_tables: dict[str, pd.DataFrame] | None = None,
    split_override: Split | None = None,
    tensorboard_writer: object | None = None,
    tensorboard_logdir: str | Path | None = None,
    world_model_bundle: str | Path | LoadedWorldModelBundle | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_output: str | Path | None = None,
) -> FeatureTrainingResult:
    opts = opts or default_options()
    loaded_world_model = (
        load_world_model_bundle(world_model_bundle)
        if isinstance(world_model_bundle, (str, Path))
        else world_model_bundle
    )
    world_model_opts = (
        loaded_world_model.options
        if loaded_world_model is not None
        else WorldModelOptions(enabled=False)
    )
    if (
        loaded_world_model is None
        and initial_model is not None
        and initial_model.world_model_opts.active_spaces()
    ):
        raise ValueError("Active frozen-target checkpoint training requires frozen_target_bundle.")
    if resume_checkpoint is not None and (
        initial_model is not None or prior_checkpoint is not None
    ):
        raise ValueError(
            "resume_checkpoint cannot be combined with checkpoint or prior warm-starting."
        )
    if initial_model is not None:
        initial_widths = tuple(
            initial_model.world_model_opts.space(name).width
            for name in ("score", "award", "rank", "pick")
        )
        requested_widths = tuple(
            world_model_opts.space(name).width for name in ("score", "award", "rank", "pick")
        )
        if initial_widths != requested_widths:
            raise ValueError("World-model bundle widths do not match the checkpoint architecture.")
        initial_model.world_model_opts = world_model_opts
    attached_sidecars = sidecar_tables
    if loaded_world_model is not None:
        table, attached_sidecars = attach_world_model_targets(
            table, sidecar_tables, loaded_world_model
        )
    indexed, team_index_map, team_event_index_map = make_v5_team_index_maps(
        _validate_feature_table(table),
        compact_base=opts.state_model == "static-z-base",
    )
    if venue_mode and venue_event_key is not None:
        indexed = indexed[indexed["event_key"].astype(str) == str(venue_event_key)].reset_index(
            drop=True
        )
    split = split_override or make_split(indexed, opts, policy=opts.split_policy)
    target_stats = fit_target_stats(indexed, split.train_mask, opts)
    v57_target_stats = fit_v57_target_stats(indexed, split.train_mask, opts)
    prepared = apply_v57_target_stats(apply_target_stats(indexed, target_stats), v57_target_stats)
    baselines = fit_baselines(prepared, split, opts)
    indexed_sidecars = index_sidecar_tables(attached_sidecars, team_index_map, team_event_index_map)
    prior_vectors_applied = 0
    training_model = initial_model
    if prior_checkpoint is not None:
        prior_table = load_prior_embedding_table(prior_checkpoint)
        if training_model is None:
            training_model = init_model(
                (
                    max(team_index_map.values(), default=0) + 1
                    if opts.state_model == "static-z-base"
                    else int(prior_table.shape[0])
                ),
                opts.latent_dim,
                len(opts.target_map),
                len(opts.binary_targets),
                opts,
                num_event_teams=len(team_event_index_map) + 1,
                num_endgame_classes=len(opts.endgame_class_order),
                num_awards=len(opts.award_targets),
                world_model_opts=world_model_opts,
            )
        prior_vectors_applied = apply_prior_checkpoint_to_model(
            training_model,
            prior_checkpoint,
            latent_dim=opts.latent_dim,
            team_index_map=(team_index_map if opts.state_model == "static-z-base" else None),
        )
    frozen_embedding_tables = ("Z_base", "Z_event") if freeze_team_embeddings else ()
    model, history, diagnostics = train_model(
        prepared,
        split,
        opts,
        verbose=verbose,
        venue_mode=venue_mode,
        freeze_team_embeddings=freeze_team_embeddings,
        initial_model=training_model,
        sidecar_tables=indexed_sidecars,
        tensorboard_writer=tensorboard_writer,
        world_model_opts=world_model_opts,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None,
        resume_output=(
            str(resume_output)
            if resume_output is not None
            else (
                str(Path(output_dir) / "resume" / "season" / "latest.ckpt")
                if output_dir is not None
                else None
            )
        ),
    )
    report = evaluate_model(model, prepared, split, target_stats, opts, baselines, v57_target_stats)
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
        tensorboard_logdir=str(tensorboard_logdir) if tensorboard_logdir is not None else None,
        split_policy=split.policy,
        split_fallback_reason=split.fallback_reason,
        frozen_embedding_tables=frozen_embedding_tables,
        world_model_bundle=(
            str(loaded_world_model.root) if loaded_world_model is not None else None
        ),
        world_model_options=world_model_opts,
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
    freeze_team_embeddings: bool = False,
    sidecar_tables: dict[str, pd.DataFrame] | None = None,
    split_override: Split | None = None,
    tensorboard_writer: object | None = None,
    tensorboard_logdir: str | Path | None = None,
    world_model_bundle: str | Path | LoadedWorldModelBundle | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_output: str | Path | None = None,
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
        freeze_team_embeddings=freeze_team_embeddings,
        sidecar_tables=sidecar_tables,
        split_override=split_override,
        tensorboard_writer=tensorboard_writer,
        tensorboard_logdir=tensorboard_logdir,
        world_model_bundle=world_model_bundle,
        resume_checkpoint=resume_checkpoint,
        resume_output=resume_output,
    )


def _feature_slice_table(slices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, table in slices.items():
        frame = table.copy()
        frame.insert(0, "slice_group", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_feature_training_artifacts(result: FeatureTrainingResult, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.prepared.to_csv(out / "feature_match_table.csv", index=False)
    result.history.to_csv(out / "feature_history.csv", index=False)
    (out / "feature_training_diagnostics.json").write_text(
        json.dumps(_json_ready(asdict(result.diagnostics)), indent=2),
        encoding="utf-8",
    )
    result.report.continuous_metrics.to_csv(out / "feature_continuous_metrics.csv", index=False)
    result.report.binary_metrics.to_csv(out / "feature_binary_metrics.csv", index=False)
    result.report.common_metrics.to_csv(out / "feature_common_metrics.csv", index=False)
    result.report.calibration.to_csv(out / "feature_calibration.csv", index=False)
    _feature_slice_table(result.report.slices).to_csv(
        out / "feature_availability_slices.csv", index=False
    )
    if hasattr(result.report, "endgame_metrics"):
        result.report.endgame_metrics.to_csv(out / "feature_endgame_metrics.csv", index=False)
    if hasattr(result.report, "award_metrics"):
        result.report.award_metrics.to_csv(out / "feature_award_metrics.csv", index=False)
    if result.sidecar_tables:
        for name, table in result.sidecar_tables.items():
            table.to_csv(out / f"feature_{name}_sidecar.csv", index=False)
    result.report.set_attention.to_csv(out / "feature_set_attention.csv", index=False)
    result.report.zero_out_diagnostics.to_csv(out / "feature_zero_out_diagnostics.csv", index=False)
    write_feature_artifact_plots(result, out)
    write_feature_phase_visuals(
        model=result.model,
        prepared=result.prepared,
        history=result.history,
        report=result.report,
        team_index_map=result.team_index_map,
        team_event_index_map=result.team_event_index_map,
        prior_checkpoint=result.prior_checkpoint,
        sidecar_tables=result.sidecar_tables,
        output_dir=out,
    )
    checkpoint = {
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
        "split_policy": result.split_policy,
        "split_fallback_reason": result.split_fallback_reason,
        "frozen_embedding_tables": list(result.frozen_embedding_tables),
        "sidecar_tables": sorted(result.sidecar_tables) if result.sidecar_tables else [],
    }
    world_model_options = result.world_model_options or WorldModelOptions(enabled=False)
    checkpoint["checkpoint_schema_version"] = 6
    checkpoint["world_model"] = world_model_options.model_dump(mode="json")
    checkpoint["world_model_bundle"] = result.world_model_bundle
    checkpoint_path = out / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    write_artifact_manifest(
        out,
        workflow="season.train",
        artifact_version="v6.1",
        resolved_config=result.model.opts.model_dump(mode="json"),
        promotion_note="Season checkpoint; compare walk-forward evidence before promotion.",
    )
    return out


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value
