"""SQLite persistence for durable LatentStrat V5 base embeddings."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from sqlmodel import Field, Session, SQLModel, create_engine, select

from latentstrat.features import load_season_checkpoint_model


class TeamBaseEmbedding(SQLModel, table=True):
    __tablename__ = "team_base_embeddings"

    team_key: str = Field(primary_key=True)
    season: int = Field(primary_key=True)
    vector_json: str
    source_event_key: str | None = None
    source_checkpoint: str | None = None
    updated_at: str


def create_embedding_engine(path: str | Path):
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def create_embedding_store(path: str | Path) -> Path:
    db_path = Path(path)
    engine = create_embedding_engine(db_path)
    SQLModel.metadata.create_all(engine)
    return db_path


def upsert_team_embedding(
    path: str | Path,
    *,
    team_key: str,
    season: int,
    vector: list[float],
    source_event_key: str | None,
    source_checkpoint: str | None,
) -> None:
    create_embedding_store(path)
    engine = create_embedding_engine(path)
    with Session(engine) as session:
        existing = session.exec(
            select(TeamBaseEmbedding).where(
                TeamBaseEmbedding.team_key == team_key,
                TeamBaseEmbedding.season == season,
            )
        ).one_or_none()
        row = existing or TeamBaseEmbedding(
            team_key=team_key,
            season=season,
            vector_json="[]",
            updated_at="",
        )
        row.vector_json = json.dumps(vector)
        row.source_event_key = source_event_key
        row.source_checkpoint = source_checkpoint
        row.updated_at = datetime.now(UTC).isoformat()
        session.merge(row)
        session.commit()


def consolidate_event_checkpoint(
    checkpoint_path: str | Path,
    *,
    event_key: str,
    embedding_db: str | Path,
    delta_weeks: float = 1.0,
) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = load_season_checkpoint_model(checkpoint_path)
    opts = model.opts
    state = checkpoint["model_state_dict"]
    base_map: dict[str, int] = checkpoint.get("team_base_index_map", {})
    event_map: dict[str, int] = checkpoint.get("team_event_index_map", {})
    reverse_base = {idx: key for key, idx in base_map.items()}
    updated_state = {name: value.clone() for name, value in state.items()}
    delta = torch.tensor([delta_weeks], dtype=torch.float32)
    for pair_key, event_idx in event_map.items():
        if not pair_key.startswith(f"{event_key}::"):
            continue
        team_key = pair_key.split("::", 1)[1]
        base_idx = base_map.get(team_key)
        if base_idx is None:
            continue
        z_base = model.Z_base.weight[base_idx : base_idx + 1]
        z_event = model.Z_event.weight[event_idx : event_idx + 1]
        new_base = model.delta_integration_gate.integrate(z_base, z_event, delta).detach()[0]
        upsert_team_embedding(
            embedding_db,
            team_key=reverse_base.get(base_idx, team_key),
            season=int(opts.season),
            vector=new_base.tolist(),
            source_event_key=event_key,
            source_checkpoint=str(checkpoint_path),
        )
        updated_state["Z_base.weight"][base_idx] = new_base
        updated_state["Z_event.weight"][event_idx].zero_()

    updated = dict(checkpoint)
    updated["model_state_dict"] = updated_state
    version = "v6" if checkpoint.get("checkpoint_schema_version") == 6 else "v5"
    output_path = Path(checkpoint_path).with_name(f"{version}_checkpoint_consolidated.pt")
    torch.save(updated, output_path)
    return output_path
