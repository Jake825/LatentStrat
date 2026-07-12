"""Streamlit presentation layer for the LatentStrat research workbench."""

# ruff: noqa: E501

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from latentstrat.app.analysis import (
    checkpoint_feature_compatibility,
    compute_pca,
    cosine_neighbors,
    diagnose_match,
    prior_embedding_frame,
    season_embedding_frame,
    season_event_trajectory_frame,
)
from latentstrat.app.catalog import (
    discover_workspace,
    verify_artifact_provenance,
    workspace_root,
)
from latentstrat.app.contracts import ArtifactKind, ArtifactRef, EmbeddingSpaceRef
from latentstrat.app.loaders import (
    checkpoint_summary,
    load_checkpoint_payload,
    load_csv,
    load_json,
    load_parquet,
    load_scouting_table,
    module_hierarchy,
    parameter_table,
    profile_dataframe,
    profile_parquet,
    scouting_feature_trace,
    scouting_overview,
    tensor_values,
)

BRAND_CSS = """
<style>
@font-face {
  font-family: "Roboto";
  src: url("/app/static/fonts/Roboto-Variable.ttf") format("truetype");
  font-style: normal;
  font-weight: 100 900;
  font-display: swap;
}
@font-face {
  font-family: "Roboto Condensed";
  src: url("/app/static/fonts/RobotoCondensed-Variable.ttf") format("truetype");
  font-style: normal;
  font-weight: 100 900;
  font-display: swap;
}
@font-face {
  font-family: "Material Symbols Rounded";
  src: url("/app/static/fonts/MaterialSymbolsRounded-Subset.ttf?v=3") format("truetype");
  font-style: normal;
  font-weight: 400;
  font-display: block;
}
:root {
  --ls-red: #ED1C24;
  --ls-red-deep: #8F1118;
  --ls-blue: #0066B3;
  --ls-blue-deep: #004D80;
  --ls-cyan: #009CD7;
  --ls-ink: #231F20;
  --ls-gray: #9A989A;
  --ls-yellow: #FFD400;
  --ls-paper: #F3F4F6;
}
.block-container { padding-top: 3.5rem; padding-bottom: 2.5rem; max-width: 1480px; }
html, body { font-family: "Roboto", Arial, sans-serif; }
[data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded" !important;
  font-feature-settings: "liga";
  font-style: normal;
  font-weight: 400;
  letter-spacing: normal;
  line-height: 1;
  text-transform: none;
  white-space: nowrap;
}
[data-testid="stMetric"] { border-radius: 2px; border-left: 5px solid var(--ls-blue); }
[data-testid="stMetricValue"] { font-family: "Roboto Condensed", Arial, sans-serif; font-weight: 800; }
.ls-context {
  align-items: center; background: #101820; color: white; display: grid;
  grid-template-columns: minmax(150px, 1fr) minmax(180px, 2fr) auto;
  margin: 0 0 1rem; min-height: 48px; padding: 0 14px; border-bottom: 5px solid var(--ls-blue);
}
.ls-context__page { font-family: "Roboto Condensed", Arial, sans-serif; font-size: 1.1rem; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
.ls-context__middle { text-align: center; font-weight: 700; }
.ls-context__status { background: var(--ls-yellow); color: #111; font-size: .78rem; font-weight: 900; padding: 6px 10px; text-transform: uppercase; }
.ls-ribbon { border-left: 8px solid var(--ls-gray); background: color-mix(in srgb, var(--ls-gray) 15%, transparent); font-weight: 750; margin: .5rem 0 1rem; padding: .65rem .8rem; text-transform: uppercase; }
.ls-ribbon--yellow { border-color: var(--ls-yellow); }
.ls-ribbon--blue { border-color: var(--ls-blue); }
.ls-ribbon--green { border-color: #138A54; }
.ls-ribbon--red { border-color: var(--ls-red); }
.ls-scoreboard { display: grid; grid-template-columns: 1fr minmax(110px, .44fr) 1fr; margin: .75rem 0 1.25rem; min-height: 116px; }
.ls-score { color: white; padding: 12px 18px; }
.ls-score--red { background: linear-gradient(90deg, var(--ls-red-deep), var(--ls-red)); text-align: right; }
.ls-score--blue { background: linear-gradient(90deg, var(--ls-blue), var(--ls-blue-deep)); }
.ls-score__label { font-weight: 800; text-transform: uppercase; }
.ls-score__value { font-family: "Roboto Condensed", Arial, sans-serif; font-size: 3.25rem; font-weight: 900; line-height: 1; }
.ls-score__teams { font-family: "Roboto Condensed", Arial, sans-serif; font-weight: 700; margin-top: 8px; }
.ls-score__center { align-items: center; background: #F4F4F4; color: #111; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.ls-score__center strong { font-family: "Roboto Condensed", Arial, sans-serif; font-size: 1.55rem; }
.ls-team-board { display: grid; gap: 0; grid-template-columns: 1fr 70px 1fr; margin: .75rem 0 1.25rem; }
.ls-alliance { padding: 14px; color: white; }
.ls-alliance--red { background: var(--ls-red-deep); }
.ls-alliance--blue { background: var(--ls-blue-deep); }
.ls-alliance h3 { color: white; font-family: "Roboto Condensed", Arial, sans-serif; margin: 0 0 10px; text-transform: uppercase; }
.ls-team-row { background: rgba(255,255,255,.96); color: #161616; font-family: "Roboto Condensed", Arial, sans-serif; font-size: 1.06rem; font-weight: 800; margin: 6px 0; padding: 8px 10px; }
.ls-versus { align-items: center; background: #F4F4F4; color: #111; display: flex; font-family: "Roboto Condensed", Arial, sans-serif; font-size: 1.8rem; font-weight: 900; justify-content: center; }
.ls-footer-note { border-top: 1px solid rgba(128,128,128,.4); font-size: .72rem; margin-top: 1rem; padding-top: .75rem; }
@media (max-width: 900px) {
  .ls-context { grid-template-columns: 1fr auto; }
  .ls-context__middle { display: none; }
  .ls-scoreboard { grid-template-columns: 1fr 90px 1fr; }
  .ls-score__value { font-size: 2.45rem; }
}
</style>
"""


def configure_page() -> None:
    st.set_page_config(
        page_title="LatentStrat Workbench",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(BRAND_CSS, unsafe_allow_html=True)


def _root() -> Path:
    return workspace_root(os.environ.get("LATENTSTRAT_APP_ROOT"))


@st.cache_data(ttl=10, show_spinner=False)
def _catalog(root: str):
    return discover_workspace(root)


@st.cache_data(show_spinner=False)
def _parquet_profile(path: str, size: int, mtime_ns: int):
    _ = size, mtime_ns
    return profile_parquet(path)


@st.cache_data(show_spinner="Reading Parquet artifact…")
def _parquet(path: str, size: int, mtime_ns: int):
    _ = size, mtime_ns
    return load_parquet(path)


@st.cache_resource(show_spinner="Loading checkpoint on CPU…")
def _checkpoint(path: str, size: int, mtime_ns: int, root: str):
    _ = size, mtime_ns
    return load_checkpoint_payload(path, root)


def _get_catalog():
    return _catalog(str(_root()))


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## LatentStrat")
        st.caption("RESEARCH WORKBENCH")
        if st.button("Refresh workspace", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption(str(_root()))
        st.markdown(
            '<div class="ls-footer-note"><strong>Unofficial community analytics project.</strong><br>'
            "Not affiliated with or endorsed by FIRST®.</div>",
            unsafe_allow_html=True,
        )


def context_bar(page: str, middle: str = "Read-only workspace", status: str = "Diagnostic") -> None:
    st.markdown(
        '<div class="ls-context">'
        f'<div class="ls-context__page">{html.escape(page)}</div>'
        f'<div class="ls-context__middle">{html.escape(middle)}</div>'
        f'<div class="ls-context__status">{html.escape(status)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def status_ribbon(text: str, tone: str = "yellow") -> None:
    st.markdown(
        f'<div class="ls-ribbon ls-ribbon--{html.escape(tone)}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _artifact_select(
    label: str,
    kinds: tuple[ArtifactKind, ...],
    key: str,
    *,
    preferred_path_token: str | None = None,
) -> ArtifactRef | None:
    artifacts = _get_catalog().of_kind(*kinds)
    if not artifacts:
        st.info(f"No {label.lower()} artifacts were found in the configured workspace.")
        return None
    if preferred_path_token:
        artifacts = tuple(
            sorted(
                artifacts,
                key=lambda artifact: (
                    0 if preferred_path_token in artifact.path.as_posix() else 1,
                    artifact.path.as_posix(),
                ),
            )
        )
    requested = str(st.query_params.get("artifact", ""))
    requested_index = next(
        (
            index
            for index, artifact in enumerate(artifacts)
            if requested and requested in artifact.path.as_posix()
        ),
        0,
    )
    selected_id = st.selectbox(
        label,
        [artifact.artifact_id for artifact in artifacts],
        format_func=lambda artifact_id: next(
            artifact.path.relative_to(_get_catalog().root).as_posix()
            for artifact in artifacts
            if artifact.artifact_id == artifact_id
        ),
        key=f"{key}.{requested or 'default'}",
        index=requested_index,
    )
    return _get_catalog().by_id(selected_id)


def _signature_args(artifact: ArtifactRef) -> tuple[str, int, int]:
    return str(artifact.path), artifact.signature.size, artifact.signature.mtime_ns


def render_workspace() -> None:
    catalog = _get_catalog()
    context_bar("Workspace", f"{len(catalog.artifacts)} discovered artifacts", "Read only")
    counts = {kind: len(catalog.of_kind(kind)) for kind in ArtifactKind}
    columns = st.columns(4)
    columns[0].metric("Feature tables", counts[ArtifactKind.FEATURE_TABLE])
    columns[1].metric(
        "Model checkpoints",
        counts[ArtifactKind.SEASON_CHECKPOINT]
        + counts[ArtifactKind.PRIOR_CHECKPOINT]
        + counts[ArtifactKind.MATCH_BREAKDOWN_CHECKPOINT],
    )
    columns[2].metric("Prediction tables", counts[ArtifactKind.PREDICTIONS])
    columns[3].metric("Evaluation files", counts[ArtifactKind.EVALUATION])
    if any(artifact.promotion_eligible is False for artifact in catalog.artifacts):
        status_ribbon("Development evidence is present; promotion limitations remain in force.")
    table = pd.DataFrame.from_records(
        [
            {
                "kind": artifact.kind.value,
                "path": artifact.path.relative_to(catalog.root).as_posix(),
                "status": artifact.status.value,
                "provenance": artifact.provenance_status.value,
                "season": artifact.season,
                "workflow": artifact.workflow,
                "promotion eligible": artifact.promotion_eligible,
                "size MB": artifact.signature.size / 1_048_576,
            }
            for artifact in catalog.artifacts
        ],
        columns=[
            "kind",
            "path",
            "status",
            "provenance",
            "season",
            "workflow",
            "promotion eligible",
            "size MB",
        ],
    )
    kind_filter = st.multiselect("Artifact types", sorted(table["kind"].unique()))
    if kind_filter:
        table = table[table["kind"].isin(kind_filter)]
    st.dataframe(table, hide_index=True, width="stretch", height=520)
    if catalog.artifacts:
        verify_id = st.selectbox(
            "Artifact provenance check",
            [artifact.artifact_id for artifact in catalog.artifacts],
            format_func=lambda value: (
                catalog.by_id(value).path.relative_to(catalog.root).as_posix()
            ),
        )
        if st.button("Verify declared SHA-256", icon=":material/verified:"):
            selected = catalog.by_id(verify_id)
            try:
                provenance = verify_artifact_provenance(selected)
            except (OSError, RuntimeError) as exc:
                st.error(str(exc))
            else:
                tone = "green" if provenance.value == "verified" else "red"
                status_ribbon(f"Provenance: {provenance.value}", tone)
    st.caption("Archived, temporary, resume, PID, and log files are intentionally excluded.")


def _filtered_matches(table: pd.DataFrame, key: str) -> pd.DataFrame:
    filtered = table
    if "event_key" in filtered:
        events = sorted(filtered["event_key"].dropna().astype(str).unique())
        selected_events = st.multiselect("Events", events, key=f"{key}.events")
        if selected_events:
            filtered = filtered[filtered["event_key"].astype(str).isin(selected_events)]
    if "comp_level" in filtered:
        levels = sorted(filtered["comp_level"].dropna().astype(str).unique())
        selected_levels = st.multiselect("Competition level", levels, key=f"{key}.levels")
        if selected_levels:
            filtered = filtered[filtered["comp_level"].astype(str).isin(selected_levels)]
    return filtered


def render_data() -> None:
    context_bar("Data", "TBA match spine and scouting coverage", "Inspect")
    feature_tab, scouting_tab = st.tabs(["Feature tables", "Native scouting"])
    with feature_tab:
        artifact = _artifact_select("Feature table", (ArtifactKind.FEATURE_TABLE,), "data.feature")
        if artifact is not None:
            profile = _parquet_profile(*_signature_args(artifact))
            cols = st.columns(4)
            cols[0].metric("Rows", f"{profile.rows:,}")
            cols[1].metric("Columns", profile.columns)
            cols[2].metric("Row groups", profile.row_groups)
            cols[3].metric("Size", f"{profile.size_bytes / 1_048_576:.1f} MB")
            schema_tab, rows_tab, coverage_tab = st.tabs(["Schema", "Rows", "Coverage"])
            with schema_tab:
                st.dataframe(profile.schema, hide_index=True, width="stretch", height=460)
                trace = scouting_feature_trace(profile.schema["column"].astype(str).tolist())
                st.subheader("Scouting merge trace")
                if trace.empty:
                    st.info("This feature table has no recognized merged scouting columns.")
                else:
                    st.dataframe(trace, hide_index=True, width="stretch")
            table = _parquet(*_signature_args(artifact))
            with rows_tab:
                filtered = _filtered_matches(table, "data")
                st.caption(
                    f"Showing {min(len(filtered), 1_000):,} of {len(filtered):,} filtered rows."
                )
                st.dataframe(filtered.head(1_000), hide_index=True, width="stretch", height=520)
            with coverage_tab:
                profile_table = profile_dataframe(table).sort_values(
                    ["missing_percent", "column"], ascending=[False, True]
                )
                st.dataframe(
                    profile_table,
                    hide_index=True,
                    width="stretch",
                    height=520,
                    column_config={
                        "missing_percent": st.column_config.ProgressColumn(
                            "Missing %", min_value=0, max_value=100, format="%.1f%%"
                        )
                    },
                )
                coverage_columns = st.columns(3)
                coverage_columns[0].metric(
                    "Events",
                    table["event_key"].nunique(dropna=True) if "event_key" in table else "—",
                )
                coverage_columns[1].metric(
                    "Matches",
                    table["match_key"].nunique(dropna=True) if "match_key" in table else "—",
                )
                team_columns = [
                    column
                    for column in table
                    if str(column).startswith(("red_team_", "blue_team_"))
                    and str(column).endswith("_key")
                ]
                team_count = (
                    pd.unique(table[team_columns].astype(str).to_numpy().reshape(-1)).size
                    if team_columns
                    else 0
                )
                coverage_columns[2].metric("Teams", team_count or "—")
                numeric = [
                    column
                    for column in table.select_dtypes(include="number").columns
                    if any(
                        token in str(column)
                        for token in ("score", "pts", "win", "foul", "rank", "rp")
                    )
                ]
                if numeric:
                    target = st.selectbox("Target distribution", numeric, key="data.target")
                    values = table[[target]].dropna()
                    if len(values) > 100_000:
                        values = values.sample(100_000, random_state=2026)
                    figure = px.histogram(values, x=target, nbins=60)
                    figure.update_traces(marker_color="#0066B3")
                    st.plotly_chart(figure, width="stretch")
    with scouting_tab:
        artifact = _artifact_select(
            "Scouting database", (ArtifactKind.SCOUTING_DATABASE,), "data.scouting"
        )
        if artifact is not None:
            overview = scouting_overview(artifact.path)
            if overview.empty:
                st.info("The scouting database contains none of the supported scouting tables.")
            else:
                st.dataframe(overview, hide_index=True, width="stretch")
                selected = st.selectbox("Scouting table", overview["table"].tolist())
                timing = overview.loc[overview["table"] == selected, "timing"].iloc[0]
                status_ribbon(f"Timing: {timing}", "yellow" if "post" in timing else "blue")
                st.dataframe(
                    load_scouting_table(artifact.path, selected),
                    hide_index=True,
                    width="stretch",
                    height=460,
                )


def _run_directories() -> list[Path]:
    catalog = _get_catalog()
    names = {
        "metrics.csv",
        "walk_forward_metrics.csv",
        "calibration.csv",
        "feature_common_metrics.csv",
        "paired_bootstrap.csv",
        "coverage.json",
        "feature_history.csv",
        "prior_training_history.csv",
        "training_history.csv",
        "walk_forward_history.csv",
        "optimization_grid.csv",
        "development_selection.json",
    }
    directories = {
        artifact.path.parent for artifact in catalog.artifacts if artifact.path.name in names
    }

    def priority(path: Path) -> tuple[int, str]:
        manifest_path = path / "manifest.json"
        workflow = load_json(manifest_path).get("workflow") if manifest_path.exists() else None
        rank = {
            "season.2026-static-championship-core": -1,
            "season.reference-2026-static-reliability": 0,
            "season.reference-2026-static": 1,
        }.get(workflow, 2)
        return (rank, path.as_posix())

    return sorted(directories, key=priority)


def render_evaluation() -> None:
    context_bar(
        "Runs + Evaluation", "Probabilistic evidence before latent interpretation", "Evidence"
    )
    directories = _run_directories()
    if not directories:
        st.info("No supported evaluation directories were found.")
        return
    reference_index = next(
        (
            index
            for index, path in enumerate(directories)
            if (path / "manifest.json").exists()
            and load_json(path / "manifest.json").get("workflow")
            in {
                "season.2026-static-championship-core",
                "season.reference-2026-static-reliability",
                "season.reference-2026-static",
            }
        ),
        0,
    )
    selected = st.selectbox(
        "Evaluation run",
        directories,
        index=reference_index,
        format_func=lambda path: path.relative_to(_get_catalog().root).as_posix(),
        key="evaluation.run.v2",
    )
    manifest = (
        load_json(selected / "manifest.json") if (selected / "manifest.json").exists() else {}
    )
    promotion = manifest.get("promotion") if isinstance(manifest.get("promotion"), dict) else {}
    resolved = (
        manifest.get("resolved_config") if isinstance(manifest.get("resolved_config"), dict) else {}
    )
    workflow = manifest.get("workflow")
    is_reliability = workflow == "season.reference-2026-static-reliability"
    is_static_reference = workflow == "season.reference-2026-static"
    is_championship = workflow == "season.2026-static-championship-core"
    is_reference = is_reliability or is_static_reference or is_championship
    if is_reference:
        run_coverage = (
            load_json(selected / "coverage.json") if (selected / "coverage.json").exists() else {}
        )
        complete = bool(
            run_coverage.get("complete") if is_championship else resolved.get("complete")
        )
        conclusion_ready = bool(resolved.get("conclusion_ready", complete))
        rejection = resolved.get("rejection") if isinstance(resolved.get("rejection"), dict) else {}
        if is_championship:
            status_ribbon(
                "Complete fixed-100 Championship diagnostic"
                if complete
                else "Incomplete Championship diagnostic - conclusions suppressed",
                "blue" if complete else "yellow",
            )
        elif is_reliability and resolved.get("stage") == "development-rejected":
            status_ribbon("Development gate rejected - no test matrix", "yellow")
        elif is_reliability and complete and not conclusion_ready:
            status_ribbon("Complete neural matrix - awaiting Statbotics", "yellow")
        else:
            status_ribbon(
                "Complete reliability matrix"
                if is_reliability and complete
                else (
                    "Complete reference matrix"
                    if complete
                    else "Incomplete reference matrix - conclusions suppressed"
                ),
                "blue" if complete and conclusion_ready else "yellow",
            )
        summary_columns = st.columns(6)
        summary_columns[0].metric("Season", resolved.get("season", "—"))
        summary_columns[1].metric("State", resolved.get("state_model", "—"))
        summary_columns[2].metric("Latent dim", resolved.get("latent_dim", "—"))
        summary_columns[3].metric(
            "Seeds",
            ", ".join(map(str, resolved.get("seeds", [resolved.get("seed", "—")]))),
        )
        if is_championship:
            summary_columns[4].metric(
                "Test divisions", len(resolved.get("championship_event_keys", []))
            )
        else:
            summary_columns[4].metric(
                "Primary weeks",
                ", ".join(map(str, resolved.get("primary_weeks", resolved.get("test_weeks", []))))
                or "—",
            )
        summary_columns[5].metric("Promotion", "NO")
        if is_championship:
            st.caption(
                f"Fixed horizon: {resolved.get('epochs', 'unrecorded')} epochs | "
                f"loss weights: {resolved.get('loss_weights', {})} | "
                "Championship is reused diagnostic evidence | Einstein excluded"
            )
            st.warning(
                "One seed and a reused Championship holdout make this non-promotable. "
                "Architecture labels are fixed-horizon diagnostic effects only."
            )
        elif is_reliability:
            st.caption(
                f"Initialization: {resolved.get('initialization', 'unselected')} | "
                f"clip={resolved.get('selected_max_grad_norm', rejection.get('selected_max_grad_norm', 'unselected'))} | "
                f"epochs={resolved.get('selected_epochs', 'unselected')} | "
                f"stress week={resolved.get('stress_week', 'unrecorded')} | "
                f"known-as-of: {resolved.get('known_as_of_policy', 'unrecorded')}"
            )
            if rejection:
                st.warning(
                    f"{rejection.get('status', 'development-rejected')}: "
                    f"{rejection.get('reason', 'The development gate rejected this run.')}"
                )
        else:
            st.caption(
                f"Initialization: {resolved.get('selected_initialization', 'unselected')} | "
                f"Known-as-of: {resolved.get('known_as_of_policy', 'unrecorded')}"
            )
        if manifest.get("sources"):
            with st.expander("Source paths and hashes"):
                st.json(manifest["sources"])
        initialization_path = selected / "initialization_selection.csv"
        if initialization_path.exists():
            st.subheader("Development-only initialization selection")
            st.caption(
                "Week 4 selected initialization only. It is intentionally separated from final test evidence."
            )
            st.dataframe(load_csv(initialization_path), hide_index=True, width="stretch")
        if is_reliability:
            selection_path = selected / "development_selection.json"
            if selection_path.exists():
                st.subheader("Development-only optimization selection")
                st.caption(
                    "Week 4 alone selects clipping and fixed training duration. Test weeks do not "
                    "select epochs."
                )
                st.json(load_json(selection_path))
            grid_path = selected / "optimization_grid.csv"
            if grid_path.exists():
                st.subheader("Development optimization grid")
                st.caption(
                    "All rows are week-4 development evidence. Rejected grids remain available "
                    "for diagnosing metric and optimization conflicts."
                )
                st.dataframe(load_csv(grid_path), hide_index=True, width="stretch", height=360)
            confirmation_path = selected / "development_confirmation.csv"
            if confirmation_path.exists():
                st.dataframe(
                    load_csv(confirmation_path), hide_index=True, width="stretch", height=300
                )
    if promotion.get("eligible") is False or "v5.8" in selected.as_posix().lower():
        status_ribbon("Development evidence — not eligible for promotion", "yellow")
    elif promotion.get("eligible") is True:
        status_ribbon("Manifest marks this evidence promotion eligible", "green")
    else:
        status_ribbon("Promotion status is unverified", "yellow")

    metric_files = [
        "metrics.csv",
        "walk_forward_metrics.csv",
        "feature_common_metrics.csv",
        "feature_continuous_metrics.csv",
        "feature_binary_metrics.csv",
    ]
    tabs = st.tabs(["History", "Metrics", "Calibration", "Uncertainty", "Coverage", "Provenance"])
    with tabs[0]:
        history_path = next(
            (
                selected / name
                for name in (
                    "feature_history.csv",
                    "training_history.csv",
                    "prior_training_history.csv",
                    "walk_forward_history.csv",
                )
                if (selected / name).exists()
            ),
            None,
        )
        if history_path is None:
            st.info("No training-history artifact is present in this run.")
        else:
            history = load_csv(history_path)
            plotted_history = history
            if "seed" in history and history["seed"].nunique() > 1:
                seed_value = st.selectbox("History seed", sorted(history["seed"].unique()))
                plotted_history = plotted_history[plotted_history["seed"] == seed_value]
            if "model" in history and history["model"].nunique() > 1:
                model_value = st.selectbox("History model", sorted(history["model"].unique()))
                plotted_history = plotted_history[plotted_history["model"] == model_value]
            if "supervision_mode" in history and history["supervision_mode"].nunique() > 1:
                supervision = st.selectbox(
                    "History supervision", sorted(history["supervision_mode"].unique())
                )
                plotted_history = plotted_history[
                    plotted_history["supervision_mode"] == supervision
                ]
            if "architecture" in history and history["architecture"].nunique() > 1:
                architecture = st.selectbox(
                    "History architecture", sorted(history["architecture"].unique())
                )
                plotted_history = plotted_history[plotted_history["architecture"] == architecture]
            if "phase" in history and history["phase"].nunique() > 1:
                phase = st.selectbox("History phase", sorted(history["phase"].unique()))
                plotted_history = plotted_history[plotted_history["phase"] == phase]
            if "fold_number" in history and history["fold_number"].nunique() > 1:
                fold = st.selectbox(
                    "History fold",
                    sorted(history["fold_number"].dropna().unique()),
                )
                plotted_history = plotted_history[plotted_history["fold_number"] == fold]
            x = next((name for name in ("epoch", "step", "fold_number") if name in history), None)
            losses = [
                name
                for name in ("train_loss", "validation_loss", "loss", "val_loss")
                if name in history
            ]
            if x and losses:
                figure = px.line(
                    plotted_history,
                    x=x,
                    y=losses,
                    color_discrete_sequence=["#0066B3", "#9A989A"],
                )
                st.plotly_chart(figure, width="stretch")
            st.dataframe(history, hide_index=True, width="stretch", height=420)
    with tabs[1]:
        found = False
        for name in metric_files:
            path = selected / name
            if path.exists():
                st.subheader(name)
                metrics = load_csv(path)
                if (
                    is_championship
                    and name == "metrics.csv"
                    and {
                        "scope",
                        "supervision_mode",
                        "architecture",
                        "metric",
                        "value",
                    }.issubset(metrics.columns)
                ):
                    primary = metrics[
                        metrics["scope"].eq("primary-clean")
                        & metrics["probability_variant"].eq("raw_pred_red_win_probability")
                    ].copy()
                    primary["model"] = (
                        primary["supervision_mode"].astype(str)
                        + " / "
                        + primary["architecture"].astype(str)
                    )
                    scoreboard = primary.pivot_table(
                        index=["metric", "direction"],
                        columns="model",
                        values="value",
                        aggfunc="first",
                    ).reset_index()
                    st.subheader("Fixed-100 Championship scoreboard")
                    st.caption(
                        "All models use the exact same 1,108 primary-clean matches. Lower is "
                        "better except winner accuracy."
                    )
                    st.dataframe(scoreboard, hide_index=True, width="stretch")
                    classification_path = selected / "decision_classifications.csv"
                    if classification_path.exists():
                        st.subheader("Predeclared diagnostic classifications")
                        st.dataframe(
                            load_csv(classification_path), hide_index=True, width="stretch"
                        )
                    detail_scope = st.selectbox(
                        "Metric detail scope", sorted(metrics["scope"].unique())
                    )
                    st.dataframe(
                        metrics[metrics["scope"].eq(detail_scope)],
                        hide_index=True,
                        width="stretch",
                        height=360,
                    )
                elif name == "metrics.csv" and {
                    "scope",
                    "model",
                    "metric",
                    "value",
                }.issubset(metrics.columns):
                    aggregate = metrics[metrics["scope"] == "aggregate"]
                    scoreboard = aggregate
                    if is_reliability:
                        scoreboard = scoreboard[
                            scoreboard["seed"].astype(str).eq("ensemble")
                            & scoreboard["probability_variant"].isin(["raw", "not-applicable"])
                        ]
                    evidence = scoreboard.pivot_table(
                        index="metric", columns="model", values="value", aggfunc="first"
                    ).reset_index()
                    if "direction" in scoreboard:
                        directions = scoreboard[["metric", "direction"]].drop_duplicates("metric")
                        evidence = evidence.merge(directions, on="metric", how="left")
                    st.subheader("Primary matched-subset scoreboard")
                    st.caption(
                        "Direction is explicit: lower favors smaller losses; higher favors accuracy. "
                        "Aggregate values are observation weighted through the underlying match rows."
                    )
                    st.dataframe(evidence, hide_index=True, width="stretch")
                    if is_reliability:
                        st.caption(
                            "The scoreboard uses the three-seed ensemble on primary weeks 6 and 8. "
                            "Raw probabilities drive architecture decisions; week 10 is stress-only."
                        )
                    scope_options = [
                        value
                        for value in (
                            "test",
                            "event",
                            "observation_slice",
                            "development",
                        )
                        if value in set(metrics["scope"])
                    ]
                    if scope_options:
                        detail_scope = st.selectbox("Metric detail scope", scope_options)
                        detail = metrics[metrics["scope"] == detail_scope]
                        st.dataframe(detail, hide_index=True, width="stretch", height=360)
                    if {"latentstrat", "statbotics"}.issubset(evidence.columns):
                        evidence["latentstrat_minus_statbotics"] = (
                            evidence["latentstrat"] - evidence["statbotics"]
                        )
                        st.subheader("LatentStrat versus Statbotics")
                        st.caption(
                            "Candidate blue and baseline charcoal are model identities, not alliances. "
                            "Negative loss deltas favor LatentStrat."
                        )
                        st.dataframe(evidence, hide_index=True, width="stretch")
                        long = aggregate[aggregate["model"].isin(["latentstrat", "statbotics"])]
                        figure = px.bar(
                            long,
                            x="metric",
                            y="value",
                            color="model",
                            barmode="group",
                            color_discrete_map={
                                "latentstrat": "#0066B3",
                                "statbotics": "#343A40",
                            },
                        )
                        st.plotly_chart(figure, width="stretch")
                st.dataframe(metrics, hide_index=True, width="stretch")
                found = True
        auxiliary_path = selected / "auxiliary_metrics.csv"
        if auxiliary_path.exists():
            st.subheader("Official auxiliary outcomes")
            st.caption(
                "Ranking, alliance-selection, playoff, and award probes are separate from match-forecast evidence."
            )
            st.dataframe(load_csv(auxiliary_path), hide_index=True, width="stretch")
        utilization_path = selected / "parameter_utilization.csv"
        if utilization_path.exists():
            st.subheader("Parameter utilization")
            st.dataframe(load_csv(utilization_path), hide_index=True, width="stretch")
        external_path = selected / "external_comparison.csv"
        if external_path.exists():
            st.subheader("External Statbotics comparison")
            external = load_csv(external_path)
            if external.empty:
                st.warning("Statbotics evidence is required before the final conclusion.")
            else:
                st.dataframe(external, hide_index=True, width="stretch")
                verdict_path = selected / "external_verdict.json"
                if verdict_path.exists():
                    st.json(load_json(verdict_path))
        if not found:
            st.info("No supported metric CSV is present in this run.")
    with tabs[2]:
        path = selected / "calibration.csv"
        if not path.exists():
            path = selected / "feature_calibration.csv"
        if path.exists():
            calibration = load_csv(path)
            if is_championship and {
                "supervision_mode",
                "architecture",
            }.issubset(calibration.columns):
                calibration["model"] = (
                    calibration["supervision_mode"].astype(str)
                    + " / "
                    + calibration["architecture"].astype(str)
                )
            st.dataframe(calibration, hide_index=True, width="stretch")
            x = next(
                (name for name in ("mean_prediction", "mean_probability") if name in calibration),
                None,
            )
            y = next(
                (name for name in ("observed_rate", "fraction_positive") if name in calibration),
                None,
            )
            if x and y:
                fig = px.line(
                    calibration,
                    x=x,
                    y=y,
                    color="model" if "model" in calibration else None,
                    line_dash=(
                        "probability_variant" if "probability_variant" in calibration else None
                    ),
                )
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash"})
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No calibration artifact is present.")
    with tabs[3]:
        path = selected / "paired_bootstrap.csv"
        if path.exists():
            uncertainty = load_csv(path)
            st.dataframe(uncertainty, hide_index=True, width="stretch")
            st.caption("Negative candidate-minus-baseline loss deltas favor the candidate.")
            decisions_path = selected / "interaction_decisions.csv"
            championship_decisions = selected / "decision_classifications.csv"
            if is_championship and championship_decisions.exists():
                st.subheader("Fixed-100 diagnostic effects")
                st.dataframe(load_csv(championship_decisions), hide_index=True, width="stretch")
                loo_path = selected / "leave_one_division_out.csv"
                if loo_path.exists():
                    st.subheader("Leave-one-division-out sensitivity")
                    st.dataframe(load_csv(loo_path), hide_index=True, width="stretch")
            elif is_reliability and decisions_path.exists():
                decisions = load_csv(decisions_path)
                if not bool(resolved.get("conclusion_ready")):
                    st.warning(
                        "Interaction decisions are provisional until the exact matched Statbotics "
                        "comparison is complete."
                    )
                st.subheader("Interaction evidence classification")
                st.dataframe(decisions, hide_index=True, width="stretch")
                hierarchical_path = selected / "hierarchical_bootstrap.csv"
                if hierarchical_path.exists():
                    st.subheader("Hierarchical seed-event bootstrap")
                    st.dataframe(load_csv(hierarchical_path), hide_index=True, width="stretch")
            elif (
                is_reference
                and resolved.get("complete")
                and {
                    "candidate",
                    "baseline",
                    "metric",
                    "test_week",
                    "delta_mean",
                    "probability_of_improvement",
                }.issubset(uncertainty.columns)
            ):
                tolerance = {"winner_brier": 0.005, "winner_log_loss": 0.01}
                conclusions = []
                for keys, comparison in uncertainty.groupby(["candidate", "baseline", "metric"]):
                    supported_weeks = int((comparison["probability_of_improvement"] >= 0.80).sum())
                    metric_name = str(keys[2])
                    if metric_name == "score_differential_squared_error" and {
                        "candidate_loss_mean",
                        "baseline_loss_mean",
                    }.issubset(comparison.columns):
                        harmful_weeks = int(
                            (
                                comparison["candidate_loss_mean"].clip(lower=0).pow(0.5)
                                > 1.02 * comparison["baseline_loss_mean"].clip(lower=0).pow(0.5)
                            ).sum()
                        )
                    else:
                        practical = tolerance.get(metric_name, float("inf"))
                        harmful_weeks = int((comparison["delta_mean"] > practical).sum())
                    label = (
                        "supported"
                        if supported_weeks >= 2
                        else ("harmful" if harmful_weeks >= 2 else "uncertain")
                    )
                    conclusions.append(
                        {
                            "candidate": keys[0],
                            "baseline": keys[1],
                            "metric": keys[2],
                            "classification": label,
                            "supporting_weeks": supported_weeks,
                            "harmful_weeks": harmful_weeks,
                        }
                    )
                st.subheader("Interaction evidence classification")
                st.dataframe(pd.DataFrame(conclusions), hide_index=True, width="stretch")
            elif is_reference:
                st.info(
                    "Architectural conclusions are hidden until the complete matrix is present."
                )
        else:
            st.info("No event-cluster paired bootstrap artifact is present.")
    with tabs[4]:
        path = selected / "coverage.json"
        if path.exists():
            st.json(load_json(path))
        else:
            st.info("No explicit coverage report is present.")
    with tabs[5]:
        st.json(manifest if manifest else {"status": "No manifest found"})


def render_model() -> None:
    context_bar("Model", "Checkpoint structure and parameter diagnostics", "CPU only")
    artifact = _artifact_select(
        "Checkpoint",
        (
            ArtifactKind.SEASON_CHECKPOINT,
            ArtifactKind.PRIOR_CHECKPOINT,
            ArtifactKind.MATCH_BREAKDOWN_CHECKPOINT,
        ),
        "model.checkpoint.reference-v1",
        preferred_path_token="artifacts/reference/",
    )
    if artifact is None:
        return
    try:
        payload = _checkpoint(*_signature_args(artifact), str(_get_catalog().root))
        summary = checkpoint_summary(artifact.path, payload)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return
    cols = st.columns(4)
    cols[0].metric("Kind", summary.kind)
    cols[1].metric("Schema", summary.schema_version or "historical")
    cols[2].metric("Latent dimensions", summary.latent_dim or "—")
    cols[3].metric("Parameters", f"{summary.parameter_count:,}")
    utilization_path = next(
        (
            parent / "parameter_utilization.csv"
            for parent in artifact.path.parents
            if (parent / "parameter_utilization.csv").exists()
        ),
        None,
    )
    if utilization_path is not None:
        utilization = load_csv(utilization_path)
        architecture = summary.metadata.get("match_architecture")
        test_week = summary.metadata.get("test_week")
        if "model" in utilization and architecture:
            utilization = utilization[utilization["model"].astype(str) == str(architecture)]
        if "test_week" in utilization and test_week is not None:
            utilization = utilization[
                pd.to_numeric(utilization["test_week"], errors="coerce") == int(test_week)
            ]
        if not utilization.empty:
            total_rows = (
                utilization[utilization["module"].astype(str) == "<all>"]
                if "module" in utilization
                else utilization
            )
            total = total_rows.iloc[0]
            utilization_columns = st.columns(4)
            utilization_columns[0].metric("Nominal", f"{int(total['nominal_parameters']):,}")
            utilization_columns[1].metric("Trainable", f"{int(total['trainable_parameters']):,}")
            utilization_columns[2].metric(
                "Gradient receiving",
                f"{int(total['gradient_receiving_parameters']):,}",
            )
            utilization_columns[3].metric("Active Z_base rows", int(total["active_z_base_rows"]))
            if "module" in utilization:
                st.caption("Final refit parameter utilization by module")
                st.dataframe(utilization, hide_index=True, width="stretch")
    metadata_tab, parameters_tab, options_tab = st.tabs(["Metadata", "Parameters", "Options"])
    with metadata_tab:
        st.json(summary.metadata)
        st.subheader("Module hierarchy")
        hierarchy = module_hierarchy(payload)
        if hierarchy.empty:
            st.info("No recognized module hierarchy is present.")
        else:
            st.dataframe(hierarchy, hide_index=True, width="stretch", height=360)
    parameters = parameter_table(payload)
    with parameters_tab:
        if parameters.empty:
            st.info("This checkpoint contains no recognized tensor state.")
        else:
            st.dataframe(parameters, hide_index=True, width="stretch", height=440)
            selected = st.selectbox("Tensor histogram", parameters["parameter"].tolist())
            values = tensor_values(payload, selected)
            fig = px.histogram(x=values, nbins=60, labels={"x": selected})
            fig.update_traces(marker_color="#0066B3")
            st.plotly_chart(fig, width="stretch")
    with options_tab:
        st.json(summary.options)
        targets = summary.options.get("target_map")
        if targets:
            st.subheader("Continuous target definitions")
            st.dataframe(pd.DataFrame(targets), hide_index=True, width="stretch")
        st.caption(
            "Persisted freeze metadata is authoritative. A reconstructed model's requires_grad "
            "flags do not prove how the checkpoint was trained."
        )


def _embedding_source():
    artifacts = _get_catalog().of_kind(
        ArtifactKind.SEASON_CHECKPOINT,
        ArtifactKind.PRIOR_CHECKPOINT,
        ArtifactKind.MATCH_BREAKDOWN_EMBEDDINGS,
    )
    if not artifacts:
        st.info("No supported embedding artifacts were found.")
        return None
    artifacts = tuple(
        sorted(
            artifacts,
            key=lambda artifact: (
                0 if "artifacts/reference/" in artifact.path.as_posix() else 1,
                artifact.path.as_posix(),
            ),
        )
    )
    requested = str(st.query_params.get("artifact", ""))
    requested_index = next(
        (
            index
            for index, artifact in enumerate(artifacts)
            if requested and requested in artifact.path.as_posix()
        ),
        0,
    )
    artifact_id = st.selectbox(
        "Embedding source",
        [artifact.artifact_id for artifact in artifacts],
        format_func=lambda value: (
            _get_catalog().by_id(value).path.relative_to(_get_catalog().root).as_posix()
        ),
        index=requested_index,
        key=f"embeddings.source.reference-v1.{requested or 'default'}",
    )
    return _get_catalog().by_id(artifact_id)


def render_embeddings() -> None:
    context_bar("Embeddings", "Post-training geometry; not semantic ground truth", "Diagnostic")
    artifact = _embedding_source()
    if artifact is None:
        return
    trajectory_mode = False
    try:
        if artifact.kind == ArtifactKind.MATCH_BREAKDOWN_EMBEDDINGS:
            frame = _parquet(*_signature_args(artifact))
            vector_columns = tuple(column for column in frame if str(column).startswith("z_"))
            basis = f"match-breakdown:{artifact.path.parent.name}"
            label = "Alliance-match breakdown embeddings"
        else:
            payload = _checkpoint(*_signature_args(artifact), str(_get_catalog().root))
            if artifact.kind == ArtifactKind.PRIOR_CHECKPOINT:
                frame, vector_columns = prior_embedding_frame(payload)
                basis = f"prior:{artifact.path.parent.name}"
                label = "Prior team embeddings"
            else:
                event_map = payload.get("team_event_index_map") or {}
                event_keys = sorted(
                    {str(key).split("::", 1)[0] for key in event_map if "::" in str(key)}
                )
                is_static = (payload.get("options") or {}).get("state_model") == "static-z-base"
                choices = (
                    ["Base table"] if is_static else ["Base table", "All event states", *event_keys]
                )
                representation = st.selectbox("Representation", choices)
                if is_static:
                    st.info(
                        "Static Z_base reference: event trajectories and temporal-state claims are unavailable by design."
                    )
                if representation == "All event states":
                    frame, vector_columns = season_event_trajectory_frame(payload)
                    trajectory_mode = True
                else:
                    selected_event = None if representation == "Base table" else representation
                    frame, vector_columns, _ = season_embedding_frame(
                        payload, event_key=selected_event
                    )
                basis = f"season:{artifact.path.parent.name}"
                label = "Season robot representation"
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return
    space = EmbeddingSpaceRef(
        space_id=basis,
        label=label,
        basis_id=basis,
        artifact=artifact,
        vector_columns=vector_columns,
        description="Post-training diagnostic space",
    )
    pca = compute_pca(space, frame)
    variance = pca.explained_variance_ratio
    st.caption(
        "Mean-centered full-SVD PCA. "
        + ", ".join(f"PC{index + 1}: {value:.1%}" for index, value in enumerate(variance))
    )
    coordinates = pca.coordinates
    plot = (
        coordinates if len(coordinates) <= 15_000 else coordinates.sample(15_000, random_state=2026)
    )
    hover = next((name for name in ("team_key", "match_key", "row_id") if name in plot), None)
    color = "season" if "season" in plot and plot["season"].nunique() > 1 else None
    fig = px.scatter(plot, x="PC1", y="PC2", color=color, hover_name=hover, opacity=0.72)
    st.plotly_chart(fig, width="stretch", on_select="ignore")
    if trajectory_mode and {"team_key", "event_key"}.issubset(coordinates.columns):
        trajectory_team = st.selectbox(
            "Event-state trajectory team",
            sorted(coordinates["team_key"].astype(str).unique()),
        )
        trajectory = coordinates[
            coordinates["team_key"].astype(str) == trajectory_team
        ].sort_values("event_key", kind="mergesort")
        trajectory_figure = px.line(
            trajectory,
            x="PC1",
            y="PC2",
            markers=True,
            hover_name="event_key",
        )
        st.plotly_chart(trajectory_figure, width="stretch")
        st.caption(
            "Points follow event-key order because event dates are not persisted in this "
            "checkpoint. Treat the line as an event-state diagnostic, not a time-causal path."
        )
    label_column = next(
        (name for name in ("team_key", "row_id", "match_key") if name in frame), None
    )
    if label_column:
        options = frame.index.tolist()
        query = st.selectbox(
            "Neighbor query",
            options,
            format_func=lambda index: str(frame.loc[index, label_column]),
        )
        neighbors = cosine_neighbors(space, frame.reset_index(drop=True), int(query), limit=12)
        shown = [
            column
            for column in ("cosine_similarity", label_column, "season", "event_key", "alliance")
            if column in neighbors
        ]
        st.dataframe(neighbors[shown], hide_index=True, width="stretch")
    st.caption(
        "PCA and cosine neighbors generate hypotheses. Coordinates are not robot traits, causal "
        "effects, or evidence that two different latent bases are comparable."
    )


def _team_board(red: tuple[str, ...], blue: tuple[str, ...], title: str) -> None:
    red_rows = "".join(f'<div class="ls-team-row">{html.escape(team)}</div>' for team in red)
    blue_rows = "".join(f'<div class="ls-team-row">{html.escape(team)}</div>' for team in blue)
    st.markdown(
        '<div class="ls-team-board">'
        f'<div class="ls-alliance ls-alliance--red"><h3>Red · {html.escape(title)}</h3>{red_rows}</div>'
        '<div class="ls-versus">VS</div>'
        f'<div class="ls-alliance ls-alliance--blue"><h3>Blue · {html.escape(title)}</h3>{blue_rows}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _scoreboard(red_score: float | None, blue_score: float | None, probability: float) -> None:
    red_text = "—" if red_score is None else f"{red_score:.1f}"
    blue_text = "—" if blue_score is None else f"{blue_score:.1f}"
    st.markdown(
        '<div class="ls-scoreboard">'
        '<div class="ls-score ls-score--red"><div class="ls-score__label">Predicted red</div>'
        f'<div class="ls-score__value">{red_text}</div></div>'
        '<div class="ls-score__center"><strong>FORECAST</strong>'
        f"<span>Red win {probability:.1%}</span></div>"
        '<div class="ls-score ls-score--blue"><div class="ls-score__label">Predicted blue</div>'
        f'<div class="ls-score__value">{blue_text}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _actual_scores(row: pd.Series) -> tuple[float | None, float | None]:
    for red, blue in (
        ("red_total_score", "blue_total_score"),
        ("red_score_raw", "blue_score_raw"),
        ("actual_red_total_score", "actual_blue_total_score"),
    ):
        if red in row and blue in row and pd.notna(row[red]) and pd.notna(row[blue]):
            return float(row[red]), float(row[blue])
    return None, None


def _score_breakdown(row: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for red_column in row.index:
        if not str(red_column).startswith("red_"):
            continue
        suffix = str(red_column)[4:]
        blue_column = f"blue_{suffix}"
        if blue_column not in row.index or "team_" in suffix or "pred" in suffix:
            continue
        red_value, blue_value = row[red_column], row[blue_column]
        if pd.api.types.is_number(red_value) and pd.api.types.is_number(blue_value):
            if pd.notna(red_value) or pd.notna(blue_value):
                rows.append(
                    {
                        "breakdown": suffix.replace("_", " ").upper(),
                        "red": red_value,
                        "blue": blue_value,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("breakdown").head(40)


def _saved_predictions(match_key: str) -> pd.DataFrame:
    frames = []
    for artifact in _get_catalog().of_kind(ArtifactKind.PREDICTIONS):
        try:
            table = _parquet(*_signature_args(artifact))
        except (OSError, RuntimeError, ValueError):
            continue
        if "match_key" not in table:
            continue
        rows = table[table["match_key"].astype(str) == match_key]
        if not rows.empty:
            rows = rows.copy()
            rows["_source_path"] = str(artifact.path)
            frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def render_match() -> None:
    context_bar("Match", "Preview → prediction → result", "Diagnostic")
    feature = _artifact_select(
        "Feature table",
        (ArtifactKind.FEATURE_TABLE,),
        "match.feature.reference-v1",
        preferred_path_token="data/features/season/features_v58_2026.parquet",
    )
    checkpoint = _artifact_select(
        "Season checkpoint",
        (ArtifactKind.SEASON_CHECKPOINT,),
        "match.checkpoint.reference-v1",
        preferred_path_token="artifacts/reference/",
    )
    if feature is None:
        return
    table = _parquet(*_signature_args(feature))
    required = {
        "event_key",
        "match_key",
        *[f"{color}_team_{slot}_key" for color in ("red", "blue") for slot in range(1, 4)],
    }
    if not required.issubset(table.columns):
        st.error("Selected feature table does not satisfy the match-spine schema.")
        return
    events = sorted(table["event_key"].dropna().astype(str).unique())
    event = st.selectbox("Event", events)
    event_rows = table[table["event_key"].astype(str) == event]
    match_key = st.selectbox("Match", event_rows["match_key"].astype(str).tolist())
    row = event_rows[event_rows["match_key"].astype(str) == match_key].iloc[0]
    red = tuple(str(row[f"red_team_{slot}_key"]) for slot in range(1, 4))
    blue = tuple(str(row[f"blue_team_{slot}_key"]) for slot in range(1, 4))
    _team_board(red, blue, match_key)

    saved_options = _saved_predictions(match_key)
    saved = None
    if not saved_options.empty:
        labels = []
        for index, candidate in saved_options.iterrows():
            identity = [
                str(candidate[name])
                for name in (
                    "model",
                    "architecture",
                    "seed",
                    "prediction_level",
                    "initialization",
                    "fold_number",
                    "test_week",
                )
                if name in candidate and pd.notna(candidate[name])
            ]
            labels.append(
                f"{index}: " + " · ".join(identity or [Path(str(candidate["_source_path"])).name])
            )
        selected_prediction = st.selectbox(
            "Saved prediction variant",
            list(range(len(saved_options))),
            format_func=lambda index: labels[index],
        )
        saved = saved_options.iloc[int(selected_prediction)]
    actual_red, actual_blue = _actual_scores(row)
    if saved is not None:
        saved_red = saved.get("pred_red_total_score", saved.get("statbotics_pred_red_score"))
        saved_blue = saved.get("pred_blue_total_score", saved.get("statbotics_pred_blue_score"))
        saved_probability = saved.get(
            "pred_red_win_probability", saved.get("statbotics_pred_red_win_probability")
        )
        if pd.notna(saved_probability):
            status_ribbon(
                "Saved prediction artifact — use its manifest for evidence status", "blue"
            )
            _scoreboard(
                float(saved_red) if pd.notna(saved_red) else None,
                float(saved_blue) if pd.notna(saved_blue) else None,
                float(saved_probability),
            )
            evidence = [str(saved["_source_path"])]
            for name in (
                "model",
                "architecture",
                "initialization",
                "seed",
                "prediction_level",
                "evidence_role",
                "known_as_of",
                "test_week",
                "val_week",
                "fold_number",
                "calibration_method",
                "calibration_source_weeks",
                "calibration_source_rows",
                "checkpoint_path",
            ):
                if name in saved and pd.notna(saved[name]):
                    evidence.append(f"{name}={saved[name]}")
            st.caption(" · ".join(evidence))

    if actual_red is not None and actual_blue is not None:
        st.subheader("Result")
        result_columns = st.columns(4)
        result_columns[0].metric("Actual red", f"{actual_red:.0f}")
        result_columns[1].metric("Actual blue", f"{actual_blue:.0f}")
        result_columns[2].metric("Score differential", f"{actual_red - actual_blue:+.0f}")
        result_columns[3].metric(
            "Actual winner",
            "TIE" if actual_red == actual_blue else ("RED" if actual_red > actual_blue else "BLUE"),
        )
        breakdown = _score_breakdown(row)
        if not breakdown.empty:
            st.dataframe(breakdown, hide_index=True, width="stretch")

    run_diagnostic = checkpoint is not None and st.button(
        "Run diagnostic checkpoint inference", icon=":material/science:"
    )
    if run_diagnostic and checkpoint is not None:
        status_ribbon("Interactive checkpoint inference is diagnostic, not out-of-time evidence")
        try:
            payload = _checkpoint(*_signature_args(checkpoint), str(_get_catalog().root))
            compatibility = checkpoint_feature_compatibility(payload, table)
            if not compatibility.compatible:
                raise ValueError("Incompatible artifacts: " + "; ".join(compatibility.reasons))
            for warning in compatibility.warnings:
                st.warning(warning)
            diagnostic = diagnose_match(payload, row)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            _scoreboard(
                diagnostic.predicted_red_score,
                diagnostic.predicted_blue_score,
                diagnostic.red_win_probability,
            )
            if diagnostic.unknown_teams:
                st.warning(
                    "Teams absent from the checkpoint vocabulary were explicitly routed to the "
                    f"learned ghost row: {', '.join(diagnostic.unknown_teams)}"
                )
            cols = st.columns(2)
            cols[0].metric("Alliance-swap gap", f"{diagnostic.swap_probability_gap:.2e}")
            cols[1].metric("Unknown teams", len(diagnostic.unknown_teams))
            if actual_red is not None and actual_blue is not None:
                error_cols = st.columns(3)
                error_cols[0].metric(
                    "Red score error",
                    "—"
                    if diagnostic.predicted_red_score is None
                    else f"{diagnostic.predicted_red_score - actual_red:+.1f}",
                )
                error_cols[1].metric(
                    "Blue score error",
                    "—"
                    if diagnostic.predicted_blue_score is None
                    else f"{diagnostic.predicted_blue_score - actual_blue:+.1f}",
                )
                error_cols[2].metric(
                    "Actual winner",
                    "TIE"
                    if actual_red == actual_blue
                    else ("RED" if actual_red > actual_blue else "BLUE"),
                )
            pma = pd.DataFrame(
                {
                    "slot": [1, 2, 3],
                    "red_team": red,
                    "red_pma_pooling_weight": diagnostic.red_pma_weights,
                    "blue_team": blue,
                    "blue_pma_pooling_weight": diagnostic.blue_pma_weights,
                }
            )
            st.subheader("PMA pooling diagnostics")
            st.dataframe(pma, hide_index=True, width="stretch")
            st.caption("Pooling weights are model-behavior diagnostics, not causal attribution.")
            st.subheader("Slot-zero sensitivity")
            st.dataframe(diagnostic.zero_slot, hide_index=True, width="stretch")


PAGES: dict[str, Any] = {
    "Workspace": render_workspace,
    "Data": render_data,
    "Runs + Evaluation": render_evaluation,
    "Model": render_model,
    "Embeddings": render_embeddings,
    "Match": render_match,
}


__all__ = ["PAGES", "configure_page", "render_sidebar"]
