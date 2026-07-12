# Research Workbench

The LatentStrat Research Workbench is a read-only Streamlit application for inspecting local datasets, scouting coverage, training artifacts, checkpoints, learned representations, saved predictions, and evaluation evidence. It does not train or fine-tune models, modify scouting data, call TBA or Statbotics, or write model artifacts.

The visual language draws on the information hierarchy of the modern FRC audience display without copying official screens or assets. The [2023 Audience Displays](https://www.firstinspires.org/hubfs/blog/frc/2023-audience-display-blog.pdf) post introduced the full redesign. The [2024](https://www.firstinspires.org/hubfs/blog/frc/2024-audience-displays-blog.pdf), [2025](https://community.firstinspires.org/2025-scoreboard-and-live-stream-graphics), and [2026](https://community.firstinspires.org/2026-scoreboard-and-live-stream-graphics) graphics refine the same red/blue symmetry, neutral comparison lanes, condensed numeric hierarchy, and status-yellow convention.

LatentStrat is an unofficial community analytics project. It is not affiliated with or endorsed by FIRST. The app does not include FIRST, FRC, season, sponsor, or ranking-point artwork.

## Install And Run

From the repository root:

```powershell
uv sync --extra app --extra dev
uv run --extra app streamlit run app/app.py
```

The default workspace is the current directory. Override it when the data and artifacts live in another LatentStrat workspace:

```powershell
$env:LATENTSTRAT_APP_ROOT = "C:\path\to\LatentStrat"
uv run --extra app streamlit run app/app.py
```

Roboto and Roboto Condensed are self-hosted under `app/static/fonts/` using Streamlit's static-file support. The bundled Open Font License applies to those files. A small Material Symbols subset is also self-hosted under its bundled Apache 2.0 license so navigation remains legible without internet access. Arial and the browser's sans-serif font remain fallbacks.

## Trusted LAN Use

The application has no authentication. Expose it only on a trusted team network:

```powershell
uv run --extra app streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Do not use this configuration for public hosting. The app displays local paths, schemas, checkpoint options, and model parameters.

## Pages

- **Workspace** catalogs completed artifacts under canonical `data/` and `artifacts/` roots. It excludes archives, logs, PID files, temporary files, and resume checkpoints.
- **Data** inspects the TBA match-grain Parquet spine and reads scouting SQLite at its native team, event, match, team-event, alliance-match, and team-match grains.
- **Runs + Evaluation** shows metrics, calibration, coverage, event-cluster bootstrap comparisons, and manifest promotion status when those files exist.
- **Model** summarizes checkpoint schemas, options, persisted freeze metadata, parameter shapes, tensor statistics, and bounded histograms.
- **Embeddings** provides post-training PCA and cosine-neighbor diagnostics for prior-team, season-team, or alliance-match breakdown spaces. Different bases cannot be compared as one coordinate system.
- **Match** shows saved prediction artifacts and optional CPU checkpoint inference for an existing match, including PMA pooling and slot-zero sensitivity diagnostics.

For the 2026 static reference contract, Runs + Evaluation adds a manifest summary, visibly
separates development-only initialization selection from test evidence, shows all architecture and
control models with metric direction, and suppresses conclusions for incomplete matrices. Model
reports schema-`7` static-state metadata and parameter utilization. Embeddings exposes only the
fixed `Z_base`, PCA, and neighbors; it cannot display event trajectories. Match lets the reviewer
choose architecture, initialization, fold, and known-as-of context from saved predictions.

For the static reliability contract, the evaluation page separates week-4 development selection
from the final matrix, filters primary scoreboard evidence to the seed ensemble and raw
probability variant, and shows hierarchical seed/event intervals plus interaction classifications.
It labels the external verdict `awaiting-statbotics` until a verified baseline is attached. Smoke,
interrupted, and partial runs remain visibly incomplete; Model and Match include seed, evidence
role, calibration provenance, and the recorded static architecture.

## Interpretation Boundaries

- PCA, cosine neighbors, PMA pooling weights, and slot-zero results are diagnostics, not named robot traits or causal explanations.
- Interactive checkpoint inference is diagnostic. It may be in-sample and is not automatically out-of-time evidence.
- The historical V5.8 predictions remain development evidence and are not promotion eligible because the reported validation weeks also selected their epochs.
- Statbotics is an external benchmark, not ground truth. The app displays a comparison only when the metric pipeline has produced aligned saved-prediction artifacts.
- Missing scouting data stays missing. Native match, alliance-match, and team-match observations are treated as during/post-match unless their source proves a stricter `known_as_of` boundary.
- A team absent from a selected checkpoint vocabulary is visibly routed to the learned ghost row for diagnostic inference. The app never hides that fallback.

## Read Safety

Checkpoint paths must stay inside the configured workspace and are loaded on CPU with PyTorch's weights-only mode. The app compares file size and modification time before and after reads; if a training process changes a file, the read is rejected and the user can refresh after the writer finishes.

The catalog distinguishes a manifest-declared hash from a verified hash. It does not hash every large file during startup; use the explicit provenance check on the Workspace page to stream and verify one artifact without mutating it.

## Validation

```powershell
uv run --extra app pytest tests/test_latentstrat_app.py
uv run ruff check .
uv lock --check
git diff --check
```

The app tests use compact fixture checkpoints and do not run optimizer or training loops.
