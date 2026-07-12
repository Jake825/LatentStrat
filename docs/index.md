# LatentStrat Documentation

LatentStrat V6.1 has three supported workflow families:

1. [Prior pretraining](prior-pretraining.md): build Day Zero `Z_base` initializers.
2. [Match-breakdown pretraining](match-breakdown-pretraining.md): learn offline 16D alliance-result
   embeddings from historical TBA payloads.
3. [Season training](season-training.md): train and validate the supervised Set Transformer.

Start with [V6.1](V6.1.md), then use [Rebuild from scratch](rebuild-from-scratch.md) and the
[CLI reference](cli-reference.md) for runnable commands.

The [2026 static robot-state reference study](reference-2026-static-study.md) defines the current
architecture-validation contract, dashboard acceptance gate, and non-promotion boundary.
The follow-up [static architecture reliability study](reference-2026-static-reliability.md)
tests whether its interaction results survive frozen optimization and three random seeds.
The [2026 static training postmortem](2026-static-training-postmortem.md) records the completed
reference, smoke runs, rejected reliability grid, score-bias analysis, and clean-reset boundary.

## Active References

- [Architecture](architecture.md)
- [Evaluation and artifacts](evaluation-and-artifacts.md)
- [Research workbench](streamlit-app.md)
- [Current state](current-state.md)
- [Roadmap](roadmap.md)
- [Data sources](data-sources.md)
- [Experiment ledger](experiment-ledger.md)
- [2026 static training postmortem](2026-static-training-postmortem.md)
- [Changelog](changelog.md)
- [Student primer](student-primer.md)
- [Scouting data layer](scouting-data-layer.md)
- [Scouting ingestion](scouting-data-ingestion.md)

## History

Older version narratives and superseded reference pages remain available in the
[archive](archive/index.md). They are historical context, not the active command or storage
contract.
