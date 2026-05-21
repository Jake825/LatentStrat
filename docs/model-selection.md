# Model Selection

LatentStrat uses one official architecture: the cross-alliance Set Transformer.

## Project History

LatentStrat began as a numerical prototype that established the core scientific
requirements: leakage prevention, valid split generation, target standard
scaling, and the cross-alliance Set Transformer formulation.

The current implementation treats the Python codebase as the source of truth.
PyTorch, Pandas, and PyArrow now define the production workflow.

## Why Python

The Python implementation turns LatentStrat into deployable ML infrastructure:

1. **Performance**: PyTorch uses Automatic Mixed Precision, `torch.compile`, and
   asynchronous DataLoaders when the local hardware supports them.
2. **Data-lake workflow**: `requests-cache` and Parquet feature files decouple
   API extraction from the training loop.
3. **Deployment options**: The model can be exported or wrapped by standard
   Python serving tools.
4. **Future modalities**: Python is the standard ecosystem for adding image and
   text encoders when future scouting data includes robot imagery or notes.

## Architectural Consolidation

Early discovery versions included additive and pairwise interaction experiments.
Those variants are useful historical ablations, but the production model path is
now consolidated around the Set Transformer.

Controls that test the architecture, such as shuffling teams or nullifying
labels, are executed at the Pandas data layer before hitting PyTorch. The neural
network only acts as a match simulator.

## Interpretation Discipline

LatentStrat enforces interpretation discipline. PMA weights and multi-task heads
are diagnostics, not proof of causal robot importance.

The model is considered useful only when held-out metrics such as RMSE, Brier
score, and log loss improve against linear baselines and data-level null
controls. Attention plots alone are not accepted as evidence.
