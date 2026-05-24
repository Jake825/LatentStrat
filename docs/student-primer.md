---
tags:
  - latentstrat
  - student-primer
aliases:
  - "Student Primer"
  - "LatentStrat Beginner Guide"
related:
  - "[[index]]"
  - "[[ghost robot]]"
  - "[[Set Transformer]]"
  - "[[prior-training]]"
  - "[[season-training]]"
  - "[[model-structure]]"
  - "[[model-architecture-reference]]"
---

# Student Primer

This page explains LatentStrat for an FRC student who knows scouting and match strategy but may not know machine learning yet.

## What LatentStrat Is Trying To Do

LatentStrat is a scouting and strategy model. It tries to answer questions like:

- Which alliance is more likely to win?
- Which teams are stronger than their record suggests?
- Which teams fit well together?
- How confident should we be in a prediction?
- What did the model learn about a team after an event?

It does this by learning a small vector for each team. A vector is just a list of numbers. In the current model, that list has 16 numbers.

## Day Zero Prior

The Day Zero prior is what the model knows before the current season's matches are used.

It is built from:

- Team narratives.
- OpenAI text embeddings of those narratives.
- Statbotics normalized EPA from previous seasons.
- Team history such as rookie year, seasons played, blue banners, and technical awards.

The prior gives every team number a starting identity. It is not perfect, but it prevents the model from treating every team as totally unknown on Week 1.

## `Z_base`

`Z_base` is the long-term team identity table. You can think of it like the model's memory of each team.

Examples:

- `Z_base[254]` is the long-term learned vector for `frc254`.
- `Z_base[2290]` is the long-term learned vector for `frc2290`.
- `Z_base[0]` is the learned ghost robot.

The V5.6.4 prior checkpoint initializes this table.

## `Z_event`

`Z_event` is what the model learns during an event or season.

If a team shows up with a better robot than expected, `Z_event` can move that team's match representation. If a team breaks often or plays differently than their history suggests, `Z_event` can capture that too.

The team representation during season training is:

```text
team = Z_base + Z_event
```

## Ghost Robot

Team `0` is the ghost robot. It represents an empty or missing robot slot.

The ghost robot is learned, not deleted. This matters because a two-robot alliance is not the same as a three-robot alliance with one robot hidden from the math. The Set Transformer still sees three slots and can learn how a missing robot changes alliance dynamics.

For the dedicated concept note, see [Ghost robot](ghost%20robot.md).

## Set Transformer

An FRC alliance is a set of three teams. The order of the team slots should not matter much. Swapping red team 1 and red team 2 should not completely change the prediction.

The Set Transformer helps the model learn:

- How teammates interact.
- How the red alliance interacts with the blue alliance.
- Which robot slots matter most for a prediction.

It pools three robot vectors into one red alliance vector and one blue alliance vector, then predicts match targets from those alliance vectors.

## Sidecars

Sidecars are extra training labels that are not match rows.

LatentStrat uses:

- Rankings sidecars.
- Alliance selection sidecars.
- Playoff sidecars.

These teach the model useful strategy structure. For example, alliance selection data can teach the model which picked teams were preferred over higher-ranked passed-over teams.

Sidecars are not pre-match inputs. They are post-event labels used for training.

## Walk-Forward Validation

Walk-forward validation simulates time.

Instead of randomly mixing matches, it does:

```text
Train on Week 1
Validate on Week 2

Train on Weeks 1-2
Validate on Week 3

Train on Weeks 1-3
Validate on Week 4
```

Each fold starts over from the Day Zero prior. This prevents the model from accidentally learning from the future.

This is the main validation method because it is close to the Friday-night question: "Given what we knew before this week, how well would we predict this week's matches?"

## Metrics

### Match Accuracy

Match accuracy asks: did the model pick the winner?

This is easy to understand, but it does not tell the whole story.

### Brier Score

Brier score checks probability quality.

If the model says red has a 70 percent chance to win, then red should win about 70 percent of those matches over many examples. Lower Brier is better.

### Log Loss

Log loss punishes confident wrong predictions. If the model says a team has a 99 percent chance to win and they lose, log loss gets very bad.

This is why the current project cares about calibration, not just accuracy.

### Score MSE

Score MSE checks how close the predicted score is to the real score. Lower is better.

LatentStrat reports:

- Phase score MSE: auto plus teleop points.
- Total score MSE: FMS total approximation when foul data is available.

## TensorBoard

TensorBoard is a local dashboard for watching training.

Start it with:

```bash
tensorboard --logdir=runs
```

Useful curves:

- Training loss.
- Validation loss.
- Learning rate.
- Task weights.
- OpenAI, EPA, culture, match, and sidecar losses.

If training loss improves but validation gets worse, the model is probably overfitting.

## How To Click Through The Vault

Recommended path:

1. [Current state](current-state.md)
2. [Data sources](data-sources.md)
3. [Prior training](prior-training.md)
4. [Season training](season-training.md)
5. [Metrics and artifacts](metrics-and-artifacts.md)
6. [Experiment ledger](experiment-ledger.md)
7. [Rebuild from scratch](rebuild-from-scratch.md)
