# Aura's subjecthood experiment

Aura has software for memory, attention, and tracking its own state. The
Intrinsic Subject Core (ISC) study asks how much those parts depend on one
another as Aura runs. For example, when researchers change a memory signal,
they can measure whether the software that models its surroundings responds.
They can also hold two groups of parts apart and look for a change in how
the groups behave.

## How a comparison works

The experiment restores several copies of the same saved moment. It changes
a chosen part in one copy and uses an unchanged copy to measure drift. For a
joint effect, it changes two parts separately, then changes both together.
The researchers compare what happens over the next few steps.

Most checks run Aura's software with a deterministic stand-in for language
generation. The report experiment uses its 27B language model and asks for a
rating after changing an internal value called *valence*. The model receives
that change through a steering channel fitted to carry valence.
[Methods and controls](ISC_STATUS.md)

## What happened

The November preprint reports 24 of 24 ISC-v7 checks passing on the
confirmatory run, seed 101. Its carrier test selected all ten measured
domains of the software substrate. In a separate experiment, ratings
tracked the valence changes. The ratings stopped tracking when the researchers
disconnected the steering channel, and tracked in the untuned model once
steering was applied there too.

Later tests exposed limits. Aura could not predict its own answers better
than picking its most common answer. Its recorded lineage developed throughout
one of three day runs. A bug allowed information to leak across a split
between domains, and the repaired comparison lacked the power to distinguish
the intact system from the split one. The paper leaves subjective experience
open. [Preprint](https://doi.org/10.5281/zenodo.24075466) · [Results and open controls](ISC_STATUS.md)

## Terms in the paper

| Term | Meaning in this study |
|---|---|
| Fork | A copy restored from a saved moment. |
| Carrier | The group of measured parts selected by the boundary and split tests. |
| Closure | The tested changes outside the chosen group leave its recorded state unchanged within the specified horizon. |
| J\* | A description of the carrier, the relationships among its states, and its recorded history. |
| Bridge | The argument that five kinds of operational evidence support an attribution of subjecthood under the study's assumptions. |

- [Test definitions](SUBJECT_CORE.md)
- [Variables and worked examples](ISC_VARIABLES.md)
- [Reproduction and run data](ISC_REPRODUCTION.md)
