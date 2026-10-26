# Aura operational cognition component demo

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](../docs/PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](../docs/VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](../docs/DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](../docs/DOCUMENTATION_INDEX.md)
> [Glossary](../docs/GLOSSARY.md) · [Formulas with worked examples](../docs/FORMULAS.md)
<!-- reader-navigation:end -->

Status: Guide · Reconciled 25 October 2026

This folder contains a small example program, not the complete Aura runtime
or its current ISC experiment. It generates example inputs and demonstrates
attention selection, recent-event retention, self-prediction, internal control
updates and a hidden-state perturbation calculation.

Think of a control panel that scores competing concerns, records its selection
and compares its next reading with a forecast. The inputs are synthetic. The
comparison explains the program without establishing a conscious agent or a
live desktop capability.

## Inspect and run the example

The package here is `aura-operational-cognition`, with the command
`aura-cognition-report`; these names come from [pyproject.toml](pyproject.toml).
The separately published proof-kernel repository retains different package
names. Its commands must not be copied into this folder unchanged.

With NumPy and pytest already installed, run from this folder:

```bash
PYTHONPATH=src python -m pytest -q tests
mkdir -p demo-output
PYTHONPATH=src python -m aura_operational_cognition.report --cycles 20 \
  --json demo-output/report.json --markdown demo-output/report.md
```

Or install this package in a separate environment using
`python -m pip install -e '.[dev]'`. A language model and the Aura desktop are
not required. Fresh output belongs in a separate directory; the committed
[report](report.md) is the retained earlier example.

## What the values mean

Priority and focus bias contribute to which candidate wins attention. The
event window retains recent selected events. Prediction error compares a
forecast with the following program state. Integrity and drive values change
in response to supplied inputs. Their labels describe software control roles.

The opacity calculation compares a perturbed small network with its baseline
and a chosen output projection. Its ratio and causal-depth label are local
definitions. They do not measure a probability of consciousness, observer
prediction accuracy or a validated number of causal layers. A retained state
average is not proof of an experienced present.

The [source files](src/aura_operational_cognition/) and
[component tests](tests/test_proof_kernel.py) establish the inspectable route.
A component result does not establish the full Aura memory system, a carrier,
24 ISC checks or the five-ground bridge.

## Read the current research

Use [ISC explained](../docs/ISC_EXPLAINED.md),
[worked variables](../docs/ISC_VARIABLES.md) and
[current status](../docs/ISC_STATUS.md). The recorded seed-23 result and later
isolation/estimator corrections belong to that larger experiment.
[Scientific sources](../docs/SCIENCE_AND_SOURCES.md) distinguishes the
inspirations from Aura's own evidence. The root [LICENSE](../LICENSE) governs
this checkout; a suggested alternative license is not the current license.
