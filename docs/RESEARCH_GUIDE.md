# Aura — Research Guide

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

For researchers reproducing Aura's claims, and reviewers running their own
evaluation and hoping to break something.

## What Aura is — and isn't

A **functional cognitive-architecture research system**. The shipped code
does not claim phenomenal consciousness, metaphysical qualia, legal
personhood, or moral patiency.

A warning about the vocabulary before you read any further. There are files
in this repo named `qualia_synthesizer.py`, `consciousness/`, `will.py`,
`soma.py`. Those are labels on mechanisms. They are not evidence, and if you
credit the system for its naming you will overrate it — which is a real risk
here, not a hypothetical one. `docs/TERMINOLOGY.md` maps every internal
poetic name to its sober technical label. Read it first and the rest of the
repo gets easier to judge.

## Architecture overview
- ``core/agency/agency_orchestrator.py`` — canonical drive→outcome loop.
- ``core/will.py`` + ``core/executive/authority_gateway.py`` — gate.
- ``core/ethics/conscience.py`` — irrevocable rule floor.
- ``core/identity/self_object.py`` — explicit "I" object.
- ``core/organism/viability.py`` — metabolic state machine.
- ``core/brain/latent_bridge.py`` — substrate→sampling modulation.
- ``core/consciousness/phi_core.py`` and ``hierarchical_phi.py`` — IIT
  4.0 spectral approximation across a 16-node and 32-node graph.
- ``core/agency/projects.py`` — long-horizon project ledger.
- ``core/social/relationship_model.py`` — durable relationship dossiers.
- ``core/sovereignty/`` — abstract wallet + migration runbook.
- ``core/embodiment/`` — permissioned world bridge + IoT bridge.
- ``core/self_modification/safe_pipeline.py`` — staged self-modification.

## Tests, baselines, ablations
- ``aura_bench/runner.py`` — pre-registered bench harness. Each test
  declares its hypothesis, metric, threshold, baseline, and ablation
  *before* running.
- ``aura_bench/courtroom/`` — 5-system, 10-task adversarial bench.
- ``aura_bench/baselines/runner.py`` — baseline-defeat matrix.
- ``aura_bench/tests/phi_uncurated.py`` (G1).
- ``aura_bench/tests/gwt_ablation.py`` (G2).
- ``aura_bench/tests/hot_calibration.py`` (G3 + H1).
- ``aura_bench/tests/qualia_behavioral.py`` (G4).
- ``aura_bench/tests/refusal_stability.py`` (I3).
- ``aura_bench/tests/continuity_30day.py`` (I1).

## Reproducibility
```bash
git clone https://github.com/youngbryan97/aura
cd aura
make setup
make test
make bench
make courtroom
make baselines
```
Outputs land at:
- ``~/.aura/data/bench/results.jsonl`` and ``report.md``
- ``aura_bench/courtroom/report.md``
- ``aura_bench/baselines/results.jsonl``

## Longevity / chaos
- ``make longevity`` (24h) — see `tools/longevity/run_gauntlet.py` for
  72h, 7d, 30d profiles.
- ``make chaos``      — single random fault.
- Each run produces an artifact bundle under
  ``~/.aura/data/longevity/<run_id>/`` with events.jsonl, resource.csv,
  identity_continuity.jsonl, and summary.md.

## Pre-registration discipline
Every benchmark refuses to record a verdict that wasn't pre-registered
through ``BenchTest.declare()``. Post-hoc "this counts because it
passed" is structurally impossible.

## Honest open items
- 30-day actual longevity run with public continuity-hash time series.
- Independent reviewers (≥3) reproducing benchmark results.
- Philosopher-of-mind consensus on the formal ontology.
- A 100,000-test suite with mutation-test scores >95%.

These are tracked as open items in the project ledger and in
``ROADMAP.md``.
