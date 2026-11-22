# Ablation legibility — reviewer-runnable subsystem deltas

External reviewers ask one thing above all else:

> Run Aura baseline vs *without-memory / without-Will / without-substrate /
> without-reasoning-amplifier / without-verifier / without-System-2* and see
> **measurable deltas.**

This is that, in one command. Each condition drives a **real organ** (no mocks,
no clamps, no hardcoded statistics) intact vs lesioned and reports the measured
metric for both, the delta, and an honest load-bearing verdict. A no-delta
result is reported as *not* load-bearing — never hidden.

## Read this first: "load-bearing" is not "helps"

Every condition in `ablation_runner.py` is a **wiring check**, and the scorecard
says so now rather than leaving it to be noticed.

`without_system2` scores 1.000 → 0.000 on `strict_proof_exact_answer_rate`, and
the lesioned component *is* the strict-proof solver. Removing the only thing
that can emit an answer in that format, and observing that the format stops
appearing, is a true statement about wiring. The number is large, real,
correctly measured and inferentially almost empty — the worst combination
available, because size reads as strength. `without_verifier` (the verifier is
the only rejector) and `without_substrate` (policy divergence is the coupler's
own output) have the same shape.

`core/evaluation/lesion_inference.py` classifies what each delta licenses:

| class | what it establishes |
| :--- | :--- |
| `tautological` | the component is wired to the metric it produces |
| `mechanistic` | the component changes its own output — real, one layer short |
| `capability` | it changes TASK SUCCESS, on tasks solvable **without** it |

The current scorecard reads: *0 of 3 conditions measure task success on tasks
solvable without the component.* It previously read `all_conditions_load_bearing:
true`, which is a different sentence and looks like the same one.

## The capability runner

For the comparison that can support an earns-its-cost claim:

```bash
.venv/bin/python tools/capability_ablation.py \
    --responder mlx --model <local-model> --history-turns 40
```

Three arms at an **identical turn budget** — `stateless` (final turn only),
`long_context` (the most recent N turns, what every chat client sends), and
`full_architecture` (the N most *relevant* turns, via Aura's real retrieval).
Beating `stateless` is nearly definitional; beating `long_context` is the claim.

Measured 2026-08-06, 40 tasks, one local model, paired bootstrap:

- history **exceeds** the window: long_context 0.000, retrieval 1.000,
  delta **+1.000**, 95% CI [1.000, 1.000]
- history **fits** the window: delta **0.000**, reported `unresolved`

Both are committed. The second matters as much as the first — the advantage
exists where history exceeds what a caller can afford to send, and nowhere else.
A delta whose confidence interval spans zero is reported as `unresolved`, never
as a finding, and arms whose budgets differ return `void` rather than a number.

## One command

```bash
.venv/bin/python tools/ablation_runner.py
```

Writes `artifacts/ablation/ablation_scorecard.json` and prints:

```
condition             metric delta    baseline→ablated      load-bearing
--------------------------------------------------------------------------
without_substrate     0.0850          0.085 → 0.000         True
without_system2       1.0000          1.000 → 0.000         True
without_verifier      0.5000          1.000 → 0.500         True
```

`--list` shows every condition; `--conditions substrate,verifier` runs a subset.
Exit code is 0 iff every executed condition is load-bearing.

## What each condition actually does

| Condition | Real organ | Metric | Method |
|-----------|-----------|--------|--------|
| `without_substrate` | `core.being.policy_coupler.ClosedLoopPolicyCoupler` | policy divergence across 7 causal self-states (mean pairwise normalized L1) | intact coupler vs the same organ blinded to a neutral state |
| `without_system2` | `core.reasoning.proof_answer_solver` | strict-proof exact-answer rate | real symbolic solver vs no deterministic derivation path |
| `without_verifier` | `…proof_answer_solver.validate_strict_proof_answer` | balanced accuracy (accept correct + reject wrong) | real validator on (correct, wrong) pairs vs a null verifier that accepts everything |

The `without_substrate` metric is exactly the felt-state → action-policy claim:
uncertainty, trust debt, resource pressure, governance pressure, identity
tension and workspace coherence measurably change temperature, verification
threshold, tool-risk budget, planning depth and memory retrieval depth. Blind
the coupler to that state and every context collapses to one identical policy
(distinct policies 7 → 1, divergence → 0).

## Context conditions — honest deferral, not a null

`will`, `memory`, and `reasoning_amplifier` only exercise their organ with
runtime or model context. A bare offline harness starves the Will of its
stakes/permission context and starves memory/amplifier of the model, so a flat
offline "0" would be *misleading theater*, not evidence. Each therefore points
to the dedicated runner that sets the context up honestly:

- **will** → `tools/agi/run_will_governance_ablation.py` — real `UnifiedWill`
  decisions vs a neutral governance packet, on a survival/critical/initiative
  benchmark.
- **memory** → a retention battery against `/api/chat` on the running kernel
  (plant a fact, recall it after a gap; grounded recall vs no-memory).
- **reasoning_amplifier** → `ReasoningAmplifierV2(generate=…)` with the live
  model: verifier-filtered self-consistency vs single-sample.
- **causal agency** → `tools/agi/run_causal_agency_lesion.py` — the same
  substrate coupling proven with a permutation p-value and receipt coverage.

## Honesty guarantees

- Every number is measured from the real organ at run time.
- No clamps, floors, or hardcoded statistics (the causal-agency runner was
  rewritten in July 2026 to remove exactly those; see its module docstring).
- A condition that shows no delta is reported `load_bearing: false`. That is a
  real, publishable result — it means the subsystem is not load-bearing on that
  battery, and the battery/claim should be revised, not the number.
