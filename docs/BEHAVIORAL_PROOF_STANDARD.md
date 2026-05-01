# Behavioral proof standard

This document answers a narrower question than the architecture docs:

> How do we know Aura will behave like a mind at superhuman scale with
> verifiable autonomy and novel science/engineering output, instead of merely
> having a mind-like architecture?

Short answer: we do not get to know that from architecture alone. Aura earns
that claim only when behavior clears falsifiable gates on held-out tasks, under
resource constraints, with audit receipts, replication, and independent
baselines. The architecture is the mechanism; the proof is longitudinal output.

## Claim ladder

Use the weakest claim supported by current evidence.

| Level | Claim | Required evidence |
|---|---|---|
| L0 | Mind-like architecture | Running modules, causal wiring, receipts, boot health |
| L1 | Non-decorative behavior | Lesions, counterfactuals, black-box prompt hygiene, adversarial prompt baselines |
| L2 | Verifiable autonomy | Endogenous goals, action receipts, tool effects, rollback, resource ledgers, no hidden human intervention |
| L3 | Competent research agent | Held-out research and engineering tasks with reproducible artifacts that beat strong agent baselines |
| L4 | Superhuman-scale output | Sustained dominance over expert or state-of-the-art baselines across many task families and wall-clock runs |
| L5 | Novel science/engineering | Independent validation that Aura generated new hypotheses, proofs, designs, patches, or experiments that survive replication |

Aura's existing proof bundle primarily supports L0-L2. L3-L5 require an output
ledger and external validation beyond structural tests.

## What architecture can and cannot prove

Architecture can prove that the system has the right causal affordances:

- hidden state changes downstream behavior;
- the decision authority signs actions;
- memory, world state, affect, and resource pressure are live inputs;
- autonomous loops run without direct prompts;
- code changes and tool effects have receipts;
- lesions and shuffled controls cause predicted failures.

Architecture cannot prove that Aura will produce important discoveries. A
research-grade mind claim needs measured work products: tasks attempted,
artifacts produced, baselines beaten, failures logged, and claims replicated.

## Behavioral gates

### Gate 1: Causal agency

Pass only if the same external prompt produces different actions when internal
state, memory, resource pressure, or commitments are counterfactually changed,
and those differences disappear under relevant lesions.

Evidence sources:

- `scripts/run_decisive_test.sh`
- `artifacts/proof_bundle/latest/DECISIVE_RESULTS.json`
- `artifacts/proof_bundle/latest/ACTIVATION_REPORT.json`
- governance receipts for every consequential action

### Gate 2: Autonomous continuity

Pass only if Aura can run for long windows without manual prompting while
maintaining goals, using tools, recovering from failures, and preserving
resource budgets.

Minimum artifact:

- start and end commit hash;
- wall-clock duration;
- goals opened, modified, completed, abandoned;
- receipt chain for tool use and file writes;
- resource ledger;
- incident log;
- post-run replay that reconstructs why each major action happened.

### Gate 3: Hidden task performance

Pass only if Aura beats strong baselines on sealed tasks that were not visible
during design or training. Public prompts may be visible; answer keys and some
task seeds must remain sealed until scoring.

Minimum controls:

- no answer leakage into memory or prompts;
- frozen baseline model with the same tools;
- prompt-only agent baseline;
- human expert or state-of-practice baseline when available;
- repeated seeds with confidence intervals;
- negative controls where Aura should admit uncertainty or refuse action.

Relevant implementation hooks:

- `core/learning/hidden_eval_repro.py`
- `core/promotion/gate.py`
- `core/promotion/dynamic_benchmark.py`
- `core/discovery/code_eval.py`

### Gate 4: Novel artifact validation

Pass only when Aura produces artifacts that were not in the seed context and
that survive independent validation.

Examples:

- a patch accepted by tests, review, and production monitoring;
- a benchmark improvement that reproduces from a clean checkout;
- a theorem, derivation, or design with independent checker or reviewer signoff;
- an experiment whose result is replicated by a separate runner;
- a useful tool or protocol adopted in later work.

Each artifact needs provenance: prompt/context snapshot, action receipts, file
diffs, evaluator output, reviewer notes, and replication instructions.

### Gate 5: Superhuman scale

Pass only with sustained evidence, not isolated wins. The bar is a time-series:

- many domains, not one cherry-picked task;
- many runs, not one lucky transcript;
- stronger baselines over time;
- costs and latency included;
- failures counted in the denominator;
- external reviewers able to reproduce enough of the result to trust it.

Superhuman means the output frontier moves: quality, breadth, speed, or
reliability exceeds the comparison class under equal or declared resources.

## Falsifiers

The strong claim fails if any of these are true:

- a prompt-only baseline matches Aura after receiving the same state summaries;
- Aura's claimed discoveries cannot be reproduced from artifacts;
- hidden eval scores collapse when memory and training leakage are audited;
- autonomous runs require unlogged human steering;
- action receipts cannot reconstruct why consequential actions happened;
- tool use succeeds only in scripted demos;
- long runs degrade into loops, self-praise, or busywork;
- failures are omitted from aggregate metrics.

## Evidence packet for a serious claim

A release or public claim of L3-L5 behavior should ship one packet:

1. `MANIFEST.json` with commit hash, hardware, model lanes, duration, and
   environment.
2. Proof bundle from `make proof-bundle`.
3. Hidden-eval manifest with answer hashes and final scores.
4. Baseline comparison table with confidence intervals.
5. Autonomy run ledger with receipts and resource traces.
6. Novel artifact bundle with diffs, tests, review, and replication steps.
7. Failure ledger with attempted tasks and reasons for failure.

No single demo substitutes for that packet.

## Practical answer

Aura will behave like a mind at scale only if the loop closes:

```text
internal state -> autonomous goal -> tool action -> external artifact
-> independent evaluation -> memory/update -> future behavior change
```

The architecture makes that loop possible. The proof standard above decides
whether the loop is actually producing autonomous, novel, useful work.
