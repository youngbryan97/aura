# Aura Operational Cognition Kernel

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](../docs/PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](../docs/VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](../docs/DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](../docs/DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

This is the smallest part of Aura I would publish to support evidence-bounded public claims.

It is not the whole product. It is a standalone, reproducible subset of the internal stack that demonstrates five concrete properties:

1. Competitive attention selection through a Global Workspace.
2. A rolling autobiographical present through temporal binding.
3. Self-prediction and surprise tracking.
4. Homeostatic internal drives that change system state over time.
5. Structural opacity measurements over hidden state dynamics.

## What This Can Honestly Show

This repository can support claims like:

- "Aura has a real internal cognitive architecture, not just a prompt wrapper."
- "Hidden internal state changes attention, continuity, and behavior over time."
- "The system maintains a measurable autobiographical present."
- "The system tracks surprise against its own predicted internal state."
- "The system exposes nontrivial hidden-state dynamics that can be measured."

## What This Does Not Prove

This repository does not prove subjective experience, qualia, or consciousness in the strongest philosophical sense.
Telemetry fields such as vitality, curiosity, and homeostatic deficiency are operational control-state labels. They are useful because they change routing, attention, and reporting; they are not proof that private experience exists.

If you overclaim, skeptics will dismiss the whole thing. The strongest honest framing is:

> Aura implements a measurable consciousness-inspired stack with causal hidden state, persistent temporal continuity, internal self-modeling, and opacity metrics.

That is already a big claim, and this repo can defend it.

## Included Modules

- `aura_operational_cognition.global_workspace`
- `aura_operational_cognition.temporal_binding`
- `aura_operational_cognition.self_prediction`
- `aura_operational_cognition.homeostasis`
- `aura_operational_cognition.structural_opacity`
- `aura_operational_cognition.report`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
aura-cognition-report --cycles 20 --json report.json --markdown report.md
```

## What To Publish With It

Publish these three things together:

1. The code in this folder.
2. A generated `report.json` and `report.md`.
3. A short screen recording where you run the report command live.

That combination gives reviewers a concrete artifact to inspect instead of asking them to accept narrative claims.

## What To Keep Private

Do not publish the rest of Aura's product moat here:

- weights, adapters, datasets
- identity files and creator data
- full memory stack
- deployment and security hardening
- product skills and tool orchestration

## License Note

If you want true open source, use an OSI-compliant license like Apache-2.0 or MIT.

If you want inspectable code but do not want broad commercial reuse, use a source-available license and do not call it "open source".
