# Aura Model and Provider Failure Policy

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

When a model is missing, slow, unsafe, or returns something it isn't
confident in, Aura has to degrade **honestly**.

That word is carrying the policy. The failure mode this exists to prevent is
not the model going down — it is the runtime covering for the model going
down, serving a confident-sounding fallback, and reporting success. A visible
degradation is recoverable. A hidden one is a wrong answer with no signal
attached to it.

## Failure Modes

- Local model missing or fails to load.
- Router timeout.
- A legacy caller requests remote model inference, which is refused.
- All admitted local model lanes are unavailable.
- Model produces policy-unsafe or structurally invalid output.
- A local lane reports an admission, capacity or model-identity failure.

## Required Behavior

- Do not invent successful model execution.
- Use admitted local fallback tiers only. Remote model routing is retired;
  a compatibility field cannot authorise a cloud-model fallback.
- If all model lanes fail, return a bounded degraded response and record
  degradation.
- Background model work must not starve foreground user work.
- Tool/action decisions remain gated by Will and Authority; LLM output is never
  final authority.

## Verification

- `docs/runbooks/model-fails-to-load.md` covers operator triage.
- `make production-gate` checks that this policy, the runbook, and model
  failure hooks remain present.
