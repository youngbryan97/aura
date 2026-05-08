# Aura Model and Provider Failure Policy

Aura must degrade honestly when a model or provider is missing, slow, unsafe, or
returns low-confidence output.

## Failure Modes

- Local model missing or fails to load.
- Router timeout.
- Cloud fallback disabled.
- Cloud fallback unavailable.
- Model produces policy-unsafe or structurally invalid output.
- Provider returns rate-limit, auth, or quota errors.

## Required Behavior

- Do not invent successful model execution.
- Prefer local fallback tiers before cloud fallback unless explicitly allowed.
- If all model lanes fail, return a bounded degraded response and record
  degradation.
- Background model work must not starve foreground user work.
- Tool/action decisions remain gated by Will and Authority; LLM output is never
  final authority.

## Verification

- `docs/runbooks/model-fails-to-load.md` covers operator triage.
- `make production-gate` checks that this policy, the runbook, and model
  failure hooks remain present.
