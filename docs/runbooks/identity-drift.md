# Runbook: identity drift

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
<!-- reader-navigation:end -->

Covers [F10](../../KNOWN_FAILURE_MODES.md).

## Symptoms
- Personality inconsistent across turns in a way that persists.
- Identity coherence check failing.
- `CanonicalSelf` hash mismatch.

## Diagnosis
1. Compare the current `CanonicalSelf` hash against the canonical snapshot
   (`core/self/canonical_self.py`, seeded from
   `core/constitution/canonical_self.json`).
2. `aura verify-state` for cross-subsystem coherence.
3. Check whether the run involved sustained adversarial prompting. That is
   the known cause; ordinary long conversations are not.

Distinguish two things before acting. A **continuity-hash change with
self-relevant edits** is correct behaviour — the hash is a pure function of
self-relevant fields. Drift is the hash moving *without* such a change.

## Safe mitigation
- Reset `CanonicalSelf` from the canonical snapshot.
- Restart to reload identity cleanly.

## Unsafe mitigation (last resort)
- Do not hand-edit the canonical snapshot to match current state. That
  ratifies the drift as the new baseline and destroys the reference you
  detect future drift against.

## Rollback
Restore identity state from backup; verify the continuity hash after.

## Verification
- Continuity hash stable across two consecutive snapshots with no
  self-relevant change (the ontology's continuity-preservation axiom).
- `pytest tests/personhood/test_self_object.py -q`

## Postmortem checklist
- Record what moved the hash. An unexplained identity change is a security
  question, not just a quality one.
