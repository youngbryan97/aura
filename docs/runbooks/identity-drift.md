# Runbook: identity drift

<!-- reader-navigation:start -->
> Aura checks whether parts are making progress, keeps state that can be restored, and has procedures for recovery.
>
> A mistaken restart can interrupt work or leave two large processes competing for memory. If a reply takes a long time, first check whether its worker is still making progress. A stalled worker needs a different response.
>
> [Plain-language guide](../PLAIN_ENGLISH_GUIDE.md#reliability) · [Terms and examples](../VARIABLES_EXPLAINED.md#reliability) · [Evidence and related work](../DEMOS_AND_RELATED_WORK.md#runtime)
> [All docs](../DOCUMENTATION_INDEX.md) · [Glossary](../GLOSSARY.md) · [Worked formulas](../FORMULAS.md)
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
