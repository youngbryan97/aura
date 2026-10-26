# Artifact integrity

**Status:** live. `core/runtime/artifact_integrity.py`, used by
`ImmuneSystem.initiate_rollback`.

## Upstream

| | |
|---|---|
| Project | The Update Framework (TUF), SLSA |
| License | TUF reference impl: MIT/Apache-2.0. SLSA: open specification. |
| Adopted | The verify-before-install principle only |
| Code copied | **None.** No TUF client, no Sigstore, no key infrastructure. |

## What was deliberately NOT built

A full TUF client — root/targets/snapshot/timestamp roles, threshold signing,
metadata expiry, rollback and freeze protection. **Aura has no self-updater.**
Building a key hierarchy for a distribution path that does not exist would be
speculative machinery, and the standing rule here is that everything must be
live at runtime.

If and when Aura ships a self-updating binary, this is the module that grows a
`SIGNED` level. The enum already names it as unimplemented so the gap is
visible rather than assumed away.

## What was built, and why it is live

TUF's *first* principle applies today: several paths promote a file into
executable Python — emergency rollback, code repair, kernel refinement, sandbox
promotion. That is the highest-consequence write in the system.

Four checks, cheapest disqualifier first:

1. **Containment** by resolved path components. `str(p).startswith(str(base))`
   accepts any sibling whose name merely begins with the base —
   `data/backups_evil` passes a `data/backups` prefix test.
2. **Regular file**, resolved strictly and checked at the resolved path.
   This does not make a later path read race-free; the verifier returns a
   path, and promotion must preserve the identity of the verified bytes.
3. **Digest** against a manifest whose **absence is a refusal**. This is a
   digest comparison, not signature verification or publisher authentication.
4. **Parses** as Python. Restoring a syntactically broken file leaves the target
   unimportable — bricking the thing the promotion was meant to rescue.

## Honesty about the guarantee

`IntegrityLevel.DIGEST` is what a passing verdict carries: the bytes are the
bytes that were recorded. It does **not** prove a trusted party recorded them.
The level is on the verdict so a caller cannot mistake integrity for
authenticity — which would be exactly the overclaim this module exists to
prevent.

## Consolidation

These checks were written inside `ImmuneSystem.initiate_rollback` first. They
now live in one place and that module calls them, so its bespoke
`_snapshot_integrity_ok` is gone. A test asserts the duplicate stays gone.

## Conformance tests

`tests/test_artifact_integrity.py` — 13 tests, plus the 9 pre-existing
immune-system regressions which still pass unchanged against the shared gate.

One test earned its keep: *"a gate that can itself explode is not a gate"*
caught the first implementation letting `ValueError` escape from `resolve()` on
a path containing a null byte.

## Known unsupported

- No signature verification (see above).
- Only the immune-system rollback calls it. `code_repair.py` and
  `kernel_refiner.py` promote files by other routes and have not been migrated.
