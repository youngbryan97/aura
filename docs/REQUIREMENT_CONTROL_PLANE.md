# Requirement-to-Proof Control Plane (SCOPE-001 / PROGRESS-CONTROL-001)

The machine-enforced system that prevents forgotten obligations, false
completion, stale proof, percentage inflation, checkpoint drift, and release
with open mandatory work. Landed 2026-07-18 (commit `b67ba49f`); this
document is the operating manual. The exact remaining-work contract lives in
[REQPROOF_SESSION_HANDOFF.json](REQPROOF_SESSION_HANDOFF.json).

## What is canonical

| Artifact | Role |
|---|---|
| `config/requirement_registry.json` | The requirement denominator: 302 requirements generated deterministically from the tracker's normative extraction. Tamper-evident (content hash), never hand-edited. |
| `config/requirement_coverage_map.json` | 137 hash-pinned passage→requirement mappings covering every non-blank line of all four source corpora. |
| `config/requirement_proof_specs.json` | Content-hashed, reviewable argv-only proof specifications. Each spec names exact source files and acceptance/evidence cells; shell and inline-code commands are rejected. |
| `config/requirement_sources/` | Verbatim snapshots of the four authoritative corpora, provenance-hashed in `MANIFEST.json`. |
| `config/reqproof_defect_baseline.json` | Shrink-only fingerprint ratchet pinning pre-existing tracker debt (58 at seed). |
| `config/reqproof_prose_token_allowlist.json` | Reviewed ID-like tokens that are deliberately not requirements (FMEA keys, defect-class names). |
| `artifacts/reqproof/GATE_REPORT.json` | Deterministic gate report (no timestamps; byte-identical for identical state). |
| `artifacts/reqproof/DOCKET_REPORT.json` | Deterministic current docket joining every requirement to verified evidence, direct dependency blockers, closure blockers, and missing acceptance/evidence cells. |
| `artifacts/reqproof/evidence/<proof-id>/<source-commit>.json` | Immutable command receipts captured only from a clean checkout whose `HEAD` exactly equals pushed `origin/main`. |
| `docs/AURA_REMAINING_DOCKET.md` | Human-readable generated view of active work and historical completion claims that still need evidence. |

## Commands

```bash
make reqproof-gate       # structural gate — required green at every commit
make reqproof-release    # release gate — blocks until zero open mandatory scope
make reqproof-docket     # regenerate the current dependency-aware docket
make reqproof-capture SPEC=<checked-proof-id>  # run and record one checked proof
make reqproof-capture SPECS='<proof-id> <proof-id>'  # all pass before batch recording
python tools/reqproof/migrate.py --write     # regenerate registry from tracker
python tools/reqproof/gate.py --refresh-baseline  # shrink-only ratchet refresh
```

The structural gate also runs inside `tools/release_preflight.py` as
`reqproof_structural` (pinned by `tests/test_preflight_gate.py`).

## Workflow rules (enforced, not advisory)

1. **Tracker edits**: any change to a normative region (requirement tables,
   Pass F/Matrix/Addendum/carryover items) requires
   `python tools/reqproof/migrate.py --write` **in the same commit** —
   `TestRealRepositoryGate` compares the registry against the tracker at
   HEAD and fails otherwise. Narrative/checkpoint prose edits do not require
   re-migration (the extraction hash covers normative content only).
2. **Closure**: a requirement reaches `complete` truthfully only when every
   class in its `evidence_required` has a verifiable `EvidenceRef`
   (existing path + matching sha256 + commit known to the repo) AND every
   member of `closure_requires` is itself closed. Anything else surfaces as
   `unproven-closure` / `false-closure` and, if new, fails the gate.
3. **New defects never enter silently**: ratcheted classes are pinned by
   exact fingerprint; a new fingerprint fails the gate. Fixing a defect
   makes the baseline stale, and `--refresh-baseline` refuses to grow.
4. **Zero-unmapped is standing**: dropping or editing corpus text under the
   map fails (`stale-coverage`); an uncovered passage fails
   (`unmapped-passage`). New corpora must be added to `MANIFEST.json` and
   fully mapped before the gate passes.
5. **No invented numbers**: the gate publishes raw counts only. There is no
   completion percentage until the evidence-weighted progress engine exists
   (Session 2); the tracker's manual estimate is explicitly superseded by
   `summary.mandatory_not_closed` (281 at seed) as the honest denominator.
6. **Proof capture is source-bound**: `tools/reqproof/capture.py` rejects a
   dirty checkout, a local commit not equal to `origin/main`, shell/inline-code
   specifications, undeclared accelerator use, timeouts, non-zero exits,
   oversized output, source mutation, and invalid acceptance cells. It records
   the complete output and a source-file hash manifest. Failed capture leaves
   neither a receipt nor a ledger entry. The capture tool/spec must therefore
   be pushed first; evidence generated from that clean commit lands in a
   subsequent checkpoint.
7. **Evidence expires when its subject changes**: every external-ledger
   artifact must be a structured passing receipt whose source commit and
   acceptance targets exactly cover the ledger entry. Its non-empty canonical
   source manifest is rehashed against current `HEAD` whenever closure,
   progress, or the docket is computed. Version-2 receipts also retain their
   exact path/glob selectors and re-expand them, so a newly added matching file
   invalidates the old proof rather than escaping its manifest. Missing,
   symlinked, resized, changed, newly matched, or no-longer-matched source
   produces a blocking `stale-evidence` defect and the affected cells receive
   zero credit until a fresh proof lands. Arbitrary JSON, logs, prose, and
   manually named files cannot enter the external ledger.

## Defect taxonomy

Blocking always: `duplicate-id`, `orphan-ref`, `parent-mismatch`,
`closure-cycle`, `impossible-evidence`, `stale-migration`,
`prose-only-token`, `missing-corpus`, `stale-coverage`,
`coverage-orphan-ref`, `unmapped-passage`.

Ratcheted (pinned baseline, shrink-only): `dependency-cycle`,
`false-closure`, `unproven-closure`, `contradictory-status`,
`withdrawn-required`, `prose-minted`.

Baseline debt at seed (2026-07-18): 23 dependency cycles among master rows,
19 migrated `complete` rows with no machine evidence, 11 requirements that
existed only as prose references (now minted open rows: `AUTONOMY-001`,
`CHAT-001`, `CLOSEOUT-001`, `GUI-001`, `OBSERVABILITY-001`, `PRIVACY-001`,
`RELEASE-001`, `RESOURCE-OWNERSHIP-001`, `RLC-MECH-006`,
`SEMANTIC-REVIEW-001`, `VOICE-001`), 3 contradictory statuses, 2 false
closures. Draining this baseline is tracked work, not background noise.

## Honest boundaries (non-claims)

* A structural pass asserts registry/coverage integrity only — it is not
  evidence that any requirement's engineering work is done.
* A docket `ready_for_implementation` disposition means only that the
  requirement has no directly open dependency. It does not waive closure
  children, required evidence classes, or live/release obligations.
* Migrated statuses restate what the tracker asserted; the 19
  `unproven-closure` defects mark exactly where those assertions lack
  machine evidence.
* The progress report inventories checkpoint history and produces an explicitly
  provisional forecast. The docket provides the canonical current work view;
  neither replaces acceptance-granular evidence or the release gate.
* A command receipt proves only the evidence cells named by its checked spec.
  A passing unit/contract suite is not live, release, soak, portability, or
  implementation evidence unless an independent checked proof establishes that
  distinct class.
* Source freshness is selector-plus-path-manifest freshness, not a claim that
  every possible transitive dependency was selected correctly. Proof
  specifications must name exact files or reviewable repo-relative globs that
  cover the production and harness surface causally material to their claim.
