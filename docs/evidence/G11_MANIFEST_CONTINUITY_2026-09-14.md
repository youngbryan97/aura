# G11: component-scoped manifest continuity

The resident manifest changed from `0d893b98ba7089708ea474be9364855326d077c724684b219560a175d86eb385`
to `d96503910e4779346a5bae714f4f94975b2974420c796c66bead3aef9d9faf15`.
The original bytes still existed in `training/fused-model/active.json.staged`.
Their digest matches the retained 27B qualification exactly.

The structured difference is confined to the signed steering component and
the containing migration contract's timestamp and digest. The model artifact,
serving profile, evaluation, persona, recurrence authority, and all other
manifest fields are unchanged. The exact semantic ingress runs on CPU without
constructing an MLX client and serializes authenticated semantic state. It does
not consume steering hooks.

## Repair

The central upgrade writer now retains content-addressed, write-once pointer
snapshots. A shared continuity comparator checks the complete predecessor
against the current manifest, permitting only the consumer's explicitly named
independent components. Its default permits none. This semantic consumer names
only steering; changed recurrence, profiles, model artifacts, evaluation, and
unknown fields still fail. Current component signatures and evidence are
reopened. The original qualification and its recorded resident identity remain
unchanged. The current serving receipt carries both manifest hashes and states
that this is not a comparative remeasurement.

A replaced-weight regression exposed a further gap: ordinary descriptor
validation checked the descriptor's structure but did not rehash files. The
continuity path now validates complete artifact bytes, caching only against
the full file inventory including inode, size, mtime and ctime. Same-size writes
with restored mtime invalidate it. First verification on the real 27B took
8.238 seconds. Boot preparation and chat evidence selection perform this I/O
off the event loop.

## Evidence

- 45 tests passed across manifest continuity and cortex upgrade lifecycle.
- 65 passed and one source-drift diagnostic skipped across serving, ingress,
  continuity, chat evidence ownership, and the installed bounded-WOW alarm.
- After stable-read and off-loop additions, 25 focused tests passed.
- The current installation returned `semantic_neural_serving_active`; the
  active pointer was byte-for-byte unchanged by retaining the predecessor.
- [Runtime replay](../../artifacts/migration/27b/recovery/manifest-continuity-20260914/runtime.json):
  seed 2026091461, 120/120 exact through foreground and service, 120/120
  coefficient-lesion disruptions, 120/120 backend-receipt equivalences.
  Thirty tasks per domain; ten per scientific surface. Mean 23.575 ms,
  median 22.562 ms, maximum 39.039 ms after verification warmup.

This is an in-process replay through the production foreground/service
functions, not a desktop HTTP interaction. G11 still requires desktop proof
on a source-matched deployment. G03-G12 remain open where their independent
requirements have not passed.

## Concurrent integration

The primary checkout contained another uncommitted proposal for the same
manifest failure. That proposal compares model-basis fields but does not
compare the old and new complete serving profiles or recurrence authorities.
Use the content-addressed comparison above for this repair. The separate
primary-checkout change that refreshes cortex registry evidence off-loop is
independent and should be preserved during deployment integration.
