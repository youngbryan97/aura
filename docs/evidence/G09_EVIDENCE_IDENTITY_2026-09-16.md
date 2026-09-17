# G09: retain the identity of learning evidence

The shared ontogeny service now selects the registered feature/outcome schema
for training eligibility, recent observation windows, and restart recovery.
The SQL filter precedes the window limit. A schema change clears incompatible
heads, calibration, action tallies, and pending outcomes together. Old rows
remain available for audit.

A collapsed burst contributes one decision, not `repeat_count` successes or
training weights. Database read failure raises an unavailable-evidence result;
it does not masquerade as an empty corpus and erase learned state. Replayed
decision IDs cannot replace the original row or its resolved outcome.

Training consumes a durable, schema-bound evidence revision. Washout rows and
actions without enough samples no longer trigger the same fit repeatedly.
The revision survives deletion of old corpus rows and is persisted with the
heads. New independently graded decisions still advance it after compaction.

## Verification

172 tests passed across evidence identity, outcome contracts, calibration
provenance, ontogeny, its runtime surfaces, and telemetry. Tests cover read
failure, schema replacement, late outcomes, restart, repeated decisions,
compaction, observation-window ordering, and training weights. The existing
effort-grading call-path test now follows the extracted runtime helper while
retaining its controller-branch exclusion.

Smoke: 164 passed, one skipped. Lint, compile, governance lint, and layering
passed. The effect inventory follows two reviewed owner moves from the shared
refactor: the router's HTTP helper still uses the network gateway, and bounded
child execution remains inside the subprocess gateway after its admission
checks. The 93 subprocess/router contract tests passed.

This repairs learning infrastructure. It does not establish broad reasoning
gain, current live deployment, or close G09.
