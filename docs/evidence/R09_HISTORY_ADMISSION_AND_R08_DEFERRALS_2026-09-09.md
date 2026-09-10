# History admission and tool deferrals

## Live observations

The supported UI reboot replaced PID 20867 with PID 34020. Boot provenance
reported commit `3feb110b03e2dfb28453fb043e4b3388050be0d2`, matching expected
and actual workspace hashes, `source_current=true`, and no source issues.
The workspace includes preserved concurrent edits; this is not a clean-tree
release claim.

A fresh browser window eventually restored the three previous exchanges,
with their original timestamps and distinct identities. During startup the
durable reader repeatedly exceeded its 1.5-second wait and returned an empty
history. Reading the same SQLite store with the production synchronous reader
from a separate process took 0.010153 seconds. That measurement excludes
runtime queueing and event-loop contention.

Two deterministic tests occupying the default executor reproduced the empty
history for owner and paired-device requests. The reader now uses the existing
foreground durable-receipt executor, preserving request context variables.
The timeout and principal/session visibility rules are unchanged. All 27
history, conversation persistence, and executor lifecycle tests pass.

At 19:08 the exact database follow-up that previously received the unrelated
provenance template reached cognition and returned a complete 927-character
answer. Worker measurements: prefill 1,894 tokens in 9.78 seconds; decode 167
tokens in 16.77 seconds; peak 15.96 GB. Its explanation remains imprecise about
recovery algorithms; this is routing evidence, not a general correctness pass.

At 19:09 a genuine source question reached cognition but referred to the older
Solaris discussion rather than the preceding database answer. Its durable
receipt records `cognitive_engine`, no tool receipts, and a grounding digest.
This wrong antecedent remains an R09 failure. No whole R item is closed here.

## Deferred work

The neural stream's earlier `Scan stalled: unknown error` followed an
`auto_refactor` admission deferral during foreground generation. The
self-development owner treated it as an execution failure. A deferred sandbox
test could also become fabricated failure feedback for a downstream proposal.

The existing tool-result contract now recognizes all shipped deferral shapes.
Constitutional bookkeeping, developer traces, and the self-development owner
use that classification. A deferred stage ends the dependent chain; the
existing periodic owner retries on its next eligible pass. Compact results
retain reason and retry metadata. Actual execution failures retain their
failure behavior. No admission policy was relaxed.

The combined focused gate passed 92 tests on rerun. An earlier randomized
batch failed `test_executive_temporal_anchor_prefers_actionable_pending_work`;
the isolated test, full constitutional suite, and repeated combined gate
passed. The order-sensitive failure is not diagnosed or dismissed as a flake.

## Remaining proof

- Deploy the reserved history reader and verify restoration during startup.
- Observe deferred self-development on the updated resident runtime.
- Trace and repair the wrong antecedent in the genuine source follow-up.
- Reproduce the randomized temporal-anchor failure with retained ordering.
- Complete the remaining R09 acceptance matrix before marking it closed.
