# R09 dialogue budget custody

The source-history repair did not remove a second lossy step in
`InferenceGate._compact_prebuilt_messages`. That allocator clipped retained
answers individually, allowed its count window to start at an answer, and
removed old answers before removing their questions under budget pressure.

The allocator now retains whole dialogue content, rounds a window boundary
back to the initiating question, and evicts the oldest complete exchange
when the aggregate budget requires it. The total character budget is unchanged.
Current-request and runtime-evidence allocation retain their existing rules.

Three new regression cases failed before the repair: an orphaned window,
a code answer clipped despite fitting the budget, and unpaired eviction.
The complete custody, prefix-cache, and inference-tiering selection passed
183 tests in 198.41 seconds. Two older count assertions now require complete
pairs plus the current question rather than permitting a leading answer.

Smoke passed 164 tests with one skipped. Lint, compilation, governance, and
layering passed. Layering first found an inherited subject-state import of
the motivation budget constants without a declaration; its exact read-only
module dependency is now declared. The layering baseline remains 37.

## Live observation on the preceding revision

Runtime PID 77385 was launched from commit 608d7859a. The allocation repair
above was not deployed for this observation. Concurrent shared-checkout edits
changed the workspace fingerprint after boot; this is not a frozen-source
campaign result.

Turn `aura-chat-c46921b2-db19-4af6-b978-88262b78dd9b` asked why database
recovery needs undo and redo. The worker measured 1325 prefill tokens in
15.80 seconds and 1124 decode tokens in 125.66 seconds. The public answer
was 1632 characters. The terminal timing receipt measured 193893.74 ms total.

A browser window opened while that turn was running restored the question
but did not add the completed answer. Its actual neural feed reported
"Answer delivered" while the transcript still contained only the question.
Reload then restored the complete answer. The answer distinguished atomicity
and durability and described ARIES repeating history before undo, although
its earlier restriction of redo to committed transactions was too narrow.

The UI cause is an empty-transcript-only hydration rule. A restored pending
question makes the transcript nonempty, so later history cannot supply its
answer. This requires identity-based reconciliation, not clearing the pane
or declaring delivery from an internal completion event.

R09 remains open. The allocation repair needs live replay, and the discovered
in-flight reconnect case needs its own repair and regression proof.
