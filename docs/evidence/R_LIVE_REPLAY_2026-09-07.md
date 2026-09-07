# Runtime replay, September 7

The supported launcher started PID 11289 on revision
`b77fc87c5585595f32d30dd77a3663679da8e9d2`. Boot reported ready,
source_current=true, and the resident persona/CRSM 27B model. This was a
direct launch with a bound source snapshot, not a signed release deployment.

At 00:40:59 local time, the browser submitted a question about lost updates
and compare-and-swap. The neural feed and desktop log were read together.
Search returned dictionary entries about "why"; subject relevance rejected
them. The foreground request continued, but did not pass live validation.

The parent suffered event-loop stalls up to 45.8 seconds. At 00:47:49 the
client cancelled for token_progress_stalled. Worker telemetry reported 1,544
prefill tokens in 10.50 seconds and 665 generated tokens in 141.57 seconds.
Native thinking remained open, leaving an empty public answer. A lower-lane
fallback later produced an explanation. This is not R03/R05 acceptance.
The reconnected browser initially showed the user turn without its final
answer, so reconnect delivery also remains unproven under R09.

An OS sample was captured at `/tmp/aura-r06-live-stack-20260907.txt`.
The desktop log's thread dump included health integrity filesystem traversal
and process-tree memory accounting; the OS sample showed substantial GIL
waiting. These identify investigation targets, not a proven single cause.

Two bounded repairs follow from the inspection:

- Continuity snapshots serialize writes per event loop and path. Flush
  reports completed write failures instead of allowing shutdown to log a
  successful save. Three storage tests cover waiting, ordering and failure.
- Inference health now receives request-bound prefill advancement from the
  existing client tracker. It accepts recent prefill without extending the
  startup timeout; stale decode evidence still takes precedence.

The focused health and continuity suite passed 40 tests. Smoke was stopped
after 224.83 seconds without progress in `core/interiority/effects.py:277`:
85 passed, one skipped, remaining tests unrun. A concurrent unrelated test
batch was present. This is not a smoke pass. Full-file Ruff also reports
pre-existing findings in inference_gate and its health tests.

No additional R item is marked complete by this replay. The repairs above
require deployment and another live replay; the live source was kept stable
during this turn.

The generation wait loop also performed process-memory observation and
garbage collection synchronously on every timed-out polling slice. Both now
run through `asyncio.to_thread`, allowing response delivery to advance while
the observation runs. The completion-ownership tests assert that observation
and collection execute off the event-loop thread: 14 passed. The client
resilience suite also passed all 96 tests. This removes one measured-path
blocking operation, not every source of loop contention.

Controlled reboot replaced PID 11289 with PID 19222 on commit
`d4b0879a89db2954425dcb26348411b80c0af209`. Concurrent source edits changed
the workspace fingerprint, so this is diagnostic evidence, not a clean
source-matched closeout. The next request kept the event loop responsive
(reported maximum 1 ms), but two drafts failed completion checks and the
protected fallback hit its outer timeout. R05 remains open independently
of the event-loop repair.

The integrity collector now serializes concurrent off-loop scans and
rechecks freshness after acquiring collection ownership. Event-loop readers
still return immediately. All 13 health-responsiveness tests passed,
including four concurrent callers sharing one scan.

The protected foreground reply had its own `asyncio.wait_for` around the
inference gate with the original time estimate. The live log measured a
worker allowance of 68 seconds while that caller cancelled after roughly
nine seconds. The route now delegates completion to the existing resident
client, keeping its budget input, cancellation and resource enforcement.
Seventeen focused route/client tests passed, including benchmark exclusion
and cancellation propagation. This repair does not establish that the
primary decoder produces complete answers; it prevents one independent
caller from cancelling the client's revised allowance.

## Desktop replay and compound facet defect

The desktop-header request `r-desktop-replay-20260907-1` completed in the
delivery journal on PID 24412. Its counter/CAS explanation reached the
public response. The neural log nevertheless rejected the draft for
`missing_requested_objective_facets`, attempted correction, then served
preserved authored work. This is not a clean R05 or R09 pass.

The facet extractor interpreted the command word inside `compare-and-swap`
as a separate comparison obligation. It now excludes command matches inside
hyphenated lexical units while retaining explicit commands elsewhere in
the request. Focused output-quality tests: 23 passed. Smoke: 164 passed,
1 skipped in 36.46 seconds. Live replay of this repair remains required.

The retired-process liveness repair was pushed as `153ee712c`. This does
not close the remaining cross-owner eviction audit in R04.

## Updated runtime replay

PID 31016 replaced PID 24412 through `/api/reboot`, expecting `df284d88e`.
Concurrent commits changed the checkout after startup; this is diagnostic
evidence, not source-matched certification. The fixed counter/CAS replay
completed with a substantive answer and high response confidence.

The compound locking request `r03-native-live-20260907-1` selected latent
execution but admission returned `answer_surface_unaffordable_before_execution`.
Its answer used `cognitive_engine_repair_retry`. It incorrectly suggested
awaiting an async lock stalls unrelated work. Its duplicate-processing
argument also needs a successful claim before processing. R03 and R10
remain open despite HTTP completion and high confidence.

The neural stream exposed on-loop boot-profile persistence and fsync under
the versioned-store state lock. Repairs `47cb71e8d` and `115299a8b` are
pushed but postdate this runtime. Boot-profile tests: 14 passed.
Versioned-store tests: 30 passed, including concurrent hold during a paused
write. Smoke: 164 passed, 1 skipped in 35.36 seconds.
