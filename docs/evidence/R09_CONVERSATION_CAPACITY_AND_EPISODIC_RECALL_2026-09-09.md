# R09 Conversation Capacity and Episodic Recall

## Observed Failure

The live 19:09:40 source question followed a database-recovery answer but
received a Solaris explanation. The route receipt reported three selected
exchanges; the worker reported no prior assistant turn. This is not evidence
that the model understood the supplied conversation incorrectly: intermediate
builders removed the conversation before dispatch.

Inspection found independent selection limits in the chat route, delivered
exchange translator, full response phase, assembler, and inference gate.
Some state and action contracts explicitly removed all previous answers.
Completed repeated questions were also excluded by matching their text to
the current question, losing exactly the exchange a repeated question pursues.

## Changes Under Test

- Chat reads the same 40-exchange bounded surface as UI restoration, through
  the existing principal-scoped durable reader. This is not unlimited history.
- Translation and both response paths retain complete delivered exchanges.
  Current evidence remains separate; question classification no longer
  removes the conversational record.
- The inference capacity owner retains the transcript when it fits its
  estimated serving input allowance. Under pressure it removes complete oldest
  exchanges, preserving a contiguous suffix and recording omission counts.
  The existing scaffold fitting and worker admission remain active.
- Repeated completed questions remain in history. IDs still deduplicate
  in-memory and durable copies; unfinished turns are not promoted.
- MemoryFacade.search now queries the existing episodic recall backend, with
  the same principal filtering and result verification as other sources.
  Superseded/non-authoritative episodes are excluded. Speaker labels and
  source exchange/session references survive normalization.

## Verification

The focused route selection suite passed 28 tests. The extended context,
memory, and tiering run passed 221 tests and exposed two test-fixture errors
(missing Episode timestamp and expected trailing spaces). After correction,
all 49 context-authority, memory-facade, and RAG integration tests passed.
The earlier capacity/history run passed 206 tests with one old seven-message
expectation; that expectation is now replaced by all 17 supplied messages.

A broad route batch was interrupted after a test initialized a real embedding
model during teardown. It reported a live-mind snapshot timeout in the
capability-inventory API test. That test isolation failure is not diagnosed
or counted as passing. Focused route tests did not reproduce it.

## Still Required

Release gates passed: smoke 164 passed, one skipped in 143.80 seconds;
lint, compile, governance lint, and layering passed. Governance migration
debt remains reported by its existing baseline. Source-matched live replay
is pending at this record.
R09 remains open. Test the database/source-question sequence, a reference to
an earlier non-adjacent exchange, repeated questions, restart history, and
episodic recall beyond the selected window. Measure latency and omissions.
This change does not prove retrieval completeness or resolve every over-window
input; a single exchange larger than capacity still needs a lossless reading
strategy. Temporal recall currently receives normalized metadata but its
formatter reads top-level timestamps/scores, which requires separate repair.

At 19:35 the actual neural feed also showed autonomous behavioral-gate clone
calls outside governed context, Pyright offline-scope denial, deferred episodic
writes, and diagnostic scheduler slips. These remain R08/R06 obligations;
the memory bridge alone does not establish successful durable capture.

## Post-Restart Replay and Recall Contract

At 19:59 the supported desktop Reboot action replaced PID 34020 with PID
58112. The boot endpoint measured expected and actual commit
`eb81668465ac0a6ab8045a853ebfb688305fb6ff` and matching workspace/source
hashes. Conversation history survived. Readiness was not stable: subsequent
health probes timed out while the model repeatedly ran readiness probes.

The actual neural feed showed deferred episodic capacity reaching 256 and
dropping pending writes (23 shed before restart). This is a durable-capture
failure, not evidence that recall alone repairs memory. The replacement also
reported conversation-history read timeouts on its reserved receipt executor.

The desktop correction replay was submitted at 20:00:44, delivery key
`aura-chat-3bb1aa28-2292-4a87-8922-5d0e36eccd34`, turn
`8aa47014253a44119d1cb9aa7597a45e`. It remained pending at 20:04:45.
R09 is not closed by this replay.

Temporal recall now consumes the facade's `score` and nested timestamp and
reinforcement fields without mutating retrieved records. Unknown timestamps
are labeled unknown rather than today. The focused facade/temporal suite
passed 37 tests; smoke passed 164 with one skipped in 180.49 seconds. This
formatter repair is not yet deployed to the live process.

At 20:05:16 the replay returned an incorrect attribution to
`core/science/environment_bench.py:313` / `RecoveryScore`. The terminal
receipt was failed, with `desktop_cognitive_engine_required_no_reply`,
`engine_think_invoked=false`, `recent_context_exchanges=0`, and no foreground
generation. The displayed reply identified a smaller-model fallback. This
does not measure the repaired engine history path: admission never invoked
that engine. It does expose incorrect evidence substitution in the fallback.

The next foreground replay reached the 27B Cortex with five recent exchanges,
but its draft was rejected as `fabricated_shared_history`. The assessment
boundary had passed only the current question to the shared-history checker,
even though generation received the transcript. It now supplies both prior
user turns and prior Aura replies from that exact message sequence. A grounded
"we were discussing database recovery" reply passes; an invented prison and
moonlit-courtyard recollection remains rejected. The focused inference and
shared-history suites passed 233 tests. This repair still requires deployment
and a successful live replay before R09 can close.

The deferred-write loss also had a control-point cause. Ontogeny's admission
contract places external and state-changing actions above its random
exploration ceiling, but the implementation omitted `write_memory` from that
set. An approved episode could therefore become `ontogeny:deferred` as an
experiment, repeat until the 256-item custody queue filled, and then be shed.
`write_memory` now carries the same high-stakes treatment as belief and state
mutation. The focused seal test passed; live capture after deployment remains
required.
