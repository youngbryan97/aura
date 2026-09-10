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
