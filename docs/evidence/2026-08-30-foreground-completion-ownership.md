# Foreground Completion Ownership

## Reproduction

Source-matched desktop Aura at `5bdb46395` received this request on August 30:

> Compare optimistic and pessimistic locking for a hot task queue, choose
> which one you would use in a single-host async runtime, explain why, and
> verify your choice with one concrete failure scenario.

The selected DEEP mode reached deliberate generation. The second decode pass
kept its worker watchdog alive. The inference gate nevertheless cancelled the
request at its 259.8-second estimate. The worker finished its stream after
297.32 seconds. The desktop delivered an apology, not the requested answer.
Incident: `INC-1788137951-0001`. This replay failed.

## Repairs

- Resident foreground completion now has one client owner. Its request-scoped
  prefill, token progress, worker health and memory observations decide whether
  work remains viable. A total duration estimate cannot cancel a healthy reply.
  User cancellation still propagates. Background, health and explicitly bounded
  evaluation calls retain their declared deadlines.
- An expired estimate no longer turns future waiting into zero-time polling.
- Both decoder passes use one response consumer. Continuations now receive
  token-ledger updates, interoception, cache rollback capture, cancellation,
  progress publication and the existing completion checks.
- The injected channel boundary is recorded in the cache token ledger. A
  rollback from before that boundary is not reused for the continued answer.
- Performance includes every decode pass, rather than reporting the private
  pass's tokens and timing as the whole generation.

These changes do not modify the model prompt or its reasoning allowance.
They do not remove explicit evaluation budgets or memory-pressure enforcement.

## Verification

Focused worker, client, cache, deadline and continuation suites: 227 passed in
58.62 seconds. This includes a real MLX hybrid-cache continuation comparison.
Smoke: 121 passed, one environment-dependent skip. Ruff, compilation and
layering passed. Live replay of the repaired source remains required.

An additional lifecycle sweep passed 215 tests and found one stale source-text
assertion in `test_warmup_shares_one_campaign_budget`: it searches for the
retired `_READINESS_PROBE_PROMPT` name. It was not reported as a passing suite.

## Remaining Work

Replay the exact desktop request on the repaired source and inspect the full
answer. Exercise a follow-up to validate continuation cache reuse. Track actual
latency separately from successful delivery. Audit the broader action and turn
deadline callers for the same ownership conflict; this repair does not certify
every tool or every nested caller. Repository-wide semantic review and long
soaks remain separate, deferred work.

## Forecast Authority

The adjacent answer-clock code also changed the chosen primary model to the
tertiary model when its reasoning estimate exceeded a fixed cap. A later branch
reduced the requested answer allowance to fit that estimate. Both decisions
have been removed. Forecasts may report duration and increase headroom; they
cannot authorize a weaker cortex or a shorter foreground answer. Explicit
output contracts and resource admission retain their independent roles.

The forecast and completion regressions pass 45 tests. The desktop replay
currently in progress uses `d1b2157fa`, before this forecast-only follow-up.

## Replay: Completion Observer Stopped the Answer

The desktop replay at `d1b2157fa`, request
`aura-chat-56ea3125-1d72-4666-8794-a5b367ae88a2`, survived the former outer
deadline. It then failed delivery completeness: the worker stopped at token
1632 on `Semantic completion contract satisfied`, while the route reported
`missing_requested_objective_facets` and delivered PARTIAL at 18:26:02 PDT.
The requested concrete failure scenario was absent. Prefill took 35.56 seconds;
all decode passes took 287.05 seconds; total stream time was 323.92 seconds.
The answer also made questionable claims about async locks and contention.
Surviving the timer is proven by this replay; complete or correct answering is not.

Both worker loops now leave termination to model EOS, explicit stop contracts,
cancellation, output/resource limits and fault detection. The coverage observer
runs after decoding; a sentence boundary and apparently covered prefix no longer
stop a model still producing an answer. Streamed completion receipts are assessed
after the stream rather than requiring an observer-triggered abort. A post-hoc
coverage result cannot relabel a token limit as a semantic stop.

Focused continuation, real MLX/cache, termination and ownership tests: 96 passed.
Smoke: 121 passed, one skip. Live replay of this additional repair remains open.

Other live observations remain open: the background reimplementation workspace
write lacked governed context; the Reflex lane failed an embedding-engine
eviction; the UI described a phi proxy as a measurement out of one while showing
1.13. Those are recorded findings, not fixed by this completion change.

## Replay: Full Ending, Incorrect Coverage Classification

Source-matched desktop Aura at `f16efaaad` received the same locking request
at 18:37:56 PDT. Request `aura-chat-1c7ace5a-784e-4895-a2ce-7159a20595df`
delivered at 18:44:02. The response included the requested concrete scenario
and its ending. Prefill took 32.58 seconds; decoding took 295.96 seconds;
the stream took 329.29 seconds for 1,848 generated tokens. Peak MLX memory
was 16.10 GB. The former deadline and prefix-stop failures did not recur.

The route still labelled the response PARTIAL. Offline assessment of the
persisted answer found the facet checker recognized compare and explain but
missed the explicit choice and worked verification. The answer also confused
async lock suspension with blocking and overstated exactly-once execution.
An inference lease expired during generation and the later repair step had
no transaction budget left. This is not full-mind or answer-quality closure.

A follow-up asking whether an async lock permits other tasks to keep running
was routed to desktop_task and computer_use. No desktop action was requested.
Aura was gracefully stopped at 18:46 PDT to prevent that route continuing;
shutdown completed cleanly. The watched-goal request parser is the next repair.

## Progress Counter Ownership

Worker progress now counts the cumulative generated-token delta instead of
counting each batched event as one token. Duplicate, decreasing, malformed,
wrong-request and idless counters cannot renew token progress. Advancing
prefill updates the turn clock only after request ownership is established;
repeated prefill messages do not renew it. Legacy visible-token messages
retain their one-token behavior. Each request resets its prefill rate sample.

Focused client, progress, resilience and clock tests: 139 passed in 99.10
seconds. Smoke: 121 passed, one skip. Lint, compile, layering and governance
passed. These counter changes have not yet been loaded into desktop Aura.

## August 31: Desktop Question Routing Replay

Signed-app launch at `bb124e5c3` verified source currency and revision token
`3170bfaf40195b8d3421d67572520ff29f154b6f9bb3f508ae606230a48289bc`.
The native composer submitted the exact async-lock follow-up at 09:51:07 PDT.
Request `aura-chat-14abe513-219a-4ec6-807c-08756ca0ba62` delivered at 09:56:32;
the route measured 324,670.49 ms. The answer was complete, model-authored and
accepted, with no desktop execution. The persisted receipt reports
`full_mind_path=true`, `bounded_contract_used=false`, `legacy_fallback_used=false`,
`answer_delivery_proven=true`, and zero text mutations. RLC was not selected.

The reply correctly distinguished suspending an async-lock waiter from running
a no-await critical section. It introduced a C#-style `lock`/Monitor comparison;
this is not a proof of Python-specific advice. The restarted desktop supplied
zero recent exchanges, so this replay proves standalone question routing, not
cross-restart conversational continuity. Latency remains poor. CPU-only test
gates ran during generation; this is not an idle-host performance benchmark.

The neural stream and terminal also showed background email activity briefly
labelled as progress for the chat, event-loop lag of 5.333 and 6.145 seconds,
a 347 ms blocking hold reported at `turn_progress.py:61`, transient health
degradation, and repeated health-probe HTTP wait overruns. The runtime remained
alive and delivered the answer. The worker reported sparse non-parametric
memory, one private-channel persona warning and 1,774 total generated tokens.
These findings remain open. Graceful shutdown completed at 09:57:24 PDT.

## August 31: Owned Progress and Renewable Foreground Waits

The next repair replaces process-global foreground progress with a reading
attached to TurnOutcome. The MLX request captures it at dispatch, and the
thread watchdog captures it before starting. Tool activity retains its original
owner through completion and cancellation cleanup. Task exit removes abandoned
activity; finalized turns reject late progress. Unbound compatibility callers
cannot renew a bound foreground turn.

The endpoint and cognitive-cycle clocks allow owned, advancing foreground work
to finish beyond their old duration ceilings. The cycle never shortens the
initial admitted window. Probes and evaluation calls do not receive this
renewal. Endpoint cancellation awaits cleanup, and task-raised TimeoutError
propagates as a worker failure rather than starting another wait.

The focused test run passed 191 tests in 38.89 seconds, including concurrent
turns, cross-context worker callbacks, abandoned tool activity, cancellation
cleanup, bounded probes, and an answer completing beyond a reduced former
ceiling. Smoke passed 121 with one skip in 19.72 seconds. Lint, compile and
layering pass. Governance reports the unchanged raw diagnostic write in
`core/perception/where_it_responds.py::_note_what_was_seen`.

The earlier replacement runtime loaded the reservation-lifetime repair. Its
code, workspace and shell-asset hashes matched, but a concurrent documentation
commit changed the commit pin from `a5be9632e` to `992cbfde8`, so boot correctly
reported `source_current=false`. No live result is attributed to the new
progress changes yet.
