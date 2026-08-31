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
