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
