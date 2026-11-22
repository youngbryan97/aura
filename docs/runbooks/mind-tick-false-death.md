# Runbook: mind_tick false-death → "Connecting to runtime"

Covers [F15](../../KNOWN_FAILURE_MODES.md). **Fixed** — this runbook is for
recognising a recurrence, not for a live bug.

The loop was never dead. It was busy. Those are different, and the liveness
check could not tell them apart.

## Symptoms
- The desktop GUI drops to the "Connecting to runtime" reconnect surface.
- Health pulse reports `contract/important: mind_tick (is_alive returned False)`.
- The whole runtime flips DEGRADED — **while conversation still works**. That
  contradiction is the signature. A genuinely dead cognitive loop does not
  keep answering you.

## Diagnosis

The cognitive-rhythm loop marks progress at the top of each iteration. One
iteration that blocked on a saturated model — typically a background
initiative running the full 32B with no bound — stopped re-marking. The
liveness check saw a stale mark and called it death.

Confirm in this order:

1. Can she still answer a turn? If yes, this is F15 and not a real stall.
2. `aura doctor --bundle` → check whether `mind_tick` is the *only* failing
   contract loop. A real stall takes neighbours with it.
3. Look for a long-running background generation holding the model.

Rule out [aura-stuck-before-ready.md](aura-stuck-before-ready.md) (never
reached READY) and [high-event-loop-lag.md](high-event-loop-lag.md) (the loop
is genuinely starved).

## Safe mitigation
- Wait. It self-recovers: dead contract loops are revived from health-pulse
  threads via the owning event loop.
- A restart clears it immediately if you don't want to wait.

## Unsafe mitigation (last resort)
- None needed. Do not kill the worker for this — you would be trading a
  cosmetic DEGRADED flag for a real 20 GB model reload, which is
  [worker-crash.md](worker-crash.md) territory.

## Rollback
Not applicable. If this reappears, the regression is in one of three places:
the background kernel tick losing its bound, the health-pulse revival path,
or the GUI's `degraded_ready` handling.

## Verification
- Conversation ready and the GUI stays on the live surface rather than the
  reconnect screen.
- `contract/important` clean in the next health pulse.

## Postmortem checklist
- The fix has three parts and all three must hold: the background kernel tick
  is bounded and yields under foreground load; dead contract loops are revived
  via the owning event loop; the GUI stays in `degraded_ready` whenever
  conversation is ready.
- Add a regression test under `tests/`.
- The general shape here recurs across this codebase: **a subsystem that is
  working gets reported as broken by the thing watching it.** Before treating
  a DEGRADED flag as a real outage, check whether the underlying work is
  still succeeding.
