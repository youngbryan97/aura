# Runbook: quadratic conversation cost from a never-reused prompt cache

Covers [F19](../../KNOWN_FAILURE_MODES.md). **Fixed.** Root cause in
`artifacts/closeout/endurance_ceiling/ROOT_CAUSE.md`.

This was the "15-turn endurance ceiling," and for a long time it was blamed
on cognition. It was not cognition. Every turn was re-reading the entire
conversation from the first token.

## Symptoms

The signature is a latency staircase with **zero deaths**:

    turn 1    11s
    turn 2    25s
    turn 3   105s
    turn 5   112s
    turn 8+  pinned at the ceiling

- Prompt-cache hit count of zero. Not low. Zero.
- The run dies around turn 20 of 200 by timing out, never by crashing.
- Nothing in the logs looks like a fault, because nothing faulted.

If turns are slow *and* something is crashing, you are in
[worker-crash.md](worker-crash.md) or
[mlx-worker-cold-lane-cascade.md](mlx-worker-cold-lane-cascade.md) instead.

## Diagnosis

Two independent bugs, either sufficient on its own:

1. `_prompt_cache_entry_budget_for_model` returned a budget of **0** for the
   32B under `desktop_resource_guard_enabled()`. The prompt-cache LRU was
   therefore never constructed on the live desktop.
2. Every live user turn carries `clean_user_surface_contract=True`, set by
   the client whenever runtime controls are bound. That contract was in
   `_job_requires_prompt_cache_bypass`, which did not merely skip the cache —
   it **cleared** it each turn.

Net effect: per-turn cost linear in history, total conversation cost
quadratic. Under paced turns the staircase reaches any fixed timeout.

Two amplifiers, both in the forensics and both worth knowing because they
disguise the cause:

- **`JobWatchdog` kills on 90s without a token.** Prefill emits no tokens. So
  once re-prefill alone crossed 90s, the watchdog killed a *healthy* worker
  mid-prefill; respawn plus a 20 GB reload cost about two minutes and
  surfaced as a 216s or 500s turn.
- **Model-load admission stalled the event loop.** During those reloads
  `_declared_mlx_worker_footprint_gb → _path_size_gb` ran a synchronous
  `rglob`+`stat` walk of the model directory on the event loop while 20 GB of
  safetensors reads saturated the disk. See `data/error_logs/stalls/`.

To confirm a recurrence: run a paced multi-turn conversation and watch the
prompt-cache hit count. If it is zero while history grows, it's this.

## Safe mitigation
- Start a new conversation. Cost resets with history length.
- Verify the cache budget is non-zero for the loaded model before assuming
  the bug is elsewhere.

## Unsafe mitigation (last resort)
- Raising the watchdog timeout hides the staircase without fixing it, and
  costs you the protection the watchdog exists for.

## Rollback
A recurrence means either the cache budget went back to 0 for the resident
model under the resource guard, or a contract was re-added to the bypass
list. Check both before looking anywhere else.

## Verification
- Prompt-cache hits greater than zero across a multi-turn conversation.
- Per-turn latency flat as history grows, instead of climbing.
- No `Loading model:` storm behind a long conversation.

## Postmortem checklist

This is the canonical instance of the dominant defect class in this
codebase: **a good answer discarded by a gate, then reported as an
infrastructure failure.** The model was fine. The cognition was fine. A
cache-bypass list threw the work away and the symptom looked like a slow
brain.

When a subsystem looks slow or dead, check first whether something upstream
is discarding correct work.
