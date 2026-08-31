# Known Failure Modes — Aura Cognitive Runtime

## Purpose

Nineteen ways this runtime fails, what each one looks like, and what to do
about it. Every one has a runbook.

Read the split before you read the list. **F01–F14 are failure classes we
plan for.** F15–F19 are different: they actually happened, on the live
desktop, under sustained conversation, with forensics still on disk. Those
five are the real daily-runtime edges. If you only have time for part of
this document, read those.

One of them, F16, is **not fully fixed** and says so. MLX cannot soft-cancel
a running generation, so freeing a busy worker means killing it and
unloading 18 GB. The kill is the recovery. Mitigations make that survivable;
they don't make it go away.

Every operator should read this before running Aura in any production-like
setting.

## Critical Failure Modes

### F01: Model fails to load

**Cause**: Insufficient RAM, corrupted weights, missing model files
**Likelihood**: Low (first boot) / Very Low (steady state)
**Impact**: No inference capability
**Detection**: Boot probe failure; health check reports `brainstem: not_initialized`
**Recovery**: `make doctor` → validates model files → re-download if needed
**Runbook**: `docs/runbooks/model-fails-to-load.md`

### F02: Worker process crash during inference

**Cause**: GPU memory pressure, MLX runtime error, corrupted prompt
**Likelihood**: Low
**Impact**: Current request fails; auto-recovery spawns a new worker
**Detection**: Worker health probe; `record_degradation("mlx_worker", ...)`
**Recovery**: Automatic — InferenceGate respawns a worker. Caveat (see F16):
respawn requires ~24GB headroom, and immediately after a kill the OS reclaim
of the ~18GB model lags process exit, so respawn is briefly refused; the gate
now waits for reclaim (`AURA_MLX_SPAWN_RECLAIM_WAIT_S`) before refusing.
**Runbook**: `docs/runbooks/worker-crash.md`

### F03: Memory database corruption

**Cause**: Dirty shutdown, disk full, concurrent write race
**Likelihood**: Very Low
**Impact**: Memory retrieval fails; boot may degrade
**Detection**: SQLite integrity check on boot; state hash mismatch
**Recovery**: `make restore` from last backup; WAL replay
**Runbook**: `docs/runbooks/memory-corruption.md`

### F04: Shutdown hangs

**Cause**: Blocked async task, hung worker, deadlocked service
**Likelihood**: Low
**Impact**: Process requires SIGKILL
**Detection**: Shutdown timeout (12s budget); watchdog
**Recovery**: SIGKILL + clean boot; bounded shutdown prevents forever-hang
**Runbook**: `docs/runbooks/shutdown-hang.md`

## High Severity Failure Modes

### F05: External interlocutor transmits more than the objective needs

**Cause**: A governed web-interlocutor session composes a message carrying
context beyond the objective it was opened for
**Likelihood**: Very Low (host allowlist, per-run turn budget, body inspection)
**Impact**: Content reaches an external AI surface through the user's browser
**Detection**: Governed network receipts; `core/security/egress_privacy.py`
inspection records
**Recovery**: Quarantine the destination host; audit the transmitted payloads
from local receipts; repair the composer
**Runbook**: `docs/runbooks/external-egress.md`

There is no cloud inference fallback to leak through. Every lane the router
can reach is local, and `allow_cloud_fallback` is coerced to `False` in the
request contract — see `docs/runbooks/local-inference-boundary.md`.

### F06: Prompt injection succeeds

**Cause**: Novel injection technique bypasses sanitizer + integrity check
**Likelihood**: Low (multi-layer defense)
**Impact**: Aura performs unintended action
**Detection**: Will receipt audit; anomalous action patterns
**Recovery**: Revert affected memory writes; review Will receipt chain
**Runbook**: `docs/runbooks/prompt-injection.md`

### F07: Resource exhaustion (RAM/GPU)

**Cause**: Large context, multiple concurrent requests, memory leak
**Likelihood**: Medium (under load)
**Impact**: Degraded performance; potential OOM kill
**Detection**: Metabolic monitor; resource governor alerts
**Recovery**: Automatic tier demotion; request throttling; restart if needed
**Runbook**: `docs/runbooks/resource-exhaustion.md`

### F08: Background task orphaning

**Cause**: Task creator dies without cleaning up background work
**Likelihood**: Low
**Impact**: Wasted resources; potential stale state
**Detection**: Task tracker orphan detection; hypervisor reaping
**Recovery**: Hypervisor kills orphaned tasks; cleanup on next boot
**Runbook**: `docs/runbooks/orphaned-tasks.md`

## Medium Severity Failure Modes

### F09: Stale memory retrieval

**Cause**: Vector DB index drift; outdated embeddings
**Likelihood**: Medium (over time)
**Impact**: Irrelevant context in responses
**Detection**: Memory retrieval quality metrics; user feedback
**Recovery**: Re-index memory; consolidation cycle
**Runbook**: `docs/runbooks/stale-memory-retrieval.md`

### F10: Identity drift

**Cause**: Sustained adversarial prompting; corrupted CanonicalSelf state
**Likelihood**: Very Low
**Impact**: Aura's personality/identity becomes inconsistent
**Detection**: Identity coherence check; CanonicalSelf hash
**Recovery**: Reset CanonicalSelf from canonical snapshot
**Runbook**: `docs/runbooks/identity-drift.md`

### F11: Tool execution timeout

**Cause**: Slow external service; large file operation; network timeout
**Likelihood**: Medium
**Impact**: Individual tool call fails
**Detection**: Timeout enforcement; degradation recording
**Recovery**: Automatic — tool reports failure; Aura retries or explains
**Runbook**: `docs/runbooks/tool-timeout-storm.md`

### F12: Lock contention/deadlock

**Cause**: Multiple subsystems contending for same resource
**Likelihood**: Low
**Impact**: Request stalls until watchdog releases
**Detection**: Lock watchdog; stall detection
**Recovery**: Automatic — watchdog releases stale locks after threshold

## Low Severity Failure Modes
**Runbook**: `docs/runbooks/lock-contention-deadlock.md`

### F13: Log rotation failure

**Cause**: Disk full; permission error
**Likelihood**: Very Low
**Impact**: Logs stop writing; no data loss
**Detection**: Log write error; disk space monitor
**Recovery**: Free disk space; restart log rotation
**Runbook**: `docs/runbooks/log-rotation-failure.md`

### F14: Telemetry emission failure

**Cause**: Metrics endpoint unavailable
**Likelihood**: Low (local deployment)
**Impact**: Missing observability data
**Detection**: Telemetry health check
**Recovery**: Restart telemetry; data gap in dashboard

## Observed Failure Modes (2026-07, live-runtime)

These were seen and root-fixed on the live desktop instance under sustained
conversation. They are documented because they are the *real* daily-runtime
edges, not hypotheticals.
**Runbook**: `docs/runbooks/telemetry-emission-failure.md`

### F15: mind_tick false-death → "Connecting to runtime"

**Cause**: The cognitive-rhythm loop marks progress at the top of each
iteration; a single iteration that blocked on a saturated model (e.g. a
background initiative running the full Cortex with no bound) stopped re-marking
progress, so `is_alive()` declared `mind_tick` dead.
**Likelihood**: Medium under sustained back-to-back turns (before fix).
**Impact**: Whole runtime flips DEGRADED even though conversation works; the
desktop GUI reverts to the "Connecting to runtime" reconnect surface.
**Detection**: Health pulse `contract/important: mind_tick (is_alive returned False)`.
**Recovery**: Fixed — the background kernel tick is bounded and yields under
foreground load; dead contract loops are revived from health-pulse threads via
the owning event loop; the GUI keeps the live UI in a `degraded_ready` state
whenever conversation is ready. Self-recovers; a restart clears it immediately.
**Runbook**: `docs/runbooks/mind-tick-false-death.md`

### F16: MLX worker-kill cold-lane cascade (the honest daily-stability edge)

**Cause**: MLX cannot soft-cancel a running generation, so freeing a busy
worker means force-killing it (unloading the ~18GB model). A foreground deep
generation that exceeds its budget therefore kills the worker; on a
memory-constrained host the reload races the next turn's timeout, which kills
the reloading worker and restarts the load — a cold-lane cascade.
**Likelihood**: Medium on a host with <~25GB free (e.g. other apps running).
**Impact**: A cluster of turns returns 503 / fail-closed until the model
finishes loading; RSS cycles (21GB→~1GB→reload). Self-recovers; RSS stays
bounded (this is NOT the OOM growth of F07).
**Detection**: `Cortex generation exceeded inference-gate timeout … aborting`
followed by repeated `Loading model:`; worker RSS drops to ~0.
**Recovery**: Partially mitigated — background timeouts no longer kill the
shared worker, respawn waits for memory reclaim, and mid-load workers are not
torn down. **Open architectural work**: a soft-cancel path into the MLX worker
or a persistent model server; more host RAM headroom removes the cascade
entirely.
**Runbook**: `docs/runbooks/mlx-worker-cold-lane-cascade.md`

### F17: Failure-lockdown escalation from expected backpressure

**Cause**: A bounded background generation (memory consolidation, dialectical
crucible) timing out while the foreground lane holds the model was recorded as
a degradation on a *fail-closed* subsystem, which escalated a plain
`TimeoutError` to a CRITICAL SERVICE FAILURE and drove
`unified_failure_lockdown` toward 1.00.
**Likelihood**: Was high under load; low after fix.
**Impact**: At lockdown 1.00, memory writes, tool execution, and
self-modification are all blocked; existential-threat spikes.
**Detection**: `unified_failure_lockdown_1.00` in the log; `Executive REJECTED`
lines for memory/tool actions.
**Recovery**: Fixed — `core/runtime/backpressure.py` records expected
backpressure on a non-fail-closed channel with the policy disabled; foreground
yields precede background generation.
**Runbook**: `docs/runbooks/failure-lockdown-from-backpressure.md`

### F18: Launch-provenance `ready:false` on source drift

**Cause**: A signed `Aura.app` pins the exact commit + workspace hash it was
built for; running code that has drifted forward (active development) fails the
provenance check.
**Likelihood**: Every launch of an actively developed checkout.
**Impact**: `ready:false` with a `launch_provenance` blocker. She stays fully
conversational (the `degraded_ready` path); it is a correct tamper-detection
signal, not a functional break.
**Detection**: `boot_phase: launch_provenance_failed`; issues
`commit_sha_mismatch` / `workspace_state_sha256_mismatch`.
**Recovery**: Expected in dev. To clear: rebuild/re-sign the app to re-pin, or
launch via `launch_aura.sh` (which does not require provenance).
**Runbook**: `docs/runbooks/launch-provenance-not-ready.md`

### F19: Quadratic conversation cost from a never-reused prompt cache

**Cause**: The conversation path could not reuse KV, twice over.
`_prompt_cache_entry_budget_for_model` gave the Cortex a budget of **0** under
`desktop_resource_guard_enabled()`, so the prompt-cache LRU was never
constructed on the live desktop; and every live user turn carries
`clean_user_surface_contract=True`, which was in the bypass list — so the
cache was not merely skipped but *cleared* each turn. Every turn re-prefilled
the entire conversation from token 0: per-turn cost linear in history, total
conversation cost quadratic.
**Likelihood**: Was every long conversation; fixed.
**Impact**: Latency staircase (11s → 25s → 105s → …) pinning at the turn
ceiling from roughly turn 8–15, with the run dying by turn 20 of 200. Deaths
were **0** — the model never crashed, the turns simply outgrew the timeout.
This was the real "15-turn endurance ceiling", which had been attributed to
cognition.
**Detection**: Monotonically climbing per-turn latency with zero deaths, and
a prompt-cache hit count of zero.
**Amplifiers, both recorded in forensics**: `JobWatchdog` kills on 90s
without a token, but prefill emits no tokens — so once re-prefill alone
crossed 90s the watchdog killed a *healthy* worker mid-prefill, and respawn
plus a 20 GB reload cost roughly two minutes. During those reloads,
`_declared_mlx_worker_footprint_gb → _path_size_gb` ran a synchronous
`rglob`+`stat` walk of the model directory **on the event loop** while 20 GB
of safetensors reads saturated the disk (`data/error_logs/stalls/`).
**Evidence**: `artifacts/closeout/endurance_ceiling/ROOT_CAUSE.md`.

**The general lesson, which recurs**: the dominant defect class in this
codebase is *a good answer discarded by a gate, then reported as an
infrastructure failure*. When a subsystem looks slow or dead, check first
whether something upstream is throwing away correct work.

## Recovery Drill Schedule

| Drill | Frequency | Procedure |
|-------|-----------|-----------|
| Backup/restore | Monthly | `make backup && make restore-test` |
| Dirty shutdown recovery | Quarterly | Kill -9 → verify boot |
| Model re-download | Quarterly | Delete model → verify re-acquisition |
| State corruption recovery | Quarterly | Corrupt test DB → verify recovery |
| Full disaster recovery | Annually | Fresh machine → full install → restore |
**Runbook**: `docs/runbooks/prompt-cache-never-reused.md`
