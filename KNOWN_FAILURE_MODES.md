# Known Failure Modes — Aura Cognitive Runtime

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](docs/DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

## Purpose

This document explains the nineteen core failure modes in the Aura runtime: what causes each issue, what it looks like, and how to fix it. Every failure mode has a dedicated troubleshooting runbook (part of 41 incident runbooks maintained in `docs/runbooks/`).

Understanding the distinction between these failures is important:
- **F01–F14 are planned failure classes.** These are operational edge cases anticipated and planned for during system design.
- **F15–F19 are real issues observed in production.** They occurred on a live desktop machine during sustained conversations, with full diagnostic records saved on disk. These five represent real-world operational challenges rather than theoretical scenarios. If you only have time to read part of this guide, start with these five.

Note that **F16 is not yet fully fixed**, as documented below. Apple's MLX framework cannot gracefully pause or cancel an AI model while it is generating text. As a result, the only way to free a busy worker process is to terminate it completely and unload its 18 GB model from memory. Terminating the process is the recovery mechanism. While existing safeguards keep the system running, they do not eliminate the root issue.

Every operator should read this guide before running Aura in any production environment.

## Critical Failure Modes

### F01: Model fails to load

**Cause**: Insufficient RAM, corrupted model weight files, or missing model files
**Likelihood**: Low (on first boot) / Very Low (during normal operation)
**Impact**: Aura cannot run language models (no inference capability)
**Detection**: Boot checks fail; health check reports `brainstem: not_initialized`
**Recovery**: Run `make doctor` to validate model files and re-download any missing or corrupted files
**Runbook**: `docs/runbooks/model-fails-to-load.md`

### F02: Worker process crash during inference

**Cause**: High GPU memory pressure, an MLX runtime error, or an invalid prompt
**Likelihood**: Low
**Impact**: The current request fails; an automatic recovery system spawns a replacement worker
**Detection**: Worker health check; `record_degradation("mlx_worker", ...)`
**Recovery**: Automatic — `InferenceGate` restarts a worker process. *Caveat (see F16):* restarting requires about 24 GB of free memory headroom. Because macOS takes a few seconds to reclaim the ~18 GB model from memory after a process terminates, an immediate restart might fail. The gate now pauses for memory to be reclaimed (`AURA_MLX_SPAWN_RECLAIM_WAIT_S`) before attempting to spawn.
**Runbook**: `docs/runbooks/worker-crash.md`

### F03: Memory database corruption

**Cause**: Unclean shutdown, full disk, or simultaneous conflicting writes
**Likelihood**: Very Low
**Impact**: Memory lookup fails; system startup may be degraded
**Detection**: SQLite integrity check fails on boot; state hash mismatch
**Recovery**: Run `make restore` from the latest backup; replay SQLite Write-Ahead Logs (WAL)
**Runbook**: `docs/runbooks/memory-corruption.md`

### F04: Shutdown hangs

**Cause**: A blocked asynchronous task, a hung worker, or deadlocked services
**Likelihood**: Low
**Impact**: The process refuses to exit cleanly and requires a forced kill (SIGKILL)
**Detection**: Shutdown timer exceeds its 12-second budget; watchdog alert
**Recovery**: The watchdog issues a SIGKILL followed by a clean reboot; a hard 12-second shutdown timeout prevents infinite hangs
**Runbook**: `docs/runbooks/shutdown-hang.md`

## High Severity Failure Modes

### F05: External interlocutor transmits more than the objective needs

**Cause**: A managed web-browsing session includes more conversation context in an outgoing message than was required for its specific task
**Likelihood**: Very Low (protected by allowed-domain lists, per-run turn limits, and outbound message inspection)
**Impact**: Private context reaches an external AI service through the user's browser
**Detection**: Logged network receipts; inspection logs in `core/security/egress_privacy.py`
**Recovery**: Quarantine the destination website; audit transmitted payloads in local receipts; update the message composer
**Runbook**: `docs/runbooks/external-egress.md`

There is no cloud inference fallback that could leak data. Every model lane Aura connects to runs locally on the host machine, and `allow_cloud_fallback` is hardcoded to `False` in the request contract — see `docs/runbooks/local-inference-boundary.md`.

### F06: Prompt injection succeeds

**Cause**: A novel prompt injection technique bypasses input sanitization and safety filters
**Likelihood**: Low (defended by multiple inspection layers)
**Impact**: Aura executes an unintended or unauthorized action
**Detection**: Audit of signed Will decision receipts; detection of abnormal action patterns
**Recovery**: Roll back affected memory writes; review the Will decision receipt chain
**Runbook**: `docs/runbooks/prompt-injection.md`

### F07: Resource exhaustion (RAM/GPU)

**Cause**: Very large prompt context, many simultaneous requests, or a memory leak
**Likelihood**: Medium (under heavy system load)
**Impact**: Sluggish performance or process termination by the operating system (Out of Memory / OOM kill)
**Detection**: System resource monitors; alerts from resource governors
**Recovery**: Automatic tier demotion (switching to smaller, lighter models); request throttling; restarting services if needed
**Runbook**: `docs/runbooks/resource-exhaustion.md`

### F08: Background task orphaning

**Cause**: A parent task terminates without cleaning up its background worker processes
**Likelihood**: Low
**Impact**: Wasted CPU and memory resources; stale background state
**Detection**: Task tracker flags orphaned jobs; system supervisor sweeps for inactive tasks
**Recovery**: Supervisor terminates orphaned tasks; cleanup routine runs on next boot
**Runbook**: `docs/runbooks/orphaned-tasks.md`

## Medium Severity Failure Modes

### F09: Stale memory retrieval

**Cause**: Search index drift in the vector database; outdated embeddings (vector representations of text)
**Likelihood**: Medium (accumulates over time)
**Impact**: Irrelevant context included in responses
**Detection**: Memory retrieval relevance scores drop; user feedback
**Recovery**: Re-index the memory database; run a memory consolidation cycle
**Runbook**: `docs/runbooks/stale-memory-retrieval.md`

### F10: Identity drift

**Cause**: Repeated adversarial user prompts; corrupted `CanonicalSelf` personality state
**Likelihood**: Very Low
**Impact**: Aura's personality or behavioral boundaries become inconsistent
**Detection**: Identity consistency check fails; `CanonicalSelf` checksum/hash mismatch
**Recovery**: Reset `CanonicalSelf` from a clean canonical snapshot
**Runbook**: `docs/runbooks/identity-drift.md`

### F11: Tool execution timeout

**Cause**: Slow external web service, large file read/write, or network delay
**Likelihood**: Medium
**Impact**: An individual tool call fails
**Detection**: Timeout limit reached; degradation event logged
**Recovery**: Automatic — the tool reports an error, and Aura either retries or explains the issue to the user
**Runbook**: `docs/runbooks/tool-timeout-storm.md`

### F12: Lock contention/deadlock

**Cause**: Multiple subsystems compete for the same resource simultaneously
**Likelihood**: Low
**Impact**: A request stalls until the lock watchdog intervenes
**Detection**: Lock watchdog alert; stall detection timer
**Recovery**: Automatic — the watchdog releases abandoned locks once a timeout threshold is reached
**Runbook**: `docs/runbooks/lock-contention-deadlock.md`

## Low Severity Failure Modes

### F13: Log rotation failure

**Cause**: Hard drive is full; file permission error
**Likelihood**: Very Low
**Impact**: New logs stop writing to disk; no existing data is lost
**Detection**: Log write error; low disk space alert
**Recovery**: Free up disk space; restart the log rotation service
**Runbook**: `docs/runbooks/log-rotation-failure.md`

### F14: Telemetry emission failure

**Cause**: Metrics service endpoint is unreachable
**Likelihood**: Low (in local deployments)
**Impact**: Monitoring dashboards miss observability data
**Detection**: Telemetry health check alert
**Recovery**: Restart the telemetry service; historical gap remains on dashboards
**Runbook**: `docs/runbooks/telemetry-emission-failure.md`

## Observed Failure Modes (2026-07, live-runtime)

These five issues were identified and resolved on a live desktop machine during extended conversations. They are documented here because they represent real, everyday operational challenges rather than hypothetical problems.

### F15: mind_tick false-death → "Connecting to runtime"

**Cause**: The cognitive rhythm loop signals progress at the start of each iteration. If an iteration stalled waiting for a busy model (such as a background task running the full Cortex model without time limits), it stopped sending progress updates. As a result, `is_alive()` incorrectly assumed `mind_tick` was dead.
**Likelihood**: Medium during rapid back-to-back conversation turns (before the fix).
**Impact**: The entire runtime switched to a DEGRADED status even though chat still worked, causing the desktop interface to revert to the "Connecting to runtime" reconnect screen.
**Detection**: Health check alert: `contract/important: mind_tick (is_alive returned False)`.
**Recovery**: Fixed — the background task loop is now strictly bounded and yields time when the user is chatting; stalled background loops are revived by health-check threads through their owning event loop; and the desktop GUI stays in a `degraded_ready` state whenever conversation is available. The system self-recovers, and a restart clears the state immediately.
**Runbook**: `docs/runbooks/mind-tick-false-death.md`

### F16: MLX worker-kill cold-lane cascade (the honest daily-stability edge)

**Cause**: Apple's MLX library cannot safely interrupt an in-progress model generation. To stop a stuck worker, the runtime must force-kill it, which purges the ~18 GB model from RAM. If a deep foreground prompt exceeds its allowed time budget, the worker is killed. On a machine with limited free memory, reloading the 18 GB model takes time. If the next user turn arrives and times out while the model is still loading, that reloading worker is also killed—triggering a repeating cascade where the model never finishes loading.
**Likelihood**: Medium on machines with less than ~25 GB of free RAM (for example, when other large applications are open).
**Impact**: Several user turns in a row return a 503 error until the model finally finishes loading into memory. Process memory (RSS) repeatedly cycles (21 GB → ~1 GB → reload). The system eventually recovers on its own without unbounded memory growth (unlike the true memory leak described in F07).
**Detection**: Log messages like `Cortex generation exceeded inference-gate timeout … aborting` followed by repeated `Loading model:` lines, while worker memory drops to near 0.
**Recovery**: Partially mitigated — background timeouts no longer terminate the shared worker, worker restarts now wait for the operating system to reclaim RAM, and workers actively loading models are protected from being killed mid-load. **Open architectural work**: adding a soft-cancel mechanism to MLX or using a persistent model server; adding more physical RAM headroom eliminates this cascade completely.
**Runbook**: `docs/runbooks/mlx-worker-cold-lane-cascade.md`

### F17: Failure-lockdown escalation from expected backpressure

**Cause**: When a bounded background task (such as memory consolidation or reflection) timed out because the main user conversation was using the model, the system treated that normal timeout as a critical failure on a security-sensitive component. This escalated a standard `TimeoutError` into a CRITICAL SERVICE FAILURE, driving `unified_failure_lockdown` toward 1.00.
**Likelihood**: High under heavy load before the fix; Low now.
**Impact**: At a lockdown level of 1.00, Aura blocks memory saving, tool use, and self-updates, triggering false alarms about system integrity.
**Detection**: `unified_failure_lockdown_1.00` in logs, accompanied by `Executive REJECTED` messages for memory and tool actions.
**Recovery**: Fixed — `core/runtime/backpressure.py` now logs expected resource contention on a standard, non-critical channel without triggering lockdown rules. User-facing requests also yield processing time before background tasks run.
**Runbook**: `docs/runbooks/failure-lockdown-from-backpressure.md`

### F18: Launch-provenance `ready:false` on source drift

**Cause**: A signed release build of `Aura.app` is cryptographically tied to a specific Git commit and workspace checksum. Running the app on modified or actively developed code fails this integrity check.
**Likelihood**: Happens on every launch from an active development checkout.
**Impact**: System status reports `ready:false` due to a `launch_provenance` blocker. The app remains fully conversational (via `degraded_ready`). This is an intended security feature indicating code modifications, not a broken system.
**Detection**: Logs show `boot_phase: launch_provenance_failed` with `commit_sha_mismatch` or `workspace_state_sha256_mismatch`.
**Recovery**: Expected during development. To resolve: rebuild and re-sign the application package to update the pinned checksums, or launch via `launch_aura.sh` (which skips strict provenance verification).
**Runbook**: `docs/runbooks/launch-provenance-not-ready.md`

### F19: Quadratic conversation cost from a never-reused prompt cache

**Cause**: The conversation engine was accidentally prevented from reusing cached prompt calculations (Key-Value cache), in two separate ways:
1. `_prompt_cache_entry_budget_for_model` set the Cortex cache size to **0** whenever `desktop_resource_guard_enabled()` was active, meaning the cache was never created on the desktop.
2. Every normal user message included `clean_user_surface_contract=True`, which was listed as a cache-bypass flag—clearing the cache on every turn.
Because the cache was wiped, every turn had to reprocess the entire conversation history from scratch (token 0). Processing time grew with conversation length, making the total conversation cost explode quadratically.
**Likelihood**: Occurred on every long conversation before the fix; now resolved.
**Impact**: Response times climbed steadily (11s → 25s → 105s, etc.) until hitting the turn timeout between turns 8 and 15, causing conversations to fail by turn 20. The model never crashed—turns simply timed out. This was the real cause of the historical "15-turn endurance limit," which had previously been blamed on reasoning degradation.
**Detection**: Steadily increasing per-turn response latency with zero crashes, and a prompt-cache hit count of zero.
**Compounding factors (both recorded in diagnostic forensics)**:
- `JobWatchdog` killed any process that went 90 seconds without emitting a token. Because reprocessing history emits no output tokens until complete, once history reprocessing exceeded 90 seconds, the watchdog killed an entirely healthy worker mid-calculation. Restarting the worker and reloading 20 GB of model files added another two-minute delay.
- During reloads, `_declared_mlx_worker_footprint_gb → _path_size_gb` scanned the entire model directory on the main event thread, blocking the event loop right while 20 GB of files were saturating the hard drive (`data/error_logs/stalls/`).
**Evidence**: `artifacts/closeout/endurance_ceiling/ROOT_CAUSE.md`.
**Runbook**: `docs/runbooks/prompt-cache-never-reused.md`

**The recurring lesson**: A frequent issue in this codebase is *a safety gate discarding a valid answer, which is then misdiagnosed as an infrastructure failure*. When a subsystem seems slow or unresponsive, always check first whether an upstream check is throwing away good work.

### F20: Model lane held by a dead worker's request

**Cause**: A hidden-state read whose caller timed out hands the model lane to the response listener until the worker sends that request's last frame. When the worker died on its own (on 17 October, killed by the memory governor at 85% system memory), the client replaced its queues and spawned a new worker, so the frame could never arrive and the lane stayed held.
**Likelihood**: Low after the fix; before it, whenever memory pressure killed the cortex worker during a background read.
**Impact**: Every chat turn waited twelve seconds for the lane and fell to the fallback ladder, for over half an hour on 17 October.
**Detection**: Repeated `Request queue timeout ... while waiting on model_hidden_features (held N s)` with N growing past the 90 s limit.
**Recovery**: Fixed (8124687e7): the died-unexpectedly branch fails what the gone worker owed and releases a detached request's lane, as the lease-loss path already did.

### F21: Answers cut off before their number

**Cause**: The repetition hold (DRY) pushes down a token that would continue a run copied from her earlier replies. Her tokenizer writes a number as a space token and one digit a token, so arithmetic that restates a number from an earlier answer looked like a copy; the hold also ran inside her private reasoning.
**Likelihood**: High in conversations with repeated arithmetic, before the fix.
**Impact**: Three answers in four were cut off before their number on 17 October ("Total: 245 + 366 +"), and she ended her turn inside her private reasoning before answering.
**Detection**: A reply that stops mid-sentence with `generation performance` showing far fewer tokens than its budget, and no deadline or cancel logged.
**Recovery**: Fixed (37db520c2): a token holding a digit is a sequence breaker, and the hold starts after her private channel closes.

## Recovery Drill Schedule

| Drill | Frequency | Procedure |
|-------|-----------|-----------|
| Backup/restore | Monthly | `make backup && make restore-test` |
| Dirty shutdown recovery | Quarterly | Kill -9 → verify boot |
| Model re-download | Quarterly | Delete model → verify re-acquisition |
| State corruption recovery | Quarterly | Corrupt test DB → verify recovery |
| Full disaster recovery | Annually | Fresh machine → full install → restore |
