# Aura FMEA — failure modes & effects registry

> GENERATED from `core/runtime/fmea.py` by `tools/render_fmea.py`
> (`make fmea-doc`). Do not edit by hand — a drift test regenerates
> and compares this file on every suite run.

Registry version: `1.1` — 25 modes (2 catastrophic, 12 critical, 9 major, 2 minor); 1 open mitigation gap(s), 0 open detection gap(s).

Every entry is REAL: it either occurred live (occurrences cite when)
or is a structurally-reachable state found by analysis. Gaps are
explicit and pinned by an allowlist test that only shrinks.

## FM-LANE-001 — First-token stall under memory pressure → force-kill → cold reload doom loop

- **Subsystem:** core/brain/llm/mlx_client.py (cortex lane)
- **Severity / blast radius:** critical / organism
- **Cause:** Concurrent lanes over-commit host RAM; the resident cortex loses first-token bandwidth; every spawn succeeds so spawn-failure backoff never engages
- **Effect:** Every turn pinned ~200s; each cycle burns a 20GB cold reload; latency unusable while deaths=0 (ladder answers)
- **Detection:** Inference-gate first-token deadline + K4 crash-loop breaker (young-death counting)
- **Mitigation:** K3 declarative lane admission (declared footprints vs host budget); K4 backoff with half-open probe; pressure-adaptive token budgets
- **Detection modules:** `core.runtime.lane_reconciler`, `core.brain.inference_gate`
- **Mitigation modules:** `core.brain.lane_admission`, `core.runtime.lane_reconciler`
- **Recorded occurrences:** 2026-07-07 200-turn soak turns 21-38; 2026-07-08 fullstack-final soak

## FM-LANE-002 — OOM SIGKILL with empty stderr on over-committed spawn (72B solver / fuse beside the resident cortex)

- **Subsystem:** core/brain/llm/mlx_client.py + core/learning/weight_compounding.py
- **Severity / blast radius:** catastrophic / host
- **Cause:** Model load or dequant-fuse transient (~2.5x base) requested beside a committed host; the OS kills the child with no diagnostic
- **Effect:** 20GB worker dies mid-load; 'fuse_failed:' with empty detail; autonomous learning cycle lost
- **Detection:** Lane-admission arithmetic refusal names the breach before the OS can kill; killed_signal classifier names likely_oom after the fact
- **Mitigation:** K3 admission refuses envelope breaches; fuse pre-admission defers with adapter preserved (status 'deferred', operator bypass)
- **Detection modules:** `core.brain.lane_admission`
- **Mitigation modules:** `core.brain.lane_admission`, `core.learning.weight_compounding`
- **Recorded occurrences:** 2026-07-08 live autonomous cycle g0000 fuse OOM (b0b13625)

## FM-LANE-003 — Duplicate heavy runtime: a second cortex spawns beside a wedged one

- **Subsystem:** launcher + core/brain/llm/mlx_client.py
- **Severity / blast radius:** catastrophic / host
- **Cause:** False-death verdict (mind_tick declared dead under load) → launcher respawn without killing the wedged process
- **Effect:** Memory doubling → worse false-death → cascade; host near-exhaustion
- **Detection:** Orphan reclamation scan before spawn (MLXWorker name match); launcher zombie marker
- **Mitigation:** Kill-before-spawn in _spawn_worker_blocking; mind_tick false-death fix (0e87f2c3); headroom-starvation guard
- **Detection modules:** `core.brain.llm.mlx_client`
- **Mitigation modules:** `core.brain.llm.mlx_client`, `core.mind_tick`
- **Recorded occurrences:** 2026-07-06 live degradation cascade (duplicate-runtime memory)

## FM-LANE-004 — Warmup wedge: warmup_in_flight stuck True blocks admission everywhere

- **Subsystem:** core/brain/llm/mlx_client.py (warmup lifecycle)
- **Severity / blast radius:** critical / organism
- **Cause:** Prewarm task dies without its finally-clear; no transition timestamp, so guards trusting client flags defer forever
- **Effect:** 11 straight 240s probe timeouts; conversation admission blocked; cortex gone at 44% RAM with nothing recovering it
- **Detection:** Watchdog-owned dead-man clock (300s grace from first not-alive observation, client flags not trusted); stale-warmup 300s force-clear in warmup()
- **Mitigation:** Dead-man intervention force-clears the wedged flag and bounds the outage to one ~5min window; K1 reconciler heals the lane in the gaps
- **Detection modules:** `core.brain.inference_gate`
- **Mitigation modules:** `core.brain.llm.mlx_client`, `core.runtime.lane_reconciler`
- **Recorded occurrences:** 2026-07-08 nightcap soak turn 28 wedge (59ca3c33)
- **Notes:** Root of the flag-stuck-without-timestamp prewarm death remains unfound; the dead-man clock contains it. See remainder item (11).

## FM-LANE-005 — Gate orphan: timed-out foreground decode holds the generation gate

- **Subsystem:** core/brain/inference_gate.py (generation gate)
- **Severity / blast radius:** critical / lane
- **Cause:** Route timeout abandons a decode that keeps holding the gate; the preemption ladder only soft-cancelled background holders, so the next turn force-aborts the warm worker
- **Effect:** ~5min doom cycle: orphan → 75s wait → force-kill → cold reload → next orphan; 34/38 soak turns dead
- **Detection:** Over-age foreground holder check in the preemption ladder
- **Mitigation:** Abandoned foreground holder gets the soft-cancel rung first (worker stays warm); force-abort reserved for unacknowledged wedges
- **Detection modules:** `core.brain.inference_gate`
- **Mitigation modules:** `core.brain.inference_gate`
- **Recorded occurrences:** 2026-07-08 'final' soak — 34/38 turns dead (7cccb8c3)

## FM-BOOT-001 — 'Booting forever' over a live mind (readiness conflated with liveness)

- **Subsystem:** core/health/boot_status.py + core/runtime/health_contract.py
- **Severity / blast radius:** critical / organism
- **Cause:** A liveness flap (loop-lag spike, important-tier degradation) flipped boot readiness false; the shell re-entered 'booting N%' although the mind was serving
- **Effect:** GUI stuck at 'Connecting to runtime…' / 'booting 48%' for 55 minutes over a fully conversational instance
- **Detection:** K2 probe split: startup latch + independent liveness/readiness verdicts; readiness_coherence daily-driver probe
- **Mitigation:** Post-latch presentation is 'degraded' (runtime_degraded, progress 100) — 'booting' after first readiness is structurally impossible; conversation_operational connects the chat surface on critical-probes-pass
- **Detection modules:** `core.runtime.health_contract`
- **Mitigation modules:** `core.health.boot_status`
- **Recorded occurrences:** 2026-07-06 55min 'booting 48%' live; 2026-07-08 phi-storm liveness flap

## FM-LOOP-001 — Event-loop stall from synchronous I/O on the loop (fsync, per-record SQLite)

- **Subsystem:** logging/persistence on the event loop
- **Severity / blast radius:** critical / organism
- **Cause:** Sync writes inside async def: root-logger file sink, per-record SQLite log handler connect+fsync, goal snapshot query per turn
- **Effect:** 4-6s loop stalls; one 20-minute freeze from an on-loop fsync; liveness flaps cascade from the lag
- **Detection:** Stall watchdog thread dumps (data/error_logs/stalls/); async-write-lane static ratchet fails NEW sync writes at build time
- **Mitigation:** QueueListener logging; batched WAL log writer thread; snapshot caches; file_write_gateway *_async lanes
- **Detection modules:** `core.resilience.stall_watchdog`
- **Mitigation modules:** `core.runtime.file_write_gateway`, `core.runtime.atomic_writer`
- **Recorded occurrences:** 2026-07-06 four distinct live 5-6s stalls (a17a1b56); 20-minute fsync freeze (pre-July)

## FM-LOOP-002 — Loop wedge from unbounded awaits

- **Subsystem:** core/mind_tick.py and long-running loops
- **Severity / blast radius:** critical / organism
- **Cause:** Rhythm-loop awaits (state read, tier recovery) had no timeout; cortex recovery can probe workers for minutes; the loop wedges, frees, re-wedges
- **Effect:** mind_tick visibly dead for 2 hours; ~13GB RAM oscillations; repair machinery unreachable from the wedged loop
- **Detection:** rhythm_stale receipts name the wedged stage; liveness_repair_unreachable is no longer silent; 12 recorded loop-wedge crash dumps fingerprint the class
- **Mitigation:** Both bare awaits bounded (30s/45s) with named tick_stage_timeout degradation; A1 bounded-await static ratchet freezes the class
- **Detection modules:** `core.mind_tick`
- **Mitigation modules:** `core.mind_tick`
- **Recorded occurrences:** 2026-07-07 mind_tick 2-hour death (20fdb6c3)

## FM-LOOP-003 — Hot-path file I/O on every Will decision (8.3s loop lag)

- **Subsystem:** core/runtime service registration hot path
- **Severity / blast radius:** major / organism
- **Cause:** ServiceDescriptor caller determination ran traceback.extract_stack (linecache file reads) + Path.resolve() on the loop for every aura_now re-registration
- **Effect:** Recurring multi-second loop lags; boot stuck at 48% during preflight; the dominant recorded lag source
- **Detection:** SIGUSR1 main-thread sampling caught it live; stall dumps
- **Mitigation:** sys._getframe walk + string slicing + cache (zero I/O, provably filesystem-free contract test); register_instance hot-path upsert
- **Detection modules:** `core.resilience.stall_watchdog`
- **Mitigation modules:** `core.runtime.service_registry`
- **Recorded occurrences:** 2026-07-09 caught live via SIGUSR1 (e422e5de)

## FM-LOOP-004 — Synchronous actuator bridge blocks the owner loop and crosses thread-bound authority

- **Subsystem:** core/actuators/actuator_registry.py + async action callers
- **Severity / blast radius:** critical / organism
- **Cause:** Async callers invoked execute_action(), which moved AuthorityGateway onto a new thread/event loop and synchronously waited; standing child leases are issuing-thread-bound
- **Effect:** The owner event loop stalls for the full action duration, while lease validation or closure can fail on a different thread from issuance
- **Detection:** Owner-thread lifecycle regression plus a behavioral event-loop callback probe; sync bridge rejects active event loops
- **Mitigation:** Canonical execute_action_async keeps authorization, verification, and closure on the owner loop while explicitly blocking actuator bodies run in worker threads
- **Detection modules:** `core.actuators.actuator_registry`
- **Mitigation modules:** `core.actuators.actuator_registry`, `core.runtime.overt_action_loop`, `core.adaptation.immune_executor`
- **Notes:** Structurally reachable in the pre-checkpoint-60 action graph; focused tests reproduce the loop-blocking dependency and thread-ownership mismatch without executing external effects.

## FM-ACTION-001 — Generated prose is parsed as an actuator command

- **Subsystem:** core/consciousness/closed_loop.py output receptor
- **Severity / blast radius:** critical / organism
- **Cause:** The inference callback regex-parsed JSON and function-like text, simulated it, then called ActuatorRegistry directly instead of observing a typed executed-action receipt
- **Effect:** Model narration can mutate world state, bypass canonical action selection, and feed a false success/failure signal back into the substrate
- **Detection:** Negative regression proves action-looking generated text leaves world and substrate unchanged; positive regression accepts only observed action outcomes
- **Mitigation:** OutputReceptor limits generated language to bounded affective feedback; overt action execution sends verified outcomes through notify_closed_loop_action_outcome
- **Detection modules:** `core.consciousness.closed_loop`
- **Mitigation modules:** `core.consciousness.closed_loop`, `core.runtime.overt_action_loop`
- **Notes:** Structurally reachable before checkpoint 60 for any generated JSON containing an actuator field, including actions that were never selected or executed by the canonical agency spine.

## FM-PHI-001 — Pool-child death cascades into a fail-closed CRITICAL storm on thread fallback

- **Subsystem:** hierarchical phi ProcessPool
- **Severity / blast radius:** critical / organism
- **Cause:** A dead pool child demoted compute onto threads (the GIL-bound lag source the pool exists to avoid) while every ~28s cycle recorded CRITICAL
- **Effect:** SLO error budget 20x burn; loop-lag spikes flip liveness; GUI 'Connecting to runtime…' over a live mind
- **Detection:** Pool-rebuild telemetry (rebuild count per process lifetime)
- **Mitigation:** Recovery REBUILDS process isolation first (budget 3/lifetime, telemetry not incident); only a persistently-breaking host demotes to threads ONCE with one degradation record
- **Detection modules:** `core.consciousness.hierarchical_phi`
- **Mitigation modules:** `core.consciousness.hierarchical_phi`
- **Recorded occurrences:** 2026-07-08 live phi-pool storm caught mid-flight (a5e05466)

## FM-MEM-001 — Linear memory growth ~242MB/h under sustained conversation

- **Subsystem:** whole-process RSS
- **Severity / blast radius:** major / host
- **Cause:** UNRESOLVED: H1 real leak vs H2 proof-load-defers-reclamation; tracemalloc instrumentation landed but the discriminating soak has not run
- **Effect:** Multi-hour sessions drift toward pressure eviction; 4h soak FAILs memory while passing lag/queue/boot
- **Detection:** Memory watchdog + sentinel ring + tombstones; soak memory trend milestones
- **Mitigation:** GAP
- **Detection modules:** `core.resilience.memory_watchdog`
- **Recorded occurrences:** 2026-07-07 4h soak memory FAIL
- **Notes:** Blocks A4/K3 fine-tuning. Needs the app-down soak or live RSS trend to discriminate H1 vs H2 — scheduled as the final soak's secondary question.

## FM-FCL-001 — Fail-closed escalation storm: expected backpressure recorded as CRITICAL

- **Subsystem:** core/runtime/errors.py + fail-closed modules (core/config.py list)
- **Severity / blast radius:** critical / organism
- **Cause:** RAM-admission warmup deferrals recorded warning+ degradations on a fail-closed module; escalation raised CRITICAL SERVICE FAILURE out of the handler; policy then disabled the cloud lane
- **Effect:** The 210s-503 anatomy: one deferral cascades into protected-lane failure and user-visible 503s
- **Detection:** SLO error-events budget burn; degradation classifier severity histogram
- **Mitigation:** Backpressure classified info-level (persistent/total conditions only become degradations); cloud-SDK error tuple resolved before try; A4 escalation-rate cap
- **Detection modules:** `core.runtime.telemetry_sli`
- **Mitigation modules:** `core.runtime.errors`
- **Recorded occurrences:** 2026-07-08 ac5a222e live anatomy

## FM-CHAT-001 — Raw HTTP 503 delivered to a real user

- **Subsystem:** api chat route + fail-closed reply paths
- **Severity / blast radius:** major / turn
- **Cause:** Fail-closed reply and memory-guard paths returned transport-level 503 instead of an honest in-band body
- **Effect:** The desktop shell drops to 'Connecting to runtime…' mid-conversation
- **Detection:** Endurance-probe turn classification (503 vs honest body)
- **Mitigation:** 200-with-honest-body for real users; benchmarks keep 503 via X-Aura-Benchmark (foreground_busy precedent)
- **Recorded occurrences:** 2026-07-08 nightcap turns 24-25 (unswept path)
- **Notes:** Sweep of remaining unconditional-503 producers on /api/chat is remainder item (10); verify during this pass's C-phase.

## FM-QUAL-001 — Quality-gate exhaustion loop delivers nothing

- **Subsystem:** response quality gates
- **Severity / blast radius:** major / turn
- **Cause:** Drafts repeatedly failing surface gates (self-claim evidence boundary, requested-phrase) burned all retries and returned empty
- **Effect:** 56s turns ending in empty_cognitive_engine_reply; user sees silence
- **Detection:** Incident narrator episodes (it diagnosed this class live)
- **Mitigation:** Exhaustion salvage: deliver the best honest draft (self-claim guard self-heals via evidence-boundary suffix; leaks stay fail-closed); surface-gate retry wall (AURA_SURFACE_RETRY_WALL_S)
- **Detection modules:** `core.observability.incident_narrator`
- **Recorded occurrences:** 2026-07-07 consciousness-question loop caught by narrator (70695ff0)

## FM-DISP-001 — Trigger overbreadth hijacks conversation

- **Subsystem:** skill/action dispatch triggers
- **Severity / blast radius:** major / turn
- **Cause:** Normalizer mangling ('really'→'recall') and all-optional-tail regexes ('paint (?:me )?(?:an? )?') routed casual words to heavy skills
- **Effect:** Casual sentences dispatched memory_ops/diffusion; one crashed CRITICAL (generic dispatch passes query, ImageGenInput demands prompt)
- **Detection:** 41-sentence benign + 8-positive permanent ratchet test
- **Mitigation:** Normalizer and regex fixes at the root; the ratchet freezes the class
- **Recorded occurrences:** 2026-07-08 live hijacks (fc273e37, c7b7f510)

## FM-WORK-001 — Zero-token generation (immediate EOS) from stale KV cache

- **Subsystem:** mlx_worker KV/prompt cache
- **Severity / blast radius:** minor / turn
- **Cause:** Cached KV disagreeing with a fresh prompt yields EOS on the first step
- **Effect:** Empty generation; retry needed; boot-window warnings
- **Detection:** token_count telemetry (tokens actually emitted)
- **Mitigation:** Worker self-heals by nuking stale prompt-cache KV + Metal cache
- **Recorded occurrences:** 2026-07-08 boot-window occurrences investigated, benign
- **Notes:** Self-healing; only the FREQUENCY under load is worth watching.

## FM-SUP-001 — Duplicate process supervisors restart or stop the same runtime independently

- **Subsystem:** core/supervisor/tree.py + core/runtime/organ_supervisor.py
- **Severity / blast radius:** critical / organism
- **Cause:** Launcher, orchestrator, actor tree, and command-organ watchdogs each carried their own singleton and restart budget, allowing ownership and circuit state to diverge
- **Effect:** Duplicate children, conflicting restart loops, lost IPC rebinding, or shutdown that reports complete while another supervisor keeps a child alive
- **Detection:** Operator control-plane report names one actor monitor, desired/observed child state, open circuits, and duplicate live-contract registration
- **Mitigation:** aura_main and orchestrator resolve one SupervisionTree singleton; the tree is a managed control-plane service; command organs delegate retries/backoff/circuits to RuntimeControlPlane and retain transport only
- **Detection modules:** `core.runtime.operator_control_plane`, `core.runtime.control_plane`
- **Mitigation modules:** `core.supervisor.tree`, `core.runtime.organ_supervisor`, `core.runtime.control_plane`
- **Notes:** Structurally reachable in the pre-unification call graph: aura_main constructed a separate SupervisionTree while OrganSupervisor owned an independent watchdog.

## FM-AUD-001 — High-frequency admission decisions exhaust filesystem inodes and process memory

- **Subsystem:** core/runtime/receipts.py + resource admission audit
- **Severity / blast radius:** major / host
- **Cause:** Every pressure deferral used one durable JSON file and stayed in the in-memory receipt index, even when the same state repeated indefinitely
- **Effect:** Long-running hosts accumulate unbounded files and index entries; diagnostics and restart reload become progressively slower until auditability harms availability
- **Detection:** Receipt storage stats expose hot-index limits, ledger availability, persistent counts, and admission coalescing counters in the operator report
- **Mitigation:** Every receipt kind uses bounded immutable hot snapshots with durable cold lookup; high-volume admission receipts use one WAL-backed ledger, and unchanged unaudited denials persist on transition and periodic heartbeat while audited requests remain one receipt per attempt
- **Detection modules:** `core.runtime.receipts`, `core.runtime.operator_control_plane`
- **Mitigation modules:** `core.runtime.receipts`, `core.runtime.control_plane`
- **Notes:** Structural long-run analysis from Pass F; no finite soak can prove unbounded per-event file creation safe over indefinite daily operation.

## FM-FORENSICS-001 — Hard death (SIGKILL / OOM-kill) leaves no record of the final moments

- **Subsystem:** whole-process death forensics
- **Severity / blast radius:** major / organism
- **Cause:** faulthandler and every in-process hook are uncatchable on SIGKILL; the continuity record is written BEFORE the death, so its shutdown reason is stale optimism; post-mortem analysis reconstructs from inference, not evidence
- **Effect:** Endurance OOMs and launcher kills were diagnosed from syslogs and memory sentinel side-channels; what the mind was doing in its final seconds was unknowable (2026-07-03 kernel-down, 2026-07-06 duplicate-runtime cascade)
- **Detection:** A5 flight recorder: absent clean-shutdown marker in the mmap ring = hard death, detected at next boot with the last recorded mind-moments
- **Mitigation:** Kernel-owned MAP_SHARED pages survive any process death; per-tick frames (stage, RSS, conditions, failures) extracted into a governed death report consumed by the incident narrator and the continuity waking sequence
- **Detection modules:** `core.runtime.flight_recorder`
- **Mitigation modules:** `core.runtime.flight_recorder`, `core.observability.incident_narrator`
- **Recorded occurrences:** 2026-07-03 endurance OOM 35GB (no relaunch, no final-moment record); 2026-07-06 duplicate-runtime cascade (diagnosed from side-channels)
- **Notes:** The ring is written by the death itself; only whole-machine loss can erase it.

## FM-LEARN-001 — Curriculum misdirection: stale, corrupt, or skewed failure evidence steers practice at the wrong domains

- **Subsystem:** core/learning/deliberate_practice.py (practice director)
- **Severity / blast radius:** minor / lane
- **Cause:** Ledger corruption, a burst of unrepresentative failures, or receipts from an old model generation dominating the decayed ranking
- **Effect:** Idle practice and specialist training drill domains that no longer need it — the waste of uniform practice returns, never worse than it (all consumers keep their uniform/least-recently-trained fallback)
- **Detection:** Per-domain curriculum is receipts-attached and served on /api/system/learning; the learning self-report states the direction in chat; mastery rail zeroes any domain holding ≥95%
- **Mitigation:** 7-day evidence half-life ages out stale failures; corrupt ledger lines are skipped (never void the ledger); AURA_DELIBERATE_PRACTICE=0 kill switch restores uniform practice instantly; observation intake never blocks practice
- **Detection modules:** `core.learning.deliberate_practice`
- **Mitigation modules:** `core.learning.deliberate_practice`
- **Notes:** Direction is quality-of-learning only: the two-sided specialist gate and the sealed compounding gate still decide what ships, so misdirection can waste idle compute but cannot promote a regression.

## FM-COG-001 — Broadcast monopoly or cartel: the workspace stops rotating in steady state

- **Subsystem:** core/consciousness/global_workspace.py (competition)
- **Severity / blast radius:** critical / organism
- **Cause:** Refractory policy that excludes losers, or adaptation whose recovery rate is fixed. A leaky integrator with gain g and fixed decay r pins each source's sustainable share at r/g, so exactly g/r sources can rotate however many bid
- **Effect:** No global workspace, only a sort. Every downstream consumer of winners — soul.py, the context stream, ignition-gated cognition — reads a broadcast that one or two producers hold permanently
- **Detection:** Steady-state competition health over a trajectory: no_starvation and rotation_entropy against order_preserving, plus no_limit_cycle for the a-b-a-b alternation that scores as a perfect rotation on entropy alone
- **Mitigation:** Adaptation recovery derived as r = g/n from the size of the field, making the equilibrium share 1/n; the lone-source and urgent-vs-idle regimes fall out rather than needing special cases
- **Detection modules:** `core.verify.dynamics`
- **Mitigation modules:** `core.consciousness.global_workspace`
- **Recorded occurrences:** 2026-08-12 monopoly: 24/24 wins to one source of four, past 46 contract tests; 2026-08-12 duopoly: 12/12 split by the top two, past the monopoly fix's own test
- **Notes:** Both states passed the assertions written against the previous one. top_share < 0.75 with two distinct winners is true of a perfect duopoly, which is why the check is now a property over a trajectory rather than a bound on a symptom.

## FM-COG-002 — Decisions settled by submission timing while presented as priority

- **Subsystem:** core/consciousness/global_workspace.py (arbitration)
- **Severity / blast radius:** major / lane
- **Cause:** effective_priority scales salience by (1 - 0.03*age), so candidates of identical salience differ by ~0.03*spread purely from when they were submitted
- **Effect:** Arbitrary arbitration that looks principled. Measured: four sources bidding an identical 0.70 produced 0 exact ties in 12 ticks and were separated by ~2e-6
- **Detection:** Tie impasses counted against the timing noise floor that mechanism creates rather than against exact equality; surfaced in get_snapshot()
- **Mitigation:** _resolve_tie decides among indistinguishable bids on two rules that cannot see arrival time: least-fatigued first (the same quantity arbitration already uses, so it hands the slot to whoever waited longest), then rotation on the tick index when fatigue is level too
- **Detection modules:** `core.consciousness.global_workspace`, `core.cognition.impasse`
- **Mitigation modules:** `core.consciousness.global_workspace`
- **Recorded occurrences:** 2026-08-12 identical-bid probe: 3 tie impasses in 12 ticks
- **Notes:** Closed. Four sources bidding an identical 0.70 over 48 ticks now take 0.25 each, and the distribution is byte-identical whether they are submitted in a fixed order or a rotating one — which is the actual property, since arrival order used to be the whole input. Rotation rather than a fixed order matters: a stable sort would hand every genuine tie to whichever source sorts first, forever.

## FM-COG-003 — A ranking term silently decays to a constant

- **Subsystem:** core/memory/episodic_memory.py (recall ranking)
- **Severity / blast radius:** major / organism
- **Cause:** Scoring keyed to an absolute wall-clock epoch rather than elapsed time; the usable window recedes as real time advances until every live input saturates
- **Effect:** Recall ranked by importance alone. Measured on 2026-08-12, every episode newer than 2026-04-12 scored exactly 1.000000 — one minute and thirty days indistinguishable — so the term contributed a constant 0.4 to every candidate
- **Detection:** Scale-freedom: identical relative histories must score identically at any wall clock, and distinct ages must produce distinct scores across the operating range
- **Mitigation:** ACT-R base-level activation B = ln(sum t_j^-d), a function of elapsed time only, so it cannot saturate and has no epoch to go stale; also reads frequency
- **Detection modules:** `core.cognition.actr_activation`
- **Mitigation modules:** `core.cognition.actr_activation`, `core.memory.episodic_memory`
- **Recorded occurrences:** 2026-08-12 recency scorer measured flat across the whole live range
- **Notes:** Time-bomb class: correct when written, degrades with the calendar, and no test that does not move the clock can see it. The detector moves the clock.

## Open gaps (the work queue)

- **FM-MEM-001** (mitigation gap): Linear memory growth ~242MB/h under sustained conversation
