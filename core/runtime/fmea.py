"""core/runtime/fmea.py — the failure-mode & effects registry (roadmap A2).

Aerospace discipline applied to Aura: every known way the organism fails,
enumerated in one machine-readable place — mode → cause → effect → blast
radius → detection → mitigation. Until now this knowledge lived in soak
artifacts, commit messages, and the improvement-pass memory; a new watchdog
was added only after each failure was *lived through*. The registry inverts
that: coverage is inspectable, gaps are explicit and ratcheted, and the
reconciler/probe/isolation work is driven by enumeration instead of by the
next incident.

Rules:
  * Entries are REAL. Every failure mode here either occurred live (the
    `occurrences` field cites when) or is a structurally-reachable state
    found by analysis. No template filler.
  * `detection_modules` / `mitigation_modules` name importable modules; the
    suite imports each one, so an entry cannot reference a control that was
    deleted or renamed (stale-mitigation drift fails the build).
  * A mode with NO working detection or mitigation must say so:
    detection="GAP" / mitigation="GAP". Gaps are pinned by an explicit
    allowlist test — adding a new gap is a conscious act, and closing one
    shrinks the list forever.
  * `docs/FMEA.md` is GENERATED from this module (tools/render_fmea.py,
    `make fmea-doc`); a drift test keeps it honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BlastRadius(StrEnum):
    TURN = "turn"          # one user turn degrades or fails
    LANE = "lane"          # one model lane / capability lost until recovery
    ORGANISM = "organism"  # whole-runtime degradation (latency, liveness)
    HOST = "host"          # host-level damage (OOM, memory exhaustion)


class Severity(StrEnum):
    CATASTROPHIC = "catastrophic"  # runtime death or host exhaustion
    CRITICAL = "critical"          # sustained unusable experience
    MAJOR = "major"                # degraded turns / lost capability
    MINOR = "minor"                # cosmetic or self-healing


@dataclass(frozen=True)
class FailureMode:
    id: str
    subsystem: str
    mode: str
    cause: str
    effect: str
    blast_radius: BlastRadius
    severity: Severity
    detection: str
    mitigation: str
    detection_modules: tuple[str, ...] = ()
    mitigation_modules: tuple[str, ...] = ()
    occurrences: tuple[str, ...] = ()
    notes: str = ""


FMEA_VERSION = "1.1"

FMEA_REGISTRY: tuple[FailureMode, ...] = (
    # ── Model-serving lane ─────────────────────────────────────────
    FailureMode(
        id="FM-LANE-001",
        subsystem="core/brain/llm/mlx_client.py (cortex lane)",
        mode="First-token stall under memory pressure → force-kill → cold reload doom loop",
        cause="Concurrent lanes over-commit host RAM; the resident cortex loses first-token "
        "bandwidth; every spawn succeeds so spawn-failure backoff never engages",
        effect="Every turn pinned ~200s; each cycle burns a 20GB cold reload; latency unusable "
        "while deaths=0 (ladder answers)",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="Inference-gate first-token deadline + K4 crash-loop breaker (young-death counting)",
        mitigation="K3 declarative lane admission (declared footprints vs host budget); K4 backoff "
        "with half-open probe; pressure-adaptive token budgets",
        detection_modules=("core.runtime.lane_reconciler", "core.brain.inference_gate"),
        mitigation_modules=("core.brain.lane_admission", "core.runtime.lane_reconciler"),
        occurrences=("2026-07-07 200-turn soak turns 21-38", "2026-07-08 fullstack-final soak"),
    ),
    FailureMode(
        id="FM-LANE-002",
        subsystem="core/brain/llm/mlx_client.py + core/learning/weight_compounding.py",
        mode="OOM SIGKILL with empty stderr on over-committed spawn (72B solver / fuse beside the resident cortex)",
        cause="Model load or dequant-fuse transient (~2.5x base) requested beside a committed host; "
        "the OS kills the child with no diagnostic",
        effect="20GB worker dies mid-load; 'fuse_failed:' with empty detail; autonomous learning "
        "cycle lost",
        blast_radius=BlastRadius.HOST,
        severity=Severity.CATASTROPHIC,
        detection="Lane-admission arithmetic refusal names the breach before the OS can kill; "
        "killed_signal classifier names likely_oom after the fact",
        mitigation="K3 admission refuses envelope breaches; fuse pre-admission defers with adapter "
        "preserved (status 'deferred', operator bypass)",
        detection_modules=("core.brain.lane_admission",),
        mitigation_modules=("core.brain.lane_admission", "core.learning.weight_compounding"),
        occurrences=("2026-07-08 live autonomous cycle g0000 fuse OOM (b0b13625)",),
    ),
    FailureMode(
        id="FM-LANE-003",
        subsystem="launcher + core/brain/llm/mlx_client.py",
        mode="Duplicate heavy runtime: a second cortex spawns beside a wedged one",
        cause="False-death verdict (mind_tick declared dead under load) → launcher respawn "
        "without killing the wedged process",
        effect="Memory doubling → worse false-death → cascade; host near-exhaustion",
        blast_radius=BlastRadius.HOST,
        severity=Severity.CATASTROPHIC,
        detection="Orphan reclamation scan before spawn (MLXWorker name match); launcher zombie marker",
        mitigation="Kill-before-spawn in _spawn_worker_blocking; mind_tick false-death fix "
        "(0e87f2c3); headroom-starvation guard",
        detection_modules=("core.brain.llm.mlx_client",),
        mitigation_modules=("core.brain.llm.mlx_client", "core.mind_tick"),
        occurrences=("2026-07-06 live degradation cascade (duplicate-runtime memory)",),
    ),
    FailureMode(
        id="FM-LANE-004",
        subsystem="core/brain/llm/mlx_client.py (warmup lifecycle)",
        mode="Warmup wedge: warmup_in_flight stuck True blocks admission everywhere",
        cause="Prewarm task dies without its finally-clear; no transition timestamp, so guards "
        "trusting client flags defer forever",
        effect="11 straight 240s probe timeouts; conversation admission blocked; cortex gone at "
        "44% RAM with nothing recovering it",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="Watchdog-owned dead-man clock (300s grace from first not-alive observation, "
        "client flags not trusted); stale-warmup 300s force-clear in warmup()",
        mitigation="Dead-man intervention force-clears the wedged flag and bounds the outage to "
        "one ~5min window; K1 reconciler heals the lane in the gaps",
        detection_modules=("core.brain.inference_gate",),
        mitigation_modules=("core.brain.llm.mlx_client", "core.runtime.lane_reconciler"),
        occurrences=("2026-07-08 nightcap soak turn 28 wedge (59ca3c33)",),
        notes="Root of the flag-stuck-without-timestamp prewarm death remains unfound; the "
        "dead-man clock contains it. See remainder item (11).",
    ),
    FailureMode(
        id="FM-LANE-005",
        subsystem="core/brain/inference_gate.py (generation gate)",
        mode="Gate orphan: timed-out foreground decode holds the generation gate",
        cause="Route timeout abandons a decode that keeps holding the gate; the preemption ladder "
        "only soft-cancelled background holders, so the next turn force-aborts the warm worker",
        effect="~5min doom cycle: orphan → 75s wait → force-kill → cold reload → next orphan; "
        "34/38 soak turns dead",
        blast_radius=BlastRadius.LANE,
        severity=Severity.CRITICAL,
        detection="Over-age foreground holder check in the preemption ladder",
        mitigation="Abandoned foreground holder gets the soft-cancel rung first (worker stays "
        "warm); force-abort reserved for unacknowledged wedges",
        detection_modules=("core.brain.inference_gate",),
        mitigation_modules=("core.brain.inference_gate",),
        occurrences=("2026-07-08 'final' soak — 34/38 turns dead (7cccb8c3)",),
    ),
    # ── Boot / readiness ───────────────────────────────────────────
    FailureMode(
        id="FM-BOOT-001",
        subsystem="core/health/boot_status.py + core/runtime/health_contract.py",
        mode="'Booting forever' over a live mind (readiness conflated with liveness)",
        cause="A liveness flap (loop-lag spike, important-tier degradation) flipped boot "
        "readiness false; the shell re-entered 'booting N%' although the mind was serving",
        effect="GUI stuck at 'Connecting to runtime…' / 'booting 48%' for 55 minutes over a "
        "fully conversational instance",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="K2 probe split: startup latch + independent liveness/readiness verdicts; "
        "readiness_coherence daily-driver probe",
        mitigation="Post-latch presentation is 'degraded' (runtime_degraded, progress 100) — "
        "'booting' after first readiness is structurally impossible; conversation_operational "
        "connects the chat surface on critical-probes-pass",
        detection_modules=("core.runtime.health_contract",),
        mitigation_modules=("core.health.boot_status",),
        occurrences=("2026-07-06 55min 'booting 48%' live", "2026-07-08 phi-storm liveness flap"),
    ),
    # ── Event loop ─────────────────────────────────────────────────
    FailureMode(
        id="FM-LOOP-001",
        subsystem="logging/persistence on the event loop",
        mode="Event-loop stall from synchronous I/O on the loop (fsync, per-record SQLite)",
        cause="Sync writes inside async def: root-logger file sink, per-record SQLite log "
        "handler connect+fsync, goal snapshot query per turn",
        effect="4-6s loop stalls; one 20-minute freeze from an on-loop fsync; liveness flaps "
        "cascade from the lag",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="Stall watchdog thread dumps (data/error_logs/stalls/); async-write-lane "
        "static ratchet fails NEW sync writes at build time",
        mitigation="QueueListener logging; batched WAL log writer thread; snapshot caches; "
        "file_write_gateway *_async lanes",
        detection_modules=("core.resilience.stall_watchdog",),
        mitigation_modules=("core.runtime.file_write_gateway", "core.runtime.atomic_writer"),
        occurrences=("2026-07-06 four distinct live 5-6s stalls (a17a1b56)", "20-minute fsync freeze (pre-July)"),
    ),
    FailureMode(
        id="FM-LOOP-002",
        subsystem="core/mind_tick.py and long-running loops",
        mode="Loop wedge from unbounded awaits",
        cause="Rhythm-loop awaits (state read, tier recovery) had no timeout; cortex recovery "
        "can probe workers for minutes; the loop wedges, frees, re-wedges",
        effect="mind_tick visibly dead for 2 hours; ~13GB RAM oscillations; repair machinery "
        "unreachable from the wedged loop",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="rhythm_stale receipts name the wedged stage; liveness_repair_unreachable "
        "is no longer silent; 12 recorded loop-wedge crash dumps fingerprint the class",
        mitigation="Both bare awaits bounded (30s/45s) with named tick_stage_timeout degradation; "
        "A1 bounded-await static ratchet freezes the class",
        detection_modules=("core.mind_tick",),
        mitigation_modules=("core.mind_tick",),
        occurrences=("2026-07-07 mind_tick 2-hour death (20fdb6c3)",),
    ),
    FailureMode(
        id="FM-LOOP-003",
        subsystem="core/runtime service registration hot path",
        mode="Hot-path file I/O on every Will decision (8.3s loop lag)",
        cause="ServiceDescriptor caller determination ran traceback.extract_stack (linecache "
        "file reads) + Path.resolve() on the loop for every aura_now re-registration",
        effect="Recurring multi-second loop lags; boot stuck at 48% during preflight; the "
        "dominant recorded lag source",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.MAJOR,
        detection="SIGUSR1 main-thread sampling caught it live; stall dumps",
        mitigation="sys._getframe walk + string slicing + cache (zero I/O, provably "
        "filesystem-free contract test); register_instance hot-path upsert",
        detection_modules=("core.resilience.stall_watchdog",),
        mitigation_modules=("core.runtime.service_registry",),
        occurrences=("2026-07-09 caught live via SIGUSR1 (e422e5de)",),
    ),
    FailureMode(
        id="FM-LOOP-004",
        subsystem="core/actuators/actuator_registry.py + async action callers",
        mode="Synchronous actuator bridge blocks the owner loop and crosses thread-bound authority",
        cause="Async callers invoked execute_action(), which moved AuthorityGateway onto a new "
        "thread/event loop and synchronously waited; standing child leases are issuing-thread-bound",
        effect="The owner event loop stalls for the full action duration, while lease validation or "
        "closure can fail on a different thread from issuance",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="Owner-thread lifecycle regression plus a behavioral event-loop callback probe; "
        "sync bridge rejects active event loops",
        mitigation="Canonical execute_action_async keeps authorization, verification, and closure on "
        "the owner loop while explicitly blocking actuator bodies run in worker threads",
        detection_modules=("core.actuators.actuator_registry",),
        mitigation_modules=(
            "core.actuators.actuator_registry",
            "core.runtime.overt_action_loop",
            "core.adaptation.immune_executor",
        ),
        notes="Structurally reachable in the pre-checkpoint-60 action graph; focused tests reproduce "
        "the loop-blocking dependency and thread-ownership mismatch without executing external effects.",
    ),
    FailureMode(
        id="FM-ACTION-001",
        subsystem="core/consciousness/closed_loop.py output receptor",
        mode="Generated prose is parsed as an actuator command",
        cause="The inference callback regex-parsed JSON and function-like text, simulated it, then "
        "called ActuatorRegistry directly instead of observing a typed executed-action receipt",
        effect="Model narration can mutate world state, bypass canonical action selection, and feed a "
        "false success/failure signal back into the substrate",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="Negative regression proves action-looking generated text leaves world and substrate "
        "unchanged; positive regression accepts only observed action outcomes",
        mitigation="OutputReceptor limits generated language to bounded affective feedback; overt action "
        "execution sends verified outcomes through notify_closed_loop_action_outcome",
        detection_modules=("core.consciousness.closed_loop",),
        mitigation_modules=(
            "core.consciousness.closed_loop",
            "core.runtime.overt_action_loop",
        ),
        notes="Structurally reachable before checkpoint 60 for any generated JSON containing an actuator "
        "field, including actions that were never selected or executed by the canonical agency spine.",
    ),
    # ── Phi / consciousness compute ────────────────────────────────
    FailureMode(
        id="FM-PHI-001",
        subsystem="hierarchical phi ProcessPool",
        mode="Pool-child death cascades into a fail-closed CRITICAL storm on thread fallback",
        cause="A dead pool child demoted compute onto threads (the GIL-bound lag source the "
        "pool exists to avoid) while every ~28s cycle recorded CRITICAL",
        effect="SLO error budget 20x burn; loop-lag spikes flip liveness; GUI 'Connecting to "
        "runtime…' over a live mind",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="Pool-rebuild telemetry (rebuild count per process lifetime)",
        mitigation="Recovery REBUILDS process isolation first (budget 3/lifetime, telemetry not "
        "incident); only a persistently-breaking host demotes to threads ONCE with one "
        "degradation record",
        detection_modules=("core.consciousness.hierarchical_phi",),
        mitigation_modules=("core.consciousness.hierarchical_phi",),
        occurrences=("2026-07-08 live phi-pool storm caught mid-flight (a5e05466)",),
    ),
    # ── Memory (host RAM) ──────────────────────────────────────────
    FailureMode(
        id="FM-MEM-001",
        subsystem="whole-process RSS",
        mode="Linear memory growth ~242MB/h under sustained conversation",
        cause="UNRESOLVED: H1 real leak vs H2 proof-load-defers-reclamation; tracemalloc "
        "instrumentation landed but the discriminating soak has not run",
        effect="Multi-hour sessions drift toward pressure eviction; 4h soak FAILs memory "
        "while passing lag/queue/boot",
        blast_radius=BlastRadius.HOST,
        severity=Severity.MAJOR,
        detection="Memory watchdog + sentinel ring + tombstones; soak memory trend milestones",
        mitigation="GAP",
        detection_modules=("core.resilience.memory_watchdog",),
        mitigation_modules=(),
        occurrences=("2026-07-07 4h soak memory FAIL",),
        notes="Blocks A4/K3 fine-tuning. Needs the app-down soak or live RSS trend to "
        "discriminate H1 vs H2 — scheduled as the final soak's secondary question.",
    ),
    # ── Fail-closed / escalation policy ────────────────────────────
    FailureMode(
        id="FM-FCL-001",
        subsystem="core/runtime/errors.py + fail-closed modules (core/config.py list)",
        mode="Fail-closed escalation storm: expected backpressure recorded as CRITICAL",
        cause="RAM-admission warmup deferrals recorded warning+ degradations on a fail-closed "
        "module; escalation raised CRITICAL SERVICE FAILURE out of the handler; policy then "
        "disabled the cloud lane",
        effect="The 210s-503 anatomy: one deferral cascades into protected-lane failure and "
        "user-visible 503s",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="SLO error-events budget burn; degradation classifier severity histogram",
        mitigation="Backpressure classified info-level (persistent/total conditions only become "
        "degradations); cloud-SDK error tuple resolved before try; A4 escalation-rate cap",
        detection_modules=("core.runtime.telemetry_sli",),
        mitigation_modules=("core.runtime.errors",),
        occurrences=("2026-07-08 ac5a222e live anatomy",),
    ),
    # ── Chat surface ───────────────────────────────────────────────
    FailureMode(
        id="FM-CHAT-001",
        subsystem="api chat route + fail-closed reply paths",
        mode="Raw HTTP 503 delivered to a real user",
        cause="Fail-closed reply and memory-guard paths returned transport-level 503 instead "
        "of an honest in-band body",
        effect="The desktop shell drops to 'Connecting to runtime…' mid-conversation",
        blast_radius=BlastRadius.TURN,
        severity=Severity.MAJOR,
        detection="Endurance-probe turn classification (503 vs honest body)",
        mitigation="200-with-honest-body for real users; benchmarks keep 503 via "
        "X-Aura-Benchmark (foreground_busy precedent)",
        detection_modules=(),
        mitigation_modules=(),
        occurrences=("2026-07-08 nightcap turns 24-25 (unswept path)",),
        notes="Sweep of remaining unconditional-503 producers on /api/chat is remainder "
        "item (10); verify during this pass's C-phase.",
    ),
    FailureMode(
        id="FM-QUAL-001",
        subsystem="response quality gates",
        mode="Quality-gate exhaustion loop delivers nothing",
        cause="Drafts repeatedly failing surface gates (self-claim evidence boundary, "
        "requested-phrase) burned all retries and returned empty",
        effect="56s turns ending in empty_cognitive_engine_reply; user sees silence",
        blast_radius=BlastRadius.TURN,
        severity=Severity.MAJOR,
        detection="Incident narrator episodes (it diagnosed this class live)",
        mitigation="Exhaustion salvage: deliver the best honest draft (self-claim guard "
        "self-heals via evidence-boundary suffix; leaks stay fail-closed); surface-gate "
        "retry wall (AURA_SURFACE_RETRY_WALL_S)",
        detection_modules=("core.observability.incident_narrator",),
        mitigation_modules=(),
        occurrences=("2026-07-07 consciousness-question loop caught by narrator (70695ff0)",),
    ),
    FailureMode(
        id="FM-DISP-001",
        subsystem="skill/action dispatch triggers",
        mode="Trigger overbreadth hijacks conversation",
        cause="Normalizer mangling ('really'→'recall') and all-optional-tail regexes ('paint "
        "(?:me )?(?:an? )?') routed casual words to heavy skills",
        effect="Casual sentences dispatched memory_ops/diffusion; one crashed CRITICAL "
        "(generic dispatch passes query, ImageGenInput demands prompt)",
        blast_radius=BlastRadius.TURN,
        severity=Severity.MAJOR,
        detection="41-sentence benign + 8-positive permanent ratchet test",
        mitigation="Normalizer and regex fixes at the root; the ratchet freezes the class",
        detection_modules=(),
        mitigation_modules=(),
        occurrences=("2026-07-08 live hijacks (fc273e37, c7b7f510)",),
    ),
    FailureMode(
        id="FM-WORK-001",
        subsystem="mlx_worker KV/prompt cache",
        mode="Zero-token generation (immediate EOS) from stale KV cache",
        cause="Cached KV disagreeing with a fresh prompt yields EOS on the first step",
        effect="Empty generation; retry needed; boot-window warnings",
        blast_radius=BlastRadius.TURN,
        severity=Severity.MINOR,
        detection="token_count telemetry (tokens actually emitted)",
        mitigation="Worker self-heals by nuking stale prompt-cache KV + Metal cache",
        detection_modules=(),
        mitigation_modules=(),
        occurrences=("2026-07-08 boot-window occurrences investigated, benign",),
        notes="Self-healing; only the FREQUENCY under load is worth watching.",
    ),
    FailureMode(
        id="FM-SUP-001",
        subsystem="core/supervisor/tree.py + core/runtime/organ_supervisor.py",
        mode="Duplicate process supervisors restart or stop the same runtime independently",
        cause="Launcher, orchestrator, actor tree, and command-organ watchdogs each carried "
        "their own singleton and restart budget, allowing ownership and circuit state to diverge",
        effect="Duplicate children, conflicting restart loops, lost IPC rebinding, or shutdown "
        "that reports complete while another supervisor keeps a child alive",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="Operator control-plane report names one actor monitor, desired/observed child "
        "state, open circuits, and duplicate live-contract registration",
        mitigation="aura_main and orchestrator resolve one SupervisionTree singleton; the tree is "
        "a managed control-plane service; command organs delegate retries/backoff/circuits to "
        "RuntimeControlPlane and retain transport only",
        detection_modules=(
            "core.runtime.operator_control_plane",
            "core.runtime.control_plane",
        ),
        mitigation_modules=(
            "core.supervisor.tree",
            "core.runtime.organ_supervisor",
            "core.runtime.control_plane",
        ),
        notes="Structurally reachable in the pre-unification call graph: aura_main constructed a "
        "separate SupervisionTree while OrganSupervisor owned an independent watchdog.",
    ),
    FailureMode(
        id="FM-AUD-001",
        subsystem="core/runtime/receipts.py + resource admission audit",
        mode="High-frequency admission decisions exhaust filesystem inodes and process memory",
        cause="Every pressure deferral used one durable JSON file and stayed in the in-memory "
        "receipt index, even when the same state repeated indefinitely",
        effect="Long-running hosts accumulate unbounded files and index entries; diagnostics and "
        "restart reload become progressively slower until auditability harms availability",
        blast_radius=BlastRadius.HOST,
        severity=Severity.MAJOR,
        detection="Receipt storage stats expose hot-index limits, ledger availability, persistent "
        "counts, and admission coalescing counters in the operator report",
        mitigation="Every receipt kind uses bounded immutable hot snapshots with durable cold "
        "lookup; high-volume admission receipts use one WAL-backed ledger, and unchanged "
        "unaudited denials persist on transition and periodic heartbeat while audited requests "
        "remain one receipt per attempt",
        detection_modules=(
            "core.runtime.receipts",
            "core.runtime.operator_control_plane",
        ),
        mitigation_modules=(
            "core.runtime.receipts",
            "core.runtime.control_plane",
        ),
        notes="Structural long-run analysis from Pass F; no finite soak can prove unbounded "
        "per-event file creation safe over indefinite daily operation.",
    ),
    # ── Forensics coverage ─────────────────────────────────────────
    FailureMode(
        id="FM-FORENSICS-001",
        subsystem="whole-process death forensics",
        mode="Hard death (SIGKILL / OOM-kill) leaves no record of the final moments",
        cause="faulthandler and every in-process hook are uncatchable on SIGKILL; the "
        "continuity record is written BEFORE the death, so its shutdown reason is stale "
        "optimism; post-mortem analysis reconstructs from inference, not evidence",
        effect="Endurance OOMs and launcher kills were diagnosed from syslogs and memory "
        "sentinel side-channels; what the mind was doing in its final seconds was "
        "unknowable (2026-07-03 kernel-down, 2026-07-06 duplicate-runtime cascade)",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.MAJOR,
        detection="A5 flight recorder: absent clean-shutdown marker in the mmap ring = hard "
        "death, detected at next boot with the last recorded mind-moments",
        mitigation="Kernel-owned MAP_SHARED pages survive any process death; per-tick frames "
        "(stage, RSS, conditions, failures) extracted into a governed death report consumed "
        "by the incident narrator and the continuity waking sequence",
        detection_modules=("core.runtime.flight_recorder",),
        mitigation_modules=("core.runtime.flight_recorder", "core.observability.incident_narrator"),
        occurrences=(
            "2026-07-03 endurance OOM 35GB (no relaunch, no final-moment record)",
            "2026-07-06 duplicate-runtime cascade (diagnosed from side-channels)",
        ),
        notes="The ring is written by the death itself; only whole-machine loss can erase it.",
    ),
    # ── Learning direction ─────────────────────────────────────────
    FailureMode(
        id="FM-LEARN-001",
        subsystem="core/learning/deliberate_practice.py (practice director)",
        mode="Curriculum misdirection: stale, corrupt, or skewed failure evidence steers "
        "practice at the wrong domains",
        cause="Ledger corruption, a burst of unrepresentative failures, or receipts from an "
        "old model generation dominating the decayed ranking",
        effect="Idle practice and specialist training drill domains that no longer need it — "
        "the waste of uniform practice returns, never worse than it (all consumers keep "
        "their uniform/least-recently-trained fallback)",
        blast_radius=BlastRadius.LANE,
        severity=Severity.MINOR,
        detection="Per-domain curriculum is receipts-attached and served on "
        "/api/system/learning; the learning self-report states the direction in chat; "
        "mastery rail zeroes any domain holding ≥95%",
        mitigation="7-day evidence half-life ages out stale failures; corrupt ledger lines "
        "are skipped (never void the ledger); AURA_DELIBERATE_PRACTICE=0 kill switch "
        "restores uniform practice instantly; observation intake never blocks practice",
        detection_modules=("core.learning.deliberate_practice",),
        mitigation_modules=("core.learning.deliberate_practice",),
        notes="Direction is quality-of-learning only: the two-sided specialist gate and the "
        "sealed compounding gate still decide what ships, so misdirection can waste idle "
        "compute but cannot promote a regression.",
    ),
    # ── Cognition ──────────────────────────────────────────────────
    #
    # Everything above this line is infrastructure: memory pressure, worker
    # lifecycle, lane admission, forensics. That is the half of the system the
    # OS can kill, and it was the only half declared here.
    #
    # The failures in this section were all found by RUNNING the mechanisms,
    # none by reading them, and none had a declared detection at the time.
    # They share a shape that infrastructure FMEA does not cover: the
    # subsystem stays up, every contract test passes, and the *dynamics* are
    # wrong — so the effect is silent degradation of cognition rather than an
    # outage. A cognitive failure mode with no steady-state detector is
    # indistinguishable from correct operation for as long as nobody looks.
    FailureMode(
        id="FM-COG-001",
        subsystem="core/consciousness/global_workspace.py (competition)",
        mode="Broadcast monopoly or cartel: the workspace stops rotating in steady state",
        cause="Refractory policy that excludes losers, or adaptation whose recovery rate is "
        "fixed. A leaky integrator with gain g and fixed decay r pins each source's "
        "sustainable share at r/g, so exactly g/r sources can rotate however many bid",
        effect="No global workspace, only a sort. Every downstream consumer of winners — "
        "soul.py, the context stream, ignition-gated cognition — reads a broadcast that "
        "one or two producers hold permanently",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.CRITICAL,
        detection="Steady-state competition health over a trajectory: no_starvation and "
        "rotation_entropy against order_preserving, plus no_limit_cycle for the a-b-a-b "
        "alternation that scores as a perfect rotation on entropy alone",
        mitigation="Adaptation recovery derived as r = g/n from the size of the field, making "
        "the equilibrium share 1/n; the lone-source and urgent-vs-idle regimes fall out "
        "rather than needing special cases",
        detection_modules=("core.verify.dynamics",),
        mitigation_modules=("core.consciousness.global_workspace",),
        occurrences=(
            "2026-08-12 monopoly: 24/24 wins to one source of four, past 46 contract tests",
            "2026-08-12 duopoly: 12/12 split by the top two, past the monopoly fix's own test",
        ),
        notes="Both states passed the assertions written against the previous one. "
        "top_share < 0.75 with two distinct winners is true of a perfect duopoly, which is "
        "why the check is now a property over a trajectory rather than a bound on a symptom.",
    ),
    FailureMode(
        id="FM-COG-002",
        subsystem="core/consciousness/global_workspace.py (arbitration)",
        mode="Decisions settled by submission timing while presented as priority",
        cause="effective_priority scales salience by (1 - 0.03*age), so candidates of "
        "identical salience differ by ~0.03*spread purely from when they were submitted",
        effect="Arbitrary arbitration that looks principled. Measured: four sources bidding "
        "an identical 0.70 produced 0 exact ties in 12 ticks and were separated by ~2e-6",
        blast_radius=BlastRadius.LANE,
        severity=Severity.MAJOR,
        detection="Tie impasses counted against the timing noise floor that mechanism "
        "creates rather than against exact equality; surfaced in get_snapshot()",
        mitigation="_resolve_tie decides among indistinguishable bids on two rules that "
        "cannot see arrival time: least-fatigued first (the same quantity arbitration "
        "already uses, so it hands the slot to whoever waited longest), then rotation on "
        "the tick index when fatigue is level too",
        detection_modules=("core.consciousness.global_workspace", "core.cognition.impasse"),
        mitigation_modules=("core.consciousness.global_workspace",),
        occurrences=("2026-08-12 identical-bid probe: 3 tie impasses in 12 ticks",),
        notes="Closed. Four sources bidding an identical 0.70 over 48 ticks now take "
        "0.25 each, and the distribution is byte-identical whether they are submitted in "
        "a fixed order or a rotating one — which is the actual property, since arrival "
        "order used to be the whole input. Rotation rather than a fixed order matters: a "
        "stable sort would hand every genuine tie to whichever source sorts first, "
        "forever.",
    ),
    FailureMode(
        id="FM-COG-003",
        subsystem="core/memory/episodic_memory.py (recall ranking)",
        mode="A ranking term silently decays to a constant",
        cause="Scoring keyed to an absolute wall-clock epoch rather than elapsed time; the "
        "usable window recedes as real time advances until every live input saturates",
        effect="Recall ranked by importance alone. Measured on 2026-08-12, every episode "
        "newer than 2026-04-12 scored exactly 1.000000 — one minute and thirty days "
        "indistinguishable — so the term contributed a constant 0.4 to every candidate",
        blast_radius=BlastRadius.ORGANISM,
        severity=Severity.MAJOR,
        detection="Scale-freedom: identical relative histories must score identically at any "
        "wall clock, and distinct ages must produce distinct scores across the operating range",
        mitigation="ACT-R base-level activation B = ln(sum t_j^-d), a function of elapsed time "
        "only, so it cannot saturate and has no epoch to go stale; also reads frequency",
        detection_modules=("core.cognition.actr_activation",),
        mitigation_modules=("core.cognition.actr_activation", "core.memory.episodic_memory"),
        occurrences=("2026-08-12 recency scorer measured flat across the whole live range",),
        notes="Time-bomb class: correct when written, degrades with the calendar, and no test "
        "that does not move the clock can see it. The detector moves the clock.",
    ),
)


def failure_modes_for(subsystem_fragment: str) -> list[FailureMode]:
    """All modes whose subsystem mentions the fragment (case-insensitive)."""
    fragment = subsystem_fragment.strip().lower()
    return [m for m in FMEA_REGISTRY if fragment in m.subsystem.lower()]


def detection_gaps() -> list[FailureMode]:
    return [m for m in FMEA_REGISTRY if m.detection.strip().upper() == "GAP"]


def mitigation_gaps() -> list[FailureMode]:
    return [m for m in FMEA_REGISTRY if m.mitigation.strip().upper() == "GAP"]


def all_referenced_modules() -> set[str]:
    modules: set[str] = set()
    for mode in FMEA_REGISTRY:
        modules.update(mode.detection_modules)
        modules.update(mode.mitigation_modules)
    return modules


def registry_summary() -> dict[str, int]:
    return {
        "total": len(FMEA_REGISTRY),
        "catastrophic": sum(1 for m in FMEA_REGISTRY if m.severity is Severity.CATASTROPHIC),
        "critical": sum(1 for m in FMEA_REGISTRY if m.severity is Severity.CRITICAL),
        "major": sum(1 for m in FMEA_REGISTRY if m.severity is Severity.MAJOR),
        "minor": sum(1 for m in FMEA_REGISTRY if m.severity is Severity.MINOR),
        "mitigation_gaps": len(mitigation_gaps()),
        "detection_gaps": len(detection_gaps()),
    }
