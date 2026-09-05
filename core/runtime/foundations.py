"""core/runtime/foundations.py — one boot entry for the engineering spine.

Aura has grown a set of disciplines borrowed, clean-room, from projects
that earned them the expensive way: the Linux kernel (taint, lockdep, PSI,
OOM policy), LLVM (verifier, pass manager, sanitizers), Kubernetes
(reconcilers, admission, quota, probes, eviction, leases), ROS 2 (managed
lifecycles, QoS, declared parameters, bags, diagnostics), Chromium
(histograms, traces, memory-infra, field trials, layering), and flight
software from F Prime / Apollo / OpenMCT (telemetry dictionaries, command
sequencing, rate groups, restart protection, assertions).

Every one of those is worth nothing if it is a module nobody calls. This
file is the single place the runtime turns them on, in dependency order,
with one report describing what came up and what did not. `aura_main`
calls :func:`activate_foundations` once during boot; nothing else needs to
know the list.

Design rules, all deliberate:

* **On by default.** No activator is behind an opt-in flag. A discipline
  that has to be enabled is a discipline that is off in the incident you
  needed it for. Individual activators may be disabled for the
  foreground-only boot profile, which genuinely has no background lanes.
* **Never fatal.** An activator that fails records a degradation, marks
  itself down in the report, and lets boot proceed. The runtime existed
  before these existed; a validator must not become a new way to fail to
  start.
* **Report, don't hide.** The returned report is written into the runtime
  manifest and surfaced by the health contract, so "is lockdep actually
  on?" has an answer that does not require reading this file.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Foundations")

#: Host memory available-fraction below which the OOM policy starts
#: shedding, and below which it treats the situation as terminal.
SOFT_PRESSURE_AVAILABLE_FRACTION = 0.12
HARD_PRESSURE_AVAILABLE_FRACTION = 0.06

#: The sentinel's duty cycle. Long enough to be free, short enough that a
#: fast allocator cannot cross both thresholds between samples.
SENTINEL_INTERVAL_S = 5.0

#: A monotonic-vs-wall-clock divergence beyond this in one sentinel period
#: means the wall clock jumped (NTP step, sleep/wake, VM migration).
CLOCK_JUMP_TOLERANCE_S = 5.0

_COGNITION_VALIDATION_LOCK = checked_lock("core.runtime.foundations.singleton")
_COGNITION_VALIDATION_STATUS: dict[str, Any] = {
    "state": "not_started",
    "started_at": None,
    "finished_at": None,
    "duration_s": None,
    "outcome": None,
    "error": "",
}


@dataclass
class ActivationResult:
    name: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "data": self.data}


class MemorySentinel:
    """The reclaim path, kept independent of every organ it may shed.

    Runs the jobs that must keep working when the rest of the runtime is
    stalled: memory-pressure reclaim, PSI memory accounting, and clock-jump
    detection. It is one small loop rather than three because these all
    need the same sample and the same independence.
    """

    def __init__(self, *, interval_s: float = SENTINEL_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._last_wall = time.time()
        self._last_mono = time.monotonic()
        self._memory_stalled = False
        self.samples = 0
        self.sheds = 0
        self.evictions = 0
        #: Re-scan the container for new shed candidates every ~60s.
        self.rescan_every = max(1, int(60.0 / max(interval_s, 0.1)))
        self._rescan_countdown = self.rescan_every

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._task = get_task_tracker().create_task(
            self._run(),
            name="foundations.memory_sentinel",
        )

    async def stop(self) -> None:
        self._stopping.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — shutdown must not raise
            pass
        # Leaving a stall open would peg PSI at 100% forever.
        self._end_memory_stall()

    def _begin_memory_stall(self) -> None:
        if self._memory_stalled:
            return
        from core.runtime.pressure_stall import Resource, get_pressure_monitor

        get_pressure_monitor().begin_stall(Resource.MEMORY)
        self._memory_stalled = True

    def _end_memory_stall(self) -> None:
        if not self._memory_stalled:
            return
        from core.runtime.pressure_stall import Resource, get_pressure_monitor

        get_pressure_monitor().end_stall(Resource.MEMORY)
        self._memory_stalled = False

    def _check_clock(self) -> None:
        wall = time.time()
        mono = time.monotonic()
        wall_delta = wall - self._last_wall
        mono_delta = mono - self._last_mono
        self._last_wall = wall
        self._last_mono = mono
        skew = wall_delta - mono_delta
        if abs(skew) > CLOCK_JUMP_TOLERANCE_S:
            from core.runtime.taint import TaintFlag, taint

            taint(
                TaintFlag.CLOCK_JUMP,
                f"wall clock moved {skew:+.1f}s relative to the monotonic clock; "
                "durations measured across this point are not trustworthy",
                subsystem="memory_sentinel",
            )

    def _sample_memory(self) -> tuple[int, int] | None:
        try:
            from core.runtime.resource_observation import get_resource_observer

            observation = get_resource_observer().memory()
            if not observation.available or observation.total_bytes <= 0:
                return None
            return int(observation.available_bytes), int(observation.total_bytes)
        except Exception:  # noqa: BLE001 — resource providers are optional at this boundary
            logger.debug("memory sentinel sample failed", exc_info=True)
            return None

    def _record_observability(self, available_fraction: float) -> None:
        """Feed the histograms, the trace counters, and memory attribution.

        The memory dump on every tick is what turns the open ~242MB/h soak
        question into an answerable one: two dumps and a diff name the
        component that grew, which neither RSS nor allocation-site
        profiling can do.
        """
        try:
            from core.observability.histograms import record
            from core.observability.trace_events import trace_counter
            from core.runtime.pressure_stall import Resource, pressure

            memory_pressure = pressure(Resource.MEMORY)
            record("Aura.Memory.AvailableFraction", available_fraction)
            record("Aura.Pressure.MemoryFull", memory_pressure * 100.0)
            trace_counter(
                "memory",
                {
                    "available_fraction": available_fraction,
                    "psi_memory_full": memory_pressure,
                    "psi_inference_full": pressure(Resource.INFERENCE),
                },
                category="resource",
            )
        except Exception:  # noqa: BLE001 — telemetry is additive to reclaim
            logger.debug("observability sampling failed", exc_info=True)
        try:
            from core.runtime.memory_infra import DetailLevel, get_memory_infra

            get_memory_infra().dump(
                DetailLevel.LIGHT
                if available_fraction <= SOFT_PRESSURE_AVAILABLE_FRACTION
                else DetailLevel.BACKGROUND
            )
        except Exception:  # noqa: BLE001 — attribution is additive to reclaim
            logger.debug("memory dump failed", exc_info=True)

    def _evaluate(self) -> None:
        from core.runtime.oom_policy import get_oom_policy

        # Organs arrive after boot — lazily constructed services, hot-swapped
        # adapters, a model that was not resident at activation. Re-scanning
        # keeps the shed order complete instead of frozen at boot; discovery
        # is idempotent and never instantiates anything.
        self._rescan_countdown -= 1
        if self._rescan_countdown <= 0:
            self._rescan_countdown = self.rescan_every
            try:
                _register_oom_organs()
            except Exception:  # noqa: BLE001 — organ discovery cannot stop reclaim
                logger.debug("OOM organ rescan failed", exc_info=True)

        # Graded eviction runs before the crude OOM ladder: reclaim caches
        # first, evict BestEffort organs next, and only then let the OOM
        # policy pick a victim. Gated fail-OPEN — skipping protective work
        # because another process might be doing it is the wrong failure
        # direction. See lease.should_act_as_singleton.
        from core.runtime.lease import RUNTIME_LEASE, should_act_as_singleton

        if should_act_as_singleton(RUNTIME_LEASE):
            try:
                from core.runtime.eviction import get_eviction_manager

                outcome = get_eviction_manager().enforce()
                self.evictions += len(
                    [a for a in outcome.get("actions", ()) if a.get("action") == "evict"]
                )
            except Exception:  # noqa: BLE001 — eviction cannot stop emergency reclaim
                logger.debug("eviction enforcement failed", exc_info=True)

        sample = self._sample_memory()
        if sample is None:
            return
        available, total = sample
        fraction = available / float(total)
        self.samples += 1
        self._record_observability(fraction)

        if fraction > SOFT_PRESSURE_AVAILABLE_FRACTION:
            self._end_memory_stall()
            return

        # Under soft pressure the runtime is waiting on reclaim whether or
        # not any single caller says so; that is exactly what PSI memory
        # pressure means.
        self._begin_memory_stall()

        policy = get_oom_policy()
        target = int(total * SOFT_PRESSURE_AVAILABLE_FRACTION * 1.5)
        reason = (
            f"host memory available {fraction * 100:.1f}% "
            # GiB: "64GB of RAM" is 1024**3 bytes per GB everywhere a person
            # reads it, including Apple's own description of the machine.
            f"({available / 1024**3:.2f}GB of {total / 1024**3:.2f}GB)"
        )
        events = policy.shed_until(
            target_free_bytes=target,
            free_bytes_now=lambda: (self._sample_memory() or (0, 1))[0],
            reason=reason,
        )
        self.sheds += len(events)

        after = self._sample_memory()
        after_fraction = (after[0] / float(after[1])) if after else fraction
        if after_fraction <= HARD_PRESSURE_AVAILABLE_FRACTION and policy.no_victim_available():
            policy.request_controlled_restart(reason)

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                self._check_clock()
                await asyncio.to_thread(self._evaluate)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — the reclaim path never dies
                from core.runtime.errors import record_degradation

                record_degradation(
                    "memory_sentinel",
                    exc,
                    severity="warning",
                    action="sentinel iteration skipped; loop continues",
                    enforce_failure_policy=False,
                )
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.interval_s)
            except TimeoutError:
                continue

    def report(self) -> dict[str, Any]:
        return {
            "running": self._task is not None and not self._task.done(),
            "interval_s": self.interval_s,
            "samples": self.samples,
            "sheds": self.sheds,
            "evictions": self.evictions,
            "memory_stall_open": self._memory_stalled,
        }


_SENTINEL: MemorySentinel | None = None


def get_memory_sentinel() -> MemorySentinel:
    global _SENTINEL
    if _SENTINEL is None:
        _SENTINEL = MemorySentinel()
    return _SENTINEL


# ══════════════════════════════════════════════════════════════════════
# Wave activators
# ══════════════════════════════════════════════════════════════════════

def _declare_pressure_capacities() -> dict[str, int]:
    """Tell PSI how many workers can contend for each resource.

    ``full`` pressure means *every* worker stalled, so these numbers decide
    whether the throughput-collapse signal is meaningful or trivially true.
    """
    from core.runtime.pressure_stall import Resource, declare_capacity
    from core.runtime.resource_observation import get_resource_observer

    cpus = max(1, int(get_resource_observer().compute().cpu_count))
    capacities: dict[str, int] = {
        # Cognition lanes contend for compute; the host's core count is the
        # honest ceiling.
        str(Resource.CPU): cpus,
        # Reclaim is global: one waiter is everyone waiting.
        str(Resource.MEMORY): 1,
        # The durability lane is a small thread pool, not a single fd.
        str(Resource.IO): max(2, min(8, cpus // 2)),
        # One resident model unless the lane controller says otherwise.
        str(Resource.INFERENCE): _model_lane_capacity(),
        str(Resource.BUS): 1,
        str(Resource.LOCK): cpus,
    }
    for name, workers in capacities.items():
        declare_capacity(name, workers)
    return capacities


def _model_lane_capacity() -> int:
    try:
        from core.runtime.model_lane_control import get_model_lane_controller

        controller = get_model_lane_controller()
        for attr in ("max_lanes", "lane_capacity", "concurrency", "max_concurrent_lanes"):
            value = getattr(controller, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    except Exception:  # noqa: BLE001 — model-lane discovery has a safe unit fallback
        logger.debug("model lane capacity probe unavailable", exc_info=True)
    return 1


#: Organs whose loss is worse than the memory they hold. The kernel gives
#: init OOM_SCORE_ADJ_MIN for the same reason: some things must not be the
#: answer to "what should we kill".
IMMUNE_SERVICES: tuple[str, ...] = (
    "unified_will",
    "will",
    "event_bus",
    "container",
    "memory_facade",
    "flight_recorder",
    "shutdown_coordinator",
    "health",
    "orchestrator",
    "identity",
    "self_object",
)


def _register_oom_organs() -> dict[str, Any]:
    """Build the shed order before the pressure arrives.

    Discovery is by capability, not by name: any *already-instantiated*
    service exposing ``shed_memory()`` volunteers. Lazily-registered
    services are deliberately not instantiated here — constructing an organ
    to learn it could be shed under memory pressure is exactly
    backwards.
    """
    from core.container import ServiceContainer
    from core.runtime.oom_policy import OOM_SCORE_ADJ_MIN, register_organ

    registered: list[str] = []
    for name in IMMUNE_SERVICES:
        register_organ(
            name,
            oom_score_adj=OOM_SCORE_ADJ_MIN,
            rationale="load-bearing: losing it costs more than the memory it holds",
            recoverable=False,
        )
        registered.append(name)

    discovered: list[str] = []
    for service_name, instance in _instantiated_services(ServiceContainer).items():
        if service_name in IMMUNE_SERVICES:
            continue
        shed = getattr(instance, "shed_memory", None)
        if not callable(shed):
            continue
        adj = int(getattr(instance, "oom_score_adj", 0) or 0)
        footprint = getattr(instance, "memory_footprint_bytes", None)
        register_organ(
            service_name,
            oom_score_adj=adj,
            footprint=footprint if callable(footprint) else None,
            shed=shed,
            rationale=getattr(instance, "oom_rationale", "")
            or f"{service_name} exposes shed_memory()",
            recoverable=bool(getattr(instance, "oom_recoverable", True)),
        )
        discovered.append(service_name)

    return {"immune": registered, "sheddable": discovered}


def _instantiated_services(container: Any) -> dict[str, Any]:
    """Live instances only — never triggers a lazy factory."""
    out: dict[str, Any] = {}
    services = getattr(container, "_services", None)
    if not isinstance(services, dict):
        return out
    for name, descriptor in list(services.items()):
        instance = getattr(descriptor, "instance", None)
        if instance is not None:
            out[str(name)] = instance
    return out


async def _activate_kernel_discipline(*, foreground_only: bool) -> ActivationResult:
    """Wave 1 — taint register, lockdep, PSI, OOM policy, memory sentinel."""
    from core.runtime.lockdep import lockdep_report, note_event_loop_thread

    note_event_loop_thread()
    capacities = _declare_pressure_capacities()
    organs = _register_oom_organs()

    sentinel_started = False
    if not foreground_only:
        await get_memory_sentinel().start()
        sentinel_started = True
        try:
            from core.runtime.shutdown_coordinator import get_shutdown_coordinator

            get_shutdown_coordinator().register(
                get_memory_sentinel().stop,
                phase="task_supervisor",
                name="foundations.memory_sentinel",
                timeout=5.0,
            )
        except Exception:  # noqa: BLE001 — shutdown registration is additive
            logger.debug("memory sentinel shutdown registration skipped", exc_info=True)

    return ActivationResult(
        name="kernel_discipline",
        ok=True,
        detail=(
            f"lockdep armed ({lockdep_report()['acquires_checked']} acquires checked), "
            f"PSI over {len(capacities)} resources, "
            f"{len(organs['immune'])} immune + {len(organs['sheddable'])} sheddable organs"
        ),
        data={
            "pressure_capacities": capacities,
            "oom_organs": organs,
            "memory_sentinel_started": sentinel_started,
        },
    )


async def _activate_verification(*, foreground_only: bool) -> ActivationResult:
    """Wave 2 — structural verifier, pass instrumentation, sanitizers."""
    # Importing registers the standing invariants; the module is a
    # declaration site, not a service.
    from core.observability.turn_observer import install_turn_observer
    from core.pipeline.pass_manager import get_instrumentation, install_default_instrumentation
    from core.verify import runtime_invariants  # noqa: F401 — import registers
    from core.verify.invariants import get_registry, verify

    instrumentation = install_default_instrumentation()

    # Accumulated causal-influence samples, restored before any traffic.
    #
    # ``InfluenceLedger`` always carried load()/as_dict() with a comment saying
    # "a ledger that resets every boot never reaches a verdict, so the samples
    # have to outlive the process that took them". Nothing ever called either,
    # so every boot started at zero observations and every channel read
    # UNMEASURED permanently — not because the measurement said nothing, but
    # because it was never allowed to accumulate. This is the load half.
    try:
        from core.verify.influence_campaign import load_persisted_ledger

        load_persisted_ledger()
    except Exception as exc:  # noqa: BLE001 — evidence must not break boot
        from core.runtime.errors import record_degradation

        record_degradation(
            "foundations",
            exc,
            severity="debug",
            action="started with an empty influence ledger",
            enforce_failure_policy=False,
        )

    # Per-turn cost and stuck verdicts. Registered here because this is where
    # the pass seam is armed, and it attaches to that same seam. It installs an
    # after-hook only, so it can observe every turn and alter none of them —
    # see core/observability/turn_observer.py for why that is structural rather
    # than a promise.
    install_turn_observer()

    # The first verification runs over the runtime as boot left it. This is
    # the moment a structural regression is cheapest to see: before any
    # traffic, with the boot path still on the stack.
    report = verify()
    declared = len(get_registry().specs())

    return ActivationResult(
        name="verification",
        ok=report.ok,
        detail=(
            f"{declared} invariants declared, {report.summary()}; "
            f"pass instrumentation {'armed' if instrumentation['installed'] else 'already armed'}"
            ", turn observer metering (observe-only)"
            + (
                f", opt-bisect limit={get_instrumentation().bisect_limit()}"
                if get_instrumentation().bisect_limit() is not None
                else ""
            )
        ),
        data={
            "invariants_declared": declared,
            "scopes": get_registry().scopes(),
            "boot_verification": report.to_dict(),
            "pass_instrumentation": instrumentation,
        },
    )


async def _activate_orchestration(*, foreground_only: bool) -> ActivationResult:
    """Wave 3 — admission, quota, eviction, controllers, leader election."""
    from core.runtime.eviction import get_eviction_manager
    from core.runtime.lease import RUNTIME_LEASE, get_elector
    from core.runtime.quota import install_quota_admission
    from core.runtime.reconcile import get_controller_manager

    quota_hook = install_quota_admission()
    oom_scores = get_eviction_manager().sync_oom_scores()
    started_controllers = await get_controller_manager().start_all()

    # Contend for the runtime lease. Not holding it is not an error — it
    # is the *answer*, and it is the answer that used to require reading
    # memory graphs after the duplicate-runtime cascade had already
    # happened. Never blocks boot.
    leader = False
    if not foreground_only:
        elector = get_elector(RUNTIME_LEASE)
        await elector.start()
        # One synchronous attempt so the boot report says something true
        # rather than "pending".
        leader = await elector.try_acquire_or_renew()
        try:
            from core.runtime.shutdown_coordinator import get_shutdown_coordinator

            coordinator = get_shutdown_coordinator()
            coordinator.register(
                elector.stop,
                phase="task_supervisor",
                name=f"lease.{RUNTIME_LEASE}",
                timeout=5.0,
            )
            coordinator.register(
                get_controller_manager().stop_all,
                phase="task_supervisor",
                name="controller_manager",
                timeout=10.0,
            )
        except Exception:  # noqa: BLE001 — shutdown registration is additive
            logger.debug("orchestration shutdown registration skipped", exc_info=True)

    return ActivationResult(
        name="orchestration",
        ok=True,
        detail=(
            f"admission chain live (quota hook {'installed' if quota_hook else 'present'}), "
            f"{len(oom_scores)} QoS→OOM scores synced, "
            f"{len(started_controllers)} controller(s) started, "
            f"runtime lease {'HELD' if leader else 'not held'}"
        ),
        data={
            "quota_admission_installed": quota_hook,
            "qos_oom_scores": oom_scores,
            "controllers": started_controllers,
            "runtime_lease_held": leader,
        },
    )


#: Topics whose volume would evict everything else from the bus ring.
#: Excluding them is what keeps the ring's minute of history useful.
BAG_EXCLUDED_TOPICS: tuple[str, ...] = (
    "metrics.sample",
    "telemetry.tick",
    "substrate.activation",
    "heartbeat",
)

#: Organs adopted into managed lifecycles at boot. Adoption gives an
#: existing start/stop object a visible state and makes its deactivation
#: distinguishable from its failure, without rewriting it.
LIFECYCLE_ADOPTIONS: tuple[tuple[str, bool], ...] = (
    ("orchestrator", True),
    ("event_bus", True),
    ("memory_facade", True),
    ("autonomy_conductor", False),
    ("research_cycle", False),
    ("curiosity_engine", False),
    ("performance_guard", False),
    ("self_healing", False),
    ("viability", False),
    ("flagship_doctor_daemon", False),
)


def _declare_core_parameters() -> list[str]:
    """Give the thresholds this module already hard-codes a real home.

    Each was a literal that could not be found, justified, or changed
    without a restart. Declared, they are inventoried, range-checked,
    observable, and retunable on a live runtime.
    """
    from core.runtime.parameters import ParameterType, declare

    specs: tuple[tuple[str, Any, dict[str, Any]], ...] = (
        (
            "memory.soft_pressure_available_fraction",
            SOFT_PRESSURE_AVAILABLE_FRACTION,
            {
                "type": ParameterType.FLOAT,
                "description": "available-memory fraction below which reclaim and shedding begin",
                "owner": "core/runtime/foundations.py",
                "minimum": 0.01,
                "maximum": 0.9,
            },
        ),
        (
            "memory.hard_pressure_available_fraction",
            HARD_PRESSURE_AVAILABLE_FRACTION,
            {
                "type": ParameterType.FLOAT,
                "description": (
                    "available-memory fraction at which a controlled restart beats "
                    "waiting to be killed"
                ),
                "owner": "core/runtime/foundations.py",
                "minimum": 0.005,
                "maximum": 0.5,
            },
        ),
        (
            "memory.sentinel_interval_s",
            SENTINEL_INTERVAL_S,
            {
                "type": ParameterType.FLOAT,
                "description": "duty cycle of the independent reclaim sentinel",
                "owner": "core/runtime/foundations.py",
                "minimum": 1.0,
                "maximum": 60.0,
            },
        ),
        (
            "lockdep.loop_blocking_hold_ms",
            50.0,
            {
                "type": ParameterType.FLOAT,
                "description": "sync-lock hold on the loop thread beyond which lockdep reports",
                "owner": "core/runtime/lockdep.py",
                "minimum": 1.0,
                "maximum": 5000.0,
            },
        ),
        (
            "pressure.saturation_threshold",
            0.20,
            {
                "type": ParameterType.FLOAT,
                "description": "PSI full-pressure fraction at which a resource counts as saturated",
                "owner": "core/runtime/pressure_stall.py",
                "minimum": 0.01,
                "maximum": 1.0,
            },
        ),
        (
            "bus.ring_capacity",
            8192,
            {
                "type": ParameterType.INT,
                "description": "messages retained in the always-on bus ring",
                "owner": "core/observability/bus_recorder.py",
                "minimum": 256,
                "maximum": 131072,
            },
        ),
        (
            "diagnostics.stale_after_s",
            30.0,
            {
                "type": ParameterType.FLOAT,
                "description": "silence after which a diagnostic task is reported STALE",
                "owner": "core/health/diagnostics_aggregator.py",
                "minimum": 5.0,
                "maximum": 600.0,
            },
        ),
    )
    declared: list[str] = []
    for name, default, kwargs in specs:
        try:
            declare(name, default, **kwargs)
            declared.append(name)
        except (ValueError, TypeError) as exc:
            logger.warning("parameter %s could not be declared: %s", name, exc)
    return declared


def _adopt_lifecycles() -> dict[str, str]:
    """Adopt already-instantiated organs into managed lifecycles."""
    from core.container import ServiceContainer
    from core.runtime.lifecycle import adopt

    adopted: dict[str, str] = {}
    instances = _instantiated_services(ServiceContainer)
    for name, critical in LIFECYCLE_ADOPTIONS:
        instance = instances.get(name)
        if instance is None:
            continue
        organ = adopt(name, instance, critical=critical)
        if organ is not None:
            adopted[name] = str(organ.state)
    return adopted


async def _activate_middleware(*, foreground_only: bool) -> ActivationResult:
    """Wave 4 — lifecycles, bus QoS, parameters, bus ring, diagnostics."""
    from core.health.diagnostics_aggregator import (
        install_default_analyzers,
        install_runtime_diagnostics,
    )
    from core.observability.bus_recorder import get_bus_recorder
    from core.runtime.lifecycle import lifecycle_report

    parameters = _declare_core_parameters()
    adopted = _adopt_lifecycles()

    recorder = get_bus_recorder()
    recorder.exclude(*BAG_EXCLUDED_TOPICS)

    analyzers = install_default_analyzers()
    tasks = install_runtime_diagnostics()
    # Registration creates diagnostic tasks with no sample yet. The cognition
    # validation wave runs in this same boot pass, so give every task its first
    # real observation instead of classifying new diagnostics as STALE.
    _safe_diagnostics_update()

    qos_topics = _declare_standard_topics()

    return ActivationResult(
        name="middleware",
        ok=True,
        detail=(
            f"{len(parameters)} parameters declared, {len(adopted)} organ(s) adopted "
            f"into managed lifecycles, {len(qos_topics)} QoS topics, "
            f"bus ring armed, {len(analyzers)} diagnostic analyzers over "
            f"{len(tasks)} tasks"
        ),
        data={
            "parameters": parameters,
            "lifecycles": adopted,
            "lifecycle_report": lifecycle_report()["by_state"],
            "qos_topics": qos_topics,
            "diagnostic_analyzers": analyzers,
            "diagnostic_tasks": tasks,
        },
    )


def _declare_standard_topics() -> list[str]:
    """Give the topics whose meaning depends on QoS an explicit contract.

    State topics get transient-local durability, which is what makes an
    organ that boots *after* a state announcement still learn the state
    instead of behaving as though it never changed.
    """
    from core.bus.qos import COMMAND, HEARTBEAT, SENSOR_DATA, STATE, declare_topic

    topics = {
        "runtime.state": STATE,
        "runtime.boot_phase": STATE,
        "cortex.lane_state": STATE,
        "autonomy.state": STATE,
        "health.verdict": STATE,
        "memory.pressure": STATE,
        "sensory.frame": SENSOR_DATA,
        "sensory.audio": SENSOR_DATA,
        "will.decision": COMMAND,
        "action.request": COMMAND,
        "mind.tick": HEARTBEAT,
    }
    for topic, profile in topics.items():
        declare_topic(topic, profile)
    return sorted(topics)


async def _activate_observability(*, foreground_only: bool) -> ActivationResult:
    """Wave 5 — histograms, traces, memory attribution, trials, Rule of Two."""
    from core.observability.histograms import install_standard_histograms
    from core.observability.trace_events import get_tracer, install_pass_tracing
    from core.runtime.memory_infra import DetailLevel, get_memory_infra, install_runtime_providers
    from core.security.rule_of_two import install_known_handlers, rule_of_two_report

    histograms = install_standard_histograms()
    tracing = install_pass_tracing()
    get_tracer().name_thread("runtime.main")

    providers = install_runtime_providers()
    # Take the first dump immediately: a leak report needs two points, and
    # the earlier one has to exist before the growth starts.
    baseline_dump = get_memory_infra().dump(DetailLevel.BACKGROUND)

    handlers = install_known_handlers()
    posture = rule_of_two_report()

    return ActivationResult(
        name="observability",
        ok=not posture["violations"],
        detail=(
            f"{len(histograms)} histograms declared, pass tracing "
            f"{'armed' if tracing else 'already armed'}, "
            f"{len(providers)} memory providers "
            f"({baseline_dump.attributed_bytes / 1e6:.0f}MB attributed of "
            f"{baseline_dump.process_rss_bytes / 1e6:.0f}MB RSS), "
            f"{len(handlers)} security postures declared"
            + (
                f", {len(posture['violations'])} RULE-OF-TWO VIOLATION(S)"
                if posture["violations"]
                else ""
            )
        ),
        data={
            "histograms": histograms,
            "pass_tracing": tracing,
            "memory_providers": providers,
            "baseline_dump": baseline_dump.to_dict()["attributed_fraction"],
            "rule_of_two": {
                "declared": handlers,
                "violations": posture["violations"],
                "at_the_limit": posture["at_the_limit"],
            },
        },
    )


def _declare_standard_telemetry() -> tuple[list[str], list[str]]:
    """The channel and event dictionary for the runtime's own state.

    Ids are the contract: anything reading Aura's telemetry can rely on
    channel 0x0101 meaning available-memory fraction forever. Limits are
    declared here so a crossing is a transition the system announces,
    rather than a threshold somebody remembered to check at a read site.
    """
    from core.fsw.telemetry_dictionary import ChannelType, EventSeverity, channel, event

    channels = (
        dict(
            identifier=0x0101,
            name="memory.available_fraction",
            unit="fraction",
            description="host memory available as a fraction of total",
            owner="core/runtime/foundations.py",
            group="resources",
            yellow_low=0.20,
            red_low=0.10,
            stale_after_s=60.0,
        ),
        dict(
            identifier=0x0102,
            name="pressure.memory_full",
            unit="fraction",
            description="PSI memory full-pressure over 10s",
            owner="core/runtime/pressure_stall.py",
            group="resources",
            yellow_high=0.20,
            red_high=0.50,
        ),
        dict(
            identifier=0x0103,
            name="pressure.inference_full",
            unit="fraction",
            description="PSI inference full-pressure over 10s",
            owner="core/runtime/pressure_stall.py",
            group="resources",
            yellow_high=0.30,
            red_high=0.70,
        ),
        dict(
            identifier=0x0104,
            name="memory.rss_bytes",
            unit="bytes",
            description="runtime process resident set size",
            owner="core/runtime/memory_infra.py",
            group="resources",
        ),
        dict(
            identifier=0x0201,
            name="runtime.taint_flags",
            type=ChannelType.INT,
            unit="count",
            description="number of taint flags set on this process",
            owner="core/runtime/taint.py",
            group="integrity",
            yellow_high=1,
            red_high=3,
            stale_after_s=300.0,
        ),
        dict(
            identifier=0x0202,
            name="runtime.lockdep_splats",
            type=ChannelType.INT,
            unit="count",
            description="distinct lock-order violations observed",
            owner="core/runtime/lockdep.py",
            group="integrity",
            yellow_high=1,
            stale_after_s=300.0,
        ),
        dict(
            identifier=0x0203,
            name="runtime.assertion_failures",
            type=ChannelType.INT,
            unit="count",
            description="distinct assertion sites that have failed",
            owner="core/fsw/assertions.py",
            group="integrity",
            yellow_high=1,
            stale_after_s=300.0,
        ),
        dict(
            identifier=0x0204,
            name="runtime.sanitizer_findings",
            type=ChannelType.INT,
            unit="count",
            description="distinct sanitizer findings",
            owner="core/runtime/sanitizers.py",
            group="integrity",
            yellow_high=1,
            stale_after_s=300.0,
        ),
        dict(
            identifier=0x0301,
            name="scheduler.core_sets_used",
            type=ChannelType.INT,
            unit="count",
            description="restart-protection core sets in use",
            owner="core/fsw/restart_protection.py",
            group="scheduling",
            yellow_high=18,
            red_high=23,
            stale_after_s=300.0,
        ),
        dict(
            identifier=0x0302,
            name="scheduler.consecutive_slips",
            type=ChannelType.INT,
            unit="count",
            description="worst consecutive cycle slips across rate groups",
            owner="core/fsw/rate_groups.py",
            group="scheduling",
            yellow_high=1,
            red_high=5,
            stale_after_s=300.0,
        ),
        dict(
            identifier=0x0401,
            name="controllers.queue_depth",
            type=ChannelType.INT,
            unit="count",
            description="total reconcile queue depth across controllers",
            owner="core/runtime/reconcile.py",
            group="orchestration",
            yellow_high=32,
            red_high=256,
            stale_after_s=300.0,
        ),
        dict(
            identifier=0x0402,
            name="health.unresponsive_components",
            type=ChannelType.INT,
            unit="count",
            description="components that stopped answering health pings",
            owner="core/fsw/health_checker.py",
            group="orchestration",
            yellow_high=1,
            red_high=2,
            stale_after_s=300.0,
        ),
    )
    events = (
        dict(
            identifier=0x1001,
            name="channel_limit_transition",
            severity=EventSeverity.WARNING_LO,
            format_string="{channel} went {previous} -> {state} at {value}{unit}",
            description="a telemetry channel crossed a declared limit",
            owner="core/fsw/telemetry_dictionary.py",
        ),
        dict(
            identifier=0x1002,
            name="program_alarm",
            severity=EventSeverity.WARNING_HI,
            format_string="ALARM {code}: {detail} (shed {shed}, kept {kept})",
            description="overload response: work was shed to protect the essential loop",
            owner="core/fsw/restart_protection.py",
        ),
        dict(
            identifier=0x1003,
            name="rate_group_slip",
            severity=EventSeverity.WARNING_LO,
            format_string=(
                "rate group {group} took {duration_ms}ms of a {period_ms}ms period "
                "({slowest} at {slowest_ms}ms), {consecutive} in a row"
            ),
            description="a rate group did not finish within its period",
            owner="core/fsw/rate_groups.py",
        ),
        dict(
            identifier=0x1004,
            name="assertion_failed",
            severity=EventSeverity.WARNING_HI,
            format_string="assertion '{condition}' failed at {site} in {function} -> {response}",
            description="a declared invariant was violated",
            owner="core/fsw/assertions.py",
        ),
        dict(
            identifier=0x1005,
            name="component_unresponsive",
            severity=EventSeverity.WARNING_HI,
            format_string="{component} missed {misses} health pings (critical={critical})",
            description="a component stopped answering active liveness pings",
            owner="core/fsw/health_checker.py",
        ),
        dict(
            identifier=0x1006,
            name="component_recovered",
            severity=EventSeverity.ACTIVITY_HI,
            format_string="{component} is answering again after {unresponsive_for_s}s",
            description="a previously unresponsive component answered",
            owner="core/fsw/health_checker.py",
        ),
        dict(
            identifier=0x1007,
            name="command_dispatched",
            severity=EventSeverity.ACTIVITY_HI,
            format_string="command {command} (0x{opcode:02x}) ok={ok} in {duration_ms}ms {error}",
            description="a dictionary command was dispatched",
            owner="core/fsw/command_dispatch.py",
        ),
    )
    declared_channels: list[str] = []
    for spec in channels:
        try:
            channel(**spec)  # type: ignore[arg-type]
            declared_channels.append(str(spec["name"]))
        except ValueError as exc:
            logger.warning("channel declaration failed: %s", exc)
    declared_events: list[str] = []
    for spec in events:
        try:
            event(**spec)  # type: ignore[arg-type]
            declared_events.append(str(spec["name"]))
        except ValueError as exc:
            logger.warning("event declaration failed: %s", exc)
    return declared_channels, declared_events


def _sample_standard_telemetry() -> None:
    """Write the declared channels from the reports that already exist.

    Runs on the 1Hz rate group. Cheap reads; the value is that a limit
    crossing becomes an announced transition instead of a number nobody
    compared to anything.
    """
    from core.fsw.telemetry_dictionary import write

    try:
        from core.runtime.resource_observation import get_resource_observer

        memory = get_resource_observer().memory()
        if memory.available and memory.total_bytes > 0:
            write("memory.available_fraction", memory.available_bytes / memory.total_bytes)
            write("memory.rss_bytes", int(memory.process_rss_bytes or 0))
    except Exception:  # noqa: BLE001 - optional telemetry must not stop the rate group
        logger.debug("memory telemetry sample failed", exc_info=True)
    try:
        from core.runtime.pressure_stall import Resource, pressure

        write("pressure.memory_full", pressure(Resource.MEMORY))
        write("pressure.inference_full", pressure(Resource.INFERENCE))
    except Exception:  # noqa: BLE001 - optional telemetry must not stop the rate group
        logger.debug("pressure telemetry sample failed", exc_info=True)
    try:
        from core.fsw.assertions import assertions_report
        from core.runtime.lockdep import lockdep_report
        from core.runtime.sanitizers import sanitizer_report
        from core.runtime.taint import taint_flags

        write("runtime.taint_flags", len(taint_flags()))
        write("runtime.lockdep_splats", len(lockdep_report()["splats"]))
        write("runtime.assertion_failures", assertions_report()["distinct_sites"])
        write("runtime.sanitizer_findings", sanitizer_report()["distinct_findings"])
    except Exception:  # noqa: BLE001 - optional telemetry must not stop the rate group
        logger.debug("integrity telemetry sample failed", exc_info=True)
    try:
        from core.fsw.rate_groups import rate_group_report
        from core.fsw.restart_protection import restart_report

        write("scheduler.core_sets_used", restart_report()["core_sets"]["used"])
        groups = rate_group_report()["groups"]
        write(
            "scheduler.consecutive_slips",
            max((g["consecutive_slips"] for g in groups), default=0),
        )
    except Exception:  # noqa: BLE001 - optional telemetry must not stop the rate group
        logger.debug("scheduler telemetry sample failed", exc_info=True)
    try:
        from core.fsw.health_checker import health_checker_report
        from core.runtime.reconcile import reconcile_report

        write("controllers.queue_depth", reconcile_report()["total_queue_depth"])
        write(
            "health.unresponsive_components",
            len(health_checker_report()["unresponsive"]),
        )
    except Exception:  # noqa: BLE001 - optional telemetry must not stop the rate group
        logger.debug("orchestration telemetry sample failed", exc_info=True)


async def _activate_flight_software(*, foreground_only: bool) -> ActivationResult:
    """Wave 6 — telemetry dictionary, rate groups, restart protection, commands."""
    from core.fsw.command_dispatch import install_runtime_commands
    from core.fsw.health_checker import get_health_checker, install_runtime_pings
    from core.fsw.rate_groups import rate_group
    from core.fsw.restart_protection import install_standard_groups
    from core.fsw.telemetry_dictionary import telemetry_report

    channels, events = _declare_standard_telemetry()
    groups = install_standard_groups()
    commands = install_runtime_commands()
    pings = install_runtime_pings()

    started_groups: list[str] = []
    if not foreground_only:
        # One 1Hz group carrying the periodic work these disciplines need,
        # in declared order under one measured budget — rather than five
        # independent sleep loops that slip together with no ordering.
        one_hz = rate_group("1hz", 1.0)
        one_hz.add("telemetry_sample", _sample_standard_telemetry, budget_fraction=0.20, order=10)
        one_hz.add(
            "diagnostics",
            lambda: _safe_diagnostics_update(),
            budget_fraction=0.20,
            order=20,
        )
        five_s = rate_group("5s", 5.0)
        five_s.add(
            "health_pings",
            get_health_checker().run_round,
            budget_fraction=0.40,
            order=10,
        )
        await one_hz.start()
        await five_s.start()
        started_groups = ["1hz", "5s"]
        try:
            from core.fsw.rate_groups import get_scheduler
            from core.runtime.shutdown_coordinator import get_shutdown_coordinator

            get_shutdown_coordinator().register(
                get_scheduler().stop_all,
                phase="task_supervisor",
                name="rate_groups",
                timeout=5.0,
            )
        except Exception:  # noqa: BLE001 - shutdown registration is an additive bridge
            logger.debug("rate group shutdown registration skipped", exc_info=True)

    # One sample immediately so the dictionary is not empty at first read.
    _sample_standard_telemetry()

    return ActivationResult(
        name="flight_software",
        ok=True,
        detail=(
            f"{len(channels)} telemetry channels and {len(events)} event types declared, "
            f"{len(groups)} restart groups, {len(commands)} commands, "
            f"{len(pings)} active health pings, "
            f"rate groups: {', '.join(started_groups) or 'not started (foreground)'}; "
            f"{len(telemetry_report()['violations'])} channel(s) out of limits"
        ),
        data={
            "channels": channels,
            "events": events,
            "restart_groups": groups,
            "commands": commands,
            "health_pings": pings,
            "rate_groups": started_groups,
        },
    )


def _safe_diagnostics_update() -> None:
    from core.health.diagnostics_aggregator import get_aggregator

    get_aggregator().update_all()


def cognition_validation_status() -> dict[str, Any]:
    """Report the boot-owned empirical validation run without implying a pass."""

    with _COGNITION_VALIDATION_LOCK:
        status = dict(_COGNITION_VALIDATION_STATUS)
        outcome = status.get("outcome")
        status["outcome"] = dict(outcome) if isinstance(outcome, dict) else None
        return status


def _set_cognition_validation_status(**values: Any) -> None:
    with _COGNITION_VALIDATION_LOCK:
        _COGNITION_VALIDATION_STATUS.update(values)


async def _activate_cognition(*, foreground_only: bool) -> ActivationResult:
    """Wave 7 — MeTTa rewriting and the self-validation suite."""
    from core.container import ServiceContainer
    from core.knowledge.metta import get_metta, install_runtime_rules, metta_report
    from core.organism.model_validation import (
        get_suite,
        install_runtime_validation,
        run_validation,
    )

    rules = install_runtime_rules()
    ServiceContainer.register_instance("metta", get_metta(), required=False)

    validation = install_runtime_validation()
    ServiceContainer.register_instance("validation_suite", get_suite(), required=False)
    # Foreground-only proof boots need the verdict before they proceed. The
    # The full desktop only installs the suite. Several empirical tests perform
    # bounded synthesis and can monopolize the interpreter for tens of seconds,
    # even from another Python thread. They belong to an explicit validation
    # process, not the serving process. Foreground-only proof boots retain the
    # synchronous verdict they explicitly requested.
    if not foreground_only:
        _set_cognition_validation_status(
            state="deferred_to_validation_process",
            started_at=None,
            finished_at=None,
            duration_s=None,
            outcome=None,
            error="",
        )
        return ActivationResult(
            name="cognition",
            ok=True,
            detail=(
                f"{len(rules)} MeTTa rules over {metta_report()['grounded_ops'].__len__()} "
                f"grounded ops; {validation['claims']} claims bound to "
                f"{len(validation['tests'])} validation tests; empirical run deferred "
                "to an explicit validation process"
            ),
            data={
                "metta_rules": rules,
                "validation": validation,
                "suite_outcome": {"state": "deferred_to_validation_process"},
                "problem_tests": [],
                "unsupported_claims": [
                    c["statement"] for c in get_suite().unsupported_claims()
                ],
            },
        )

    validation_started = time.time()
    outcome = run_validation()
    validation_finished = time.time()
    _set_cognition_validation_status(
        state="completed",
        started_at=validation_started,
        finished_at=validation_finished,
        duration_s=round(validation_finished - validation_started, 3),
        outcome={
            key: outcome[key]
            for key in (
                "passed",
                "failed",
                "errored",
                "not_measured",
                "applicable",
                "measured",
            )
        },
        error="",
    )
    problem_tests = [
        result.get("test", "unknown")
        for group in (outcome["failures"], outcome["errors"])
        for result in group
    ]
    problem_detail = (
        f"; affected tests: {', '.join(problem_tests)}" if problem_tests else ""
    )

    return ActivationResult(
        name="cognition",
        ok=outcome["failed"] == 0 and outcome["errored"] == 0,
        detail=(
            f"{len(rules)} MeTTa rules over {metta_report()['grounded_ops'].__len__()} "
            f"grounded ops; {validation['claims']} claims bound to "
            f"{len(validation['tests'])} validation tests — "
            f"{outcome['passed']} passed, {outcome['failed']} failed, "
            f"{outcome['errored']} errored{problem_detail}"
        ),
        data={
            "metta_rules": rules,
            "validation": validation,
            "suite_outcome": {
                k: outcome[k] for k in ("passed", "failed", "errored", "applicable")
            },
            "problem_tests": problem_tests,
            "unsupported_claims": [c["statement"] for c in get_suite().unsupported_claims()],
        },
    )


#: (name, activator) in dependency order. Later waves append here; the
#: order is the boot order and is meaningful.
async def _activate_ontogeny(*, foreground_only: bool) -> ActivationResult:
    """Wave 8 — the organ that turns consequence into disposition.

    It comes up at boot rather than lazily on first use, because the part that
    matters most is the part that runs when nobody is asking: the resolver that
    finds out what came of a decision, and the sweeper that closes episodes
    nobody observed. Without those registered the organ still records, and
    never learns a thing.
    """
    from core.ontogeny.wiring import install

    installed = install()
    if not installed:
        return ActivationResult(
            name="ontogeny", ok=True,
            detail="ontogeny unavailable; every control point keeps its incumbent",
        )
    from core.ontogeny.service import get_ontogeny

    core = get_ontogeny()
    return ActivationResult(
        name="ontogeny", ok=True,
        detail=core.summary(),
        data={
            "control_points": list(core.control_points()),
            "stages": {cp: str(core.authority.stage(cp)) for cp in core.control_points()},
        },
    )


async def _activate_conation(*, foreground_only: bool) -> ActivationResult:
    """Bring up the motivational organ.

    It comes up at boot rather than lazily, for the same reason the ontogenetic
    organ does: the parts that matter run when nobody is asking. The telemetry
    channels have to exist before the first sample, and the invariants have to
    be registered before the first state they would have caught. An organ that
    registers its checks on first use is unchecked exactly during the boot it
    is most likely to be wrong in.
    """
    from core.conation.wiring import boot

    result = boot()
    channels = result.get("telemetry") or []
    return ActivationResult(
        name="conation", ok=True,
        detail={
            "channels": len(channels),
            "invariants": bool(result.get("invariants")),
        },
    )


_ACTIVATORS: list[tuple[str, Callable[..., Any]]] = [
    ("kernel_discipline", _activate_kernel_discipline),
    ("verification", _activate_verification),
    ("orchestration", _activate_orchestration),
    ("middleware", _activate_middleware),
    ("observability", _activate_observability),
    ("flight_software", _activate_flight_software),
    ("cognition", _activate_cognition),
    ("ontogeny", _activate_ontogeny),
    ("conation", _activate_conation),
]


def register_activator(name: str, activator: Callable[..., Any]) -> None:
    """Append a wave activator. Idempotent by name."""
    for existing, _ in _ACTIVATORS:
        if existing == name:
            return
    _ACTIVATORS.append((name, activator))


_LAST_REPORT: dict[str, Any] = {"activated": False}


async def activate_foundations(*, foreground_only: bool = False) -> dict[str, Any]:
    """Turn on every borrowed discipline. Called once, from boot."""
    started = time.time()
    results: list[ActivationResult] = []
    for name, activator in _ACTIVATORS:
        try:
            result = activator(foreground_only=foreground_only)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, ActivationResult):
                result = ActivationResult(name=name, ok=True, detail=str(result))
        except Exception as exc:  # noqa: BLE001 — a validator must not break boot
            from core.runtime.errors import record_degradation

            record_degradation(
                "foundations",
                exc,
                severity="warning",
                action=f"{name} activation skipped; runtime continues without it",
                enforce_failure_policy=False,
            )
            result = ActivationResult(name=name, ok=False, detail=repr(exc))
        results.append(result)
        logger.info(
            "%s foundations/%s — %s",
            "✅" if result.ok else "⚠️",
            result.name,
            result.detail or ("active" if result.ok else "unavailable"),
        )

    report = {
        "activated": True,
        "at": started,
        "duration_s": round(time.time() - started, 3),
        "foreground_only": foreground_only,
        "waves": [r.to_dict() for r in results],
        "ok": all(r.ok for r in results),
        "failed": [r.name for r in results if not r.ok],
    }
    _LAST_REPORT.clear()
    _LAST_REPORT.update(report)

    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("foundations_report", report, required=False)
    except Exception:  # noqa: BLE001 — report registration is additive
        logger.debug("foundations report registration skipped", exc_info=True)
    return report


def foundations_report() -> dict[str, Any]:
    return dict(_LAST_REPORT)


def reset_foundations_for_test() -> None:
    global _SENTINEL
    _SENTINEL = None
    _LAST_REPORT.clear()
    _LAST_REPORT["activated"] = False
    _set_cognition_validation_status(
        state="not_started",
        started_at=None,
        finished_at=None,
        duration_s=None,
        outcome=None,
        error="",
    )


__all__ = [
    "ActivationResult",
    "HARD_PRESSURE_AVAILABLE_FRACTION",
    "IMMUNE_SERVICES",
    "MemorySentinel",
    "SOFT_PRESSURE_AVAILABLE_FRACTION",
    "activate_foundations",
    "cognition_validation_status",
    "foundations_report",
    "get_memory_sentinel",
    "register_activator",
    "reset_foundations_for_test",
]
