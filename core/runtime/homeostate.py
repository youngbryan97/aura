"""Homeostate — Salt-style declarative convergence for Aura's own runtime.

SaltStack's state system, fused into the organism: instead of imperative
maintenance scattered across subsystems, the desired shape of the runtime is
*declared* and an engine converges reality toward it, idempotently, with an
honest dry-run.

- **StateSpec / lowstate**: each state names an idempotent state function
  (from a registry) plus Salt's requisites — ``require`` (hard ordering +
  failure gate), ``watch`` (require + refresh on upstream changes),
  ``onchanges`` (run only when a referenced state actually changed) and
  ``onfail`` (run only when a referenced state failed — remediation states).
- **Compiler**: requisites become a DAG; unknown references and cycles are
  compile errors; execution order is deterministic (topological, stable by
  declaration order).
- **Runner**: every state function is idempotent — it inspects reality first
  and reports ``changes`` only for what it actually altered. ``test=True`` is
  Salt's dry-run: predicted changes, zero mutation.
- **Beacon → Reactor**: Salt's event-driven remediation. The
  ``DegradationBeacon`` watches the degradation tracker and publishes bus
  events when a subsystem crosses its threshold; the ``HomeostateReactor``
  maps event topics to highstates and re-converges, rate-limited.

Consequential writes go through the file-write gateway under a governed
scope; state application from async contexts uses ``apply_async`` (thread
offload) so the live event loop never blocks on fsync.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.resource_observation import get_resource_observer

logger = logging.getLogger("Aura.Runtime.Homeostate")


# ── Declarations ──────────────────────────────────────────────────────────

class CompileError(ValueError):
    """A lowstate failed to compile (duplicate id, unknown requisite, cycle)."""


@dataclass(frozen=True)
class StateSpec:
    """One declared desired state (Salt's lowstate chunk).

    ``retries``/``retry_interval_s`` are Salt's per-state retry option: a
    failing state function is re-attempted (bounded: at most 5 retries, 30s
    interval) before the failure is final. Dry runs never retry.
    """

    id: str
    fn: str
    args: Mapping[str, Any] = field(default_factory=dict)
    require: tuple[str, ...] = ()
    watch: tuple[str, ...] = ()
    onchanges: tuple[str, ...] = ()
    onfail: tuple[str, ...] = ()
    retries: int = 0
    retry_interval_s: float = 0.0

    def requisite_ids(self) -> tuple[str, ...]:
        return tuple(self.require) + tuple(self.watch) + tuple(self.onchanges) + tuple(self.onfail)


@dataclass
class StateResult:
    """Outcome of one state application.

    ``result`` is True (satisfied/converged), False (failed), or None
    (not run — an upstream ``require``/``watch`` failed).
    """

    id: str
    fn: str
    result: bool | None
    changes: dict[str, Any] = field(default_factory=dict)
    comment: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fn": self.fn,
            "result": self.result,
            "changes": dict(self.changes),
            "comment": self.comment,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class HighstateReport:
    """Aggregate outcome of one convergence run."""

    name: str
    test: bool
    results: list[StateResult] = field(default_factory=list)
    duration_ms: float = 0.0
    finished_at: float = 0.0

    @property
    def ok(self) -> bool:
        return all(r.result is True for r in self.results)

    @property
    def failed(self) -> list[str]:
        return [r.id for r in self.results if r.result is False]

    @property
    def not_run(self) -> list[str]:
        return [r.id for r in self.results if r.result is None]

    @property
    def changed(self) -> list[str]:
        return [r.id for r in self.results if r.changes]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "test": self.test,
            "ok": self.ok,
            "failed": self.failed,
            "not_run": self.not_run,
            "changed": self.changed,
            "results": [r.to_dict() for r in self.results],
            "duration_ms": round(self.duration_ms, 3),
            "finished_at": self.finished_at,
        }


# ── Compiler ──────────────────────────────────────────────────────────────

def compile_lowstate(specs: Sequence[StateSpec]) -> list[StateSpec]:
    """Validate and order a lowstate: unique ids, known requisites, no cycles.

    Deterministic: among ready states, declaration order wins (so a compiled
    highstate is reproducible run to run).
    """
    by_id: dict[str, StateSpec] = {}
    for spec in specs:
        if spec.id in by_id:
            raise CompileError(f"duplicate state id: {spec.id!r}")
        by_id[spec.id] = spec
    for spec in specs:
        for req in spec.requisite_ids():
            if req not in by_id:
                raise CompileError(f"state {spec.id!r} references unknown state {req!r}")

    order_index = {spec.id: i for i, spec in enumerate(specs)}
    deps: dict[str, set[str]] = {spec.id: set(spec.requisite_ids()) for spec in specs}
    ordered: list[StateSpec] = []
    ready = sorted((sid for sid, d in deps.items() if not d), key=order_index.__getitem__)
    remaining = {sid for sid, d in deps.items() if d}
    while ready:
        sid = ready.pop(0)
        ordered.append(by_id[sid])
        newly_ready = []
        for other in list(remaining):
            deps[other].discard(sid)
            if not deps[other]:
                remaining.discard(other)
                newly_ready.append(other)
        if newly_ready:
            ready = sorted(ready + newly_ready, key=order_index.__getitem__)
    if remaining:
        cycle = ", ".join(sorted(remaining))
        raise CompileError(f"requisite cycle among states: {cycle}")
    return ordered


# ── Grains: system facts (Salt's grains interface) ────────────────────────

_grains_cache: dict[str, Any] | None = None


def grains(*, refresh: bool = False) -> dict[str, Any]:
    """Static facts about the host this runtime lives on (cached).

    Salt's grains: cheap, stable facts states and beacons can consult and
    events can carry, so every convergence report is grounded in *which body*
    it ran on.
    """
    global _grains_cache
    if _grains_cache is not None and not refresh:
        return dict(_grains_cache)
    import os
    import platform
    import sys

    observer = get_resource_observer()
    compute = observer.compute()
    memory = observer.memory(include_process_tree=False)
    facts: dict[str, Any] = {
        "os": platform.system().lower(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "cpu_count": int(compute.cpu_count) if compute.available else 0,
    }
    if memory.available:
        facts["memory_total_gb"] = round(memory.total_bytes / (1024**3), 2)
        facts["memory_available_gb"] = round(memory.available_bytes / (1024**3), 2)
    _grains_cache = facts
    return dict(facts)


# ── State function registry + built-in modules ────────────────────────────

# A state function receives (test: bool, watch_triggered: bool, **args) and
# returns {"result": bool, "changes": dict, "comment": str}. It must be
# idempotent: inspect reality first, change only what differs, and under
# test=True predict changes without mutating anything.
StateFn = Callable[..., Mapping[str, Any]]


class StateModuleRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, StateFn] = {}
        self._register_builtins()

    def register(self, name: str, fn: StateFn) -> None:
        self._fns[name] = fn

    def get(self, name: str) -> StateFn | None:
        return self._fns.get(name)

    def names(self) -> list[str]:
        return sorted(self._fns)

    # ── built-ins ─────────────────────────────────────────────────────

    def _register_builtins(self) -> None:
        self.register("file.directory", _state_file_directory)
        self.register("service.available", _state_service_available)
        self.register("check.predicate", _state_check_predicate)
        self.register("remedy.callable", _state_remedy_callable)


# Universal extension points: named predicates (pure checks) and remedies
# (governed repair actions). Only pre-registered callables can run — a spec
# can never smuggle arbitrary code in through args.
_check_predicates: dict[str, Callable[[], bool]] = {}
_remedies: dict[str, Callable[[bool], Mapping[str, Any]]] = {}


def register_check_predicate(name: str, fn: Callable[[], bool]) -> None:
    _check_predicates[name] = fn


def register_remedy(name: str, fn: Callable[[bool], Mapping[str, Any]]) -> None:
    """Register a remedy: ``fn(test) -> {result, changes, comment}``."""
    _remedies[name] = fn


def _state_check_predicate(
    test: bool = False, watch_triggered: bool = False, *, name: str, **_: Any
) -> dict[str, Any]:
    """Evaluate a registered pure predicate; never mutates (test-mode equal)."""
    fn = _check_predicates.get(name)
    if fn is None:
        return {"result": False, "changes": {}, "comment": f"unknown check predicate {name!r}"}
    ok = bool(fn())
    return {"result": ok, "changes": {}, "comment": f"check {name!r} {'passed' if ok else 'FAILED'}"}


def _state_remedy_callable(
    test: bool = False, watch_triggered: bool = False, *, name: str, **_: Any
) -> dict[str, Any]:
    """Run a registered remedy (idempotent repair action), honoring dry-run."""
    fn = _remedies.get(name)
    if fn is None:
        return {"result": False, "changes": {}, "comment": f"unknown remedy {name!r}"}
    raw = fn(test)
    return {
        "result": bool(raw.get("result")),
        "changes": dict(raw.get("changes") or {}),
        "comment": str(raw.get("comment") or ""),
    }


def _state_file_directory(
    test: bool = False, watch_triggered: bool = False, *, path: str, **_: Any
) -> dict[str, Any]:
    """Ensure a directory exists (via the governed file-write gateway)."""
    target = Path(path).expanduser()
    if target.is_dir():
        return {"result": True, "changes": {}, "comment": f"{target} already present"}
    if target.exists():
        return {"result": False, "changes": {}, "comment": f"{target} exists and is not a directory"}
    if test:
        return {"result": True, "changes": {"created": str(target)}, "comment": f"would create {target}"}
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("homeostate.file.directory"):
        get_file_write_gateway().ensure_directory(target, source="homeostate")
    return {"result": True, "changes": {"created": str(target)}, "comment": f"created {target}"}


def _state_service_available(
    test: bool = False, watch_triggered: bool = False, *, name: str, **_: Any
) -> dict[str, Any]:
    """Ensure a ServiceContainer service resolves to a live instance."""
    from core.container import get_container

    container = get_container()
    if test:
        # Dry-run must not instantiate: only check registration.
        registered = bool(container.has(name))
        return {
            "result": registered,
            "changes": {},
            "comment": f"service {name!r} {'registered' if registered else 'NOT registered'}",
        }
    instance = container.get(name, default=None)
    ok = instance is not None
    return {
        "result": ok,
        "changes": {},
        "comment": f"service {name!r} {'available' if ok else 'unavailable'}",
    }


# ── Engine ────────────────────────────────────────────────────────────────

class HomeostateEngine:
    """Compile, converge, and report on declared runtime states."""

    def __init__(self, registry: StateModuleRegistry | None = None) -> None:
        self.registry = registry if registry is not None else StateModuleRegistry()
        self._catalog: dict[str, list[StateSpec]] = {}
        self._last_reports: dict[str, HighstateReport] = {}

    # ── catalog ───────────────────────────────────────────────────────

    def define(self, name: str, specs: Sequence[StateSpec]) -> None:
        """Register a named highstate (compiled now so errors surface early)."""
        self._catalog[name] = compile_lowstate(list(specs))

    def catalog(self) -> dict[str, int]:
        return {name: len(specs) for name, specs in self._catalog.items()}

    def last_report(self, name: str) -> HighstateReport | None:
        return self._last_reports.get(name)

    # ── convergence ───────────────────────────────────────────────────

    def apply(self, name: str, *, test: bool = False) -> HighstateReport:
        """Converge one named highstate. ``test=True`` is the honest dry-run."""
        specs = self._catalog.get(name)
        if specs is None:
            raise KeyError(f"unknown highstate {name!r}")
        started = time.monotonic()
        results: list[StateResult] = []
        outcome: dict[str, StateResult] = {}
        for spec in specs:
            result = self._run_one(spec, outcome, test=test)
            outcome[spec.id] = result
            results.append(result)
        report = HighstateReport(
            name=name,
            test=test,
            results=results,
            duration_ms=(time.monotonic() - started) * 1000.0,
            finished_at=time.time(),
        )
        self._last_reports[name] = report
        # Surface convergence truthfully in the subsystem health registry:
        # a failed highstate is a degraded runtime shape, not a hidden retry.
        try:
            from core.runtime.errors import get_subsystem_registry

            health = get_subsystem_registry().register(f"homeostate.{name}")
            if report.ok:
                health.mark_ok()
            else:
                health.mark_degraded(
                    f"failed states: {report.failed}",
                    impact="declared runtime baseline not fully converged",
                )
        except (ImportError, RuntimeError, AttributeError, TypeError):
            pass
        if report.failed:
            record_degradation(
                "homeostate",
                RuntimeError(f"highstate {name!r} failed states: {report.failed}"),
                severity="warning",
                action="reported convergence failure; states left as found",
                extra={"report": {"failed": report.failed, "not_run": report.not_run}},
            )
        return report

    async def apply_async(self, name: str, *, test: bool = False) -> HighstateReport:
        """Thread-offloaded :meth:`apply` for async callers (no on-loop fsync)."""
        return await asyncio.to_thread(self.apply, name, test=test)

    def orchestrate(
        self,
        plan: Sequence[str],
        *,
        stop_on_failure: bool = True,
        test: bool = False,
    ) -> dict[str, HighstateReport]:
        """Salt's orchestrate runner: converge named highstates in order.

        With ``stop_on_failure`` (default), a highstate that does not fully
        converge halts the plan — later stages assume earlier ones hold.
        """
        reports: dict[str, HighstateReport] = {}
        for name in plan:
            report = self.apply(name, test=test)
            reports[name] = report
            if stop_on_failure and not report.ok:
                break
        return reports

    async def orchestrate_async(
        self,
        plan: Sequence[str],
        *,
        stop_on_failure: bool = True,
        test: bool = False,
    ) -> dict[str, HighstateReport]:
        return await asyncio.to_thread(
            self.orchestrate, plan, stop_on_failure=stop_on_failure, test=test
        )

    def _run_one(
        self,
        spec: StateSpec,
        outcome: Mapping[str, StateResult],
        *,
        test: bool,
    ) -> StateResult:
        started = time.monotonic()

        def done(result: bool | None, changes: dict[str, Any], comment: str) -> StateResult:
            return StateResult(
                id=spec.id,
                fn=spec.fn,
                result=result,
                changes=changes,
                comment=comment,
                duration_ms=(time.monotonic() - started) * 1000.0,
            )

        # require/watch: hard gates — upstream must have succeeded.
        for req in tuple(spec.require) + tuple(spec.watch):
            up = outcome.get(req)
            if up is None or up.result is not True:
                return done(None, {}, f"not run: requisite {req!r} did not succeed")
        # onchanges: run only when something upstream actually changed.
        if spec.onchanges and not any(outcome[r].changes for r in spec.onchanges if r in outcome):
            return done(True, {}, "skipped: no upstream changes (onchanges)")
        # onfail: remediation states run only on upstream failure.
        if spec.onfail and not any(
            outcome[r].result is False for r in spec.onfail if r in outcome
        ):
            return done(True, {}, "skipped: no upstream failure (onfail)")

        fn = self.registry.get(spec.fn)
        if fn is None:
            return done(False, {}, f"unknown state function {spec.fn!r}")
        watch_triggered = any(
            outcome[r].changes for r in spec.watch if r in outcome
        )
        attempts = 1 + (0 if test else max(0, min(int(spec.retries), 5)))
        interval = max(0.0, min(float(spec.retry_interval_s), 30.0))
        last_comment = ""
        for attempt in range(attempts):
            if attempt > 0 and interval > 0:
                time.sleep(interval)
            try:
                raw = fn(test=test, watch_triggered=watch_triggered, **dict(spec.args))
            except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
                record_degradation(
                    "homeostate",
                    exc,
                    severity="warning",
                    action=f"state {spec.id!r} raised; marked failed and continued",
                    extra={"state": spec.id, "fn": spec.fn, "attempt": attempt + 1},
                )
                last_comment = f"state function raised: {exc!r}"
                continue
            if bool(raw.get("result")):
                comment = str(raw.get("comment") or "")
                if attempt > 0:
                    comment = f"{comment} (succeeded on retry {attempt})".strip()
                return done(True, dict(raw.get("changes") or {}), comment)
            last_comment = str(raw.get("comment") or "")
        if attempts > 1:
            last_comment = f"{last_comment} (after {attempts} attempts)".strip()
        return done(False, {}, last_comment)


# ── Beacon: degradations → bus events (Salt's beacon system) ──────────────

class DegradationBeacon:
    """Watches the degradation tracker; fires bus events on threshold crossings.

    Salt's beacons watch local state and translate it into events; reactors
    then map events to states. Here the local state is Aura's own degradation
    stream: when a subsystem accumulates enough warning+ records inside the
    window, one ``homeostate.beacon.degradation`` event fires (with a cooldown
    per subsystem so a storm cannot flood the bus).
    """

    TOPIC = "homeostate.beacon.degradation"

    def __init__(
        self,
        *,
        window_s: float = 300.0,
        threshold: int = 5,
        cooldown_s: float = 300.0,
        interval_s: float = 60.0,
    ) -> None:
        self.window_s = float(window_s)
        self.threshold = int(threshold)
        self.cooldown_s = float(cooldown_s)
        self.interval_s = float(interval_s)
        self._last_fired: dict[str, float] = {}
        self._task: asyncio.Task | None = None
        self._running = False
        self.events_fired = 0

    def poll_once(self) -> list[dict[str, Any]]:
        """One beacon poll: which subsystems crossed the threshold now?"""
        from core.runtime.errors import get_degradation_tracker

        counts = get_degradation_tracker().recent_counts_by_subsystem(self.window_s)
        now = time.monotonic()
        events: list[dict[str, Any]] = []
        for subsystem, by_severity in counts.items():
            serious = sum(
                n for sev, n in by_severity.items() if sev in ("warning", "degraded", "critical")
            )
            if serious < self.threshold:
                continue
            last = self._last_fired.get(subsystem, 0.0)
            if now - last < self.cooldown_s:
                continue
            self._last_fired[subsystem] = now
            events.append(
                {
                    "subsystem": subsystem,
                    "serious_count": serious,
                    "window_s": self.window_s,
                    "counts": dict(by_severity),
                    "grains": grains(),
                }
            )
        return events

    async def run(self) -> None:
        """Beacon loop: poll on an interval, publish crossings on the bus."""
        from core.event_bus import get_event_bus

        bus = get_event_bus()
        self._running = True
        while self._running:
            try:
                for event in self.poll_once():
                    await bus.publish(self.TOPIC, event)
                    self.events_fired += 1
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "homeostate_beacon",
                    exc,
                    severity="warning",
                    action="kept beacon loop alive after poll failure",
                )
            await asyncio.sleep(self.interval_s)

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            from core.utils.task_tracker import get_task_tracker

            self._task = get_task_tracker().create_task(self.run(), name="homeostate.beacon")
        return self._task

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._task = None


# ── Reactor: bus events → convergence (Salt's reactor system) ─────────────

@dataclass
class _Reaction:
    topic_pattern: str
    highstate: str
    cooldown_s: float
    last_run: float = 0.0
    runs: int = 0


class HomeostateReactor:
    """Maps event-bus topics to highstate re-convergence, rate-limited."""

    def __init__(self, engine: HomeostateEngine) -> None:
        self.engine = engine
        self._reactions: list[_Reaction] = []
        self._tasks: list[asyncio.Task] = []
        self._topics: dict[str, Any] = {}
        self._active = False
        self.reactions_fired = 0

    def bind(self, topic_pattern: str, highstate: str, *, cooldown_s: float = 120.0) -> None:
        """React to events matching ``topic_pattern`` by converging ``highstate``."""
        self._reactions.append(_Reaction(topic_pattern, highstate, float(cooldown_s)))

    def reactions(self) -> list[dict[str, Any]]:
        return [
            {
                "topic": r.topic_pattern,
                "highstate": r.highstate,
                "cooldown_s": r.cooldown_s,
                "runs": r.runs,
            }
            for r in self._reactions
        ]

    async def _react(self, reaction: _Reaction, event: Any) -> None:
        now = time.monotonic()
        if now - reaction.last_run < reaction.cooldown_s:
            return
        reaction.last_run = now
        reaction.runs += 1
        self.reactions_fired += 1
        report = await self.engine.apply_async(reaction.highstate)
        logger.info(
            "[Homeostate] reactor converged %r after event on %r: ok=%s changed=%s failed=%s",
            reaction.highstate,
            reaction.topic_pattern,
            report.ok,
            report.changed,
            report.failed,
        )

    async def _listen(self, topic: str) -> None:
        from core.event_bus import get_event_bus

        bus = get_event_bus()
        queue = await bus.subscribe(topic)
        try:
            while self._active:
                # Bus queue items are (priority, sequence, {"topic", "data"}).
                # Timed wait (not a bare .get()) so the reactor wakes to
                # re-check self._active and a silently-wedged bus stays
                # visible — the deliberately-infinite-consumer discipline the
                # bounded-await ratchet requires. PriorityQueue.get() is
                # cancel-safe, so an idle timeout drops no event.
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                event = item[2].get("data") if isinstance(item, tuple) and len(item) == 3 else item
                for reaction in self._reactions:
                    if fnmatch.fnmatch(topic, reaction.topic_pattern):
                        try:
                            await self._react(reaction, event)
                        except (RuntimeError, AttributeError, TypeError, ValueError, KeyError) as exc:
                            record_degradation(
                                "homeostate_reactor",
                                exc,
                                severity="warning",
                                action="kept reactor alive after reaction failure",
                                extra={"topic": topic, "highstate": reaction.highstate},
                            )
        except asyncio.CancelledError:
            raise

    def start(self) -> None:
        """Subscribe one listener per distinct concrete topic bound so far."""
        from core.utils.task_tracker import get_task_tracker

        self._active = True
        topics = {r.topic_pattern for r in self._reactions}
        for topic in topics:
            if topic in self._topics:
                continue
            task = get_task_tracker().create_task(
                self._listen(topic), name=f"homeostate.reactor.{topic}"
            )
            self._topics[topic] = task
            self._tasks.append(task)

    async def stop(self) -> None:
        self._active = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, RuntimeError):
                pass
        self._tasks.clear()
        self._topics.clear()


# ── Scheduled convergence (Salt's highstate schedule) ─────────────────────

class ScheduledConvergence:
    """Periodic re-convergence of a named highstate — drift never accumulates.

    Salt runs ``state.highstate`` on a schedule so reality is pulled back to
    the declaration even when no event fires. Same here: a bounded loop that
    re-applies the highstate every ``interval_s`` (beacon/reactor handle the
    acute cases; this handles slow drift).
    """

    def __init__(self, engine: HomeostateEngine, highstate: str, interval_s: float = 1800.0) -> None:
        self.engine = engine
        self.highstate = highstate
        self.interval_s = float(interval_s)
        self._task: asyncio.Task | None = None
        self._running = False
        self.runs = 0

    async def run(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval_s)
            if not self._running:
                break
            try:
                report = await self.engine.apply_async(self.highstate)
                self.runs += 1
                if not report.ok:
                    logger.warning(
                        "[Homeostate] scheduled convergence of %r left failures: %s",
                        self.highstate,
                        report.failed,
                    )
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError, KeyError) as exc:
                record_degradation(
                    "homeostate_schedule",
                    exc,
                    severity="warning",
                    action="kept scheduled convergence loop alive after failure",
                )

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            from core.utils.task_tracker import get_task_tracker

            self._task = get_task_tracker().create_task(
                self.run(), name=f"homeostate.schedule.{self.highstate}"
            )
        return self._task

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._task = None


# ── Default catalog: Aura's declared runtime baseline ─────────────────────

def install_default_catalog(engine: HomeostateEngine) -> None:
    """Declare the baseline shape of a healthy Aura runtime.

    Directories the organism assumes exist (crash forensics, stall and memory
    telemetry) and the spine services that must resolve. The reactor re-applies
    this baseline when the degradation beacon fires — event-driven self-repair
    of the substrate the forensics themselves depend on.
    """
    from core.config import config

    data_dir = Path(getattr(config.paths, "data_dir", Path("data")))
    error_root = data_dir / "error_logs"

    # Cross-fusion trust leg: the baseline includes replaying the proof
    # kernel's theorem ledger — every stored proof must still re-verify from
    # its serialized certificate. Runs at boot and on every reactor
    # re-convergence, so ledger integrity is continuously demonstrated.
    def _proof_kernel_replay_ok() -> bool:
        from core.reasoning.proof_kernel import get_theorem_ledger

        return bool(get_theorem_ledger().replay(limit=64)["ok"])

    register_check_predicate("proof_kernel_replay", _proof_kernel_replay_ok)

    engine.define(
        "runtime_baseline",
        [
            StateSpec(id="error_log_root", fn="file.directory", args={"path": str(error_root)}),
            StateSpec(
                id="crash_forensics_dir",
                fn="file.directory",
                args={"path": str(error_root / "crash")},
                require=("error_log_root",),
            ),
            StateSpec(
                id="stall_forensics_dir",
                fn="file.directory",
                args={"path": str(error_root / "stalls")},
                require=("error_log_root",),
            ),
            StateSpec(
                id="memory_forensics_dir",
                fn="file.directory",
                args={"path": str(error_root / "memory")},
                require=("error_log_root",),
            ),
            StateSpec(id="event_bus_available", fn="service.available", args={"name": "event_bus"}),
            StateSpec(id="atomspace_available", fn="service.available", args={"name": "atomspace"}),
            StateSpec(
                id="proof_kernel_available", fn="service.available", args={"name": "proof_kernel"}
            ),
            StateSpec(
                id="proof_kernel_replay",
                fn="check.predicate",
                args={"name": "proof_kernel_replay"},
                require=("proof_kernel_available",),
            ),
        ],
    )


# ── Singleton + live assembly ─────────────────────────────────────────────

# RLock, load-bearing: get_homeostate_reactor() and
# get_convergence_scheduler() call get_homeostate_engine() WHILE holding
# this lock. With a plain Lock that nested acquire self-deadlocked the
# event loop at boot PHASE 5.2 forever — every desktop boot after the
# triad wave wedged there (reproduced 2026-07-24: 280+s of continuous 5s
# loop stalls, stall dumps bottoming out at `with _engine_lock:`).
_engine_lock = threading.RLock()
_engine: HomeostateEngine | None = None
_reactor: HomeostateReactor | None = None
_beacon: DegradationBeacon | None = None
_scheduler: ScheduledConvergence | None = None


def get_homeostate_engine() -> HomeostateEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = HomeostateEngine()
            install_default_catalog(_engine)
        return _engine


def get_homeostate_reactor() -> HomeostateReactor:
    global _reactor
    with _engine_lock:
        if _reactor is None:
            _reactor = HomeostateReactor(get_homeostate_engine())
            _reactor.bind(DegradationBeacon.TOPIC, "runtime_baseline", cooldown_s=300.0)
        return _reactor


def get_degradation_beacon() -> DegradationBeacon:
    global _beacon
    with _engine_lock:
        if _beacon is None:
            _beacon = DegradationBeacon()
        return _beacon


def get_convergence_scheduler() -> ScheduledConvergence:
    global _scheduler
    with _engine_lock:
        if _scheduler is None:
            _scheduler = ScheduledConvergence(get_homeostate_engine(), "runtime_baseline")
        return _scheduler


def reset_homeostate_for_test() -> HomeostateEngine:
    global _engine, _reactor, _beacon, _scheduler
    with _engine_lock:
        _engine = HomeostateEngine()
        install_default_catalog(_engine)
        _reactor = None
        _beacon = None
        _scheduler = None
        return _engine


async def start_homeostate_runtime() -> dict[str, Any]:
    """Boot hook: apply the baseline once, then start beacon + reactor.

    Called from the orchestrator boot sequence. Returns a summary the boot
    log can record. Never raises — a homeostate failure must not stop boot.
    """
    summary: dict[str, Any] = {"ok": False}
    try:
        engine = get_homeostate_engine()
        report = await engine.apply_async("runtime_baseline")
        reactor = get_homeostate_reactor()
        reactor.start()
        beacon = get_degradation_beacon()
        beacon.start()
        scheduler = get_convergence_scheduler()
        scheduler.start()
        summary = {
            "ok": True,
            "baseline_ok": report.ok,
            "baseline_changed": report.changed,
            "baseline_failed": report.failed,
            "reactions": reactor.reactions(),
            "scheduled_interval_s": scheduler.interval_s,
        }
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError, KeyError) as exc:
        record_degradation(
            "homeostate",
            exc,
            severity="warning",
            action="continued boot without homeostate convergence runtime",
        )
        summary = {"ok": False, "error": repr(exc)}
    return summary


__all__ = [
    "CompileError",
    "DegradationBeacon",
    "HighstateReport",
    "HomeostateEngine",
    "HomeostateReactor",
    "ScheduledConvergence",
    "StateModuleRegistry",
    "StateResult",
    "StateSpec",
    "compile_lowstate",
    "get_convergence_scheduler",
    "get_degradation_beacon",
    "get_homeostate_engine",
    "get_homeostate_reactor",
    "grains",
    "install_default_catalog",
    "register_check_predicate",
    "register_remedy",
    "reset_homeostate_for_test",
    "start_homeostate_runtime",
]
