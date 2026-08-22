"""MemoryWatchdog: out-of-band memory ceiling enforcement.

Every other memory-enforcement path in Aura (MemoryGovernor,
AppleSiliconMemoryMonitor, VRAM purges) runs as an asyncio task. When the
host starts swapping, the event loop stalls — which means the enforcement
paths go blind at exactly the moment they are needed. Observed failure
this module exists to prevent: a single live chat turn pushed the process
tree from ~17 GB RSS to 110 GB, macOS exhausted swap, and the machine
froze while the in-loop governor never got scheduled again.

Like StallWatchdog, this runs in its own daemon thread so it keeps acting
even when the loop is wedged. It enforces a three-stage ladder over the
managed RSS (core process + all child workers):

- soft ceiling: schedule the in-loop MemoryGovernor sweep (graceful
  prune/unload). If the loop is healthy this is the whole story.
- hard ceiling: act from the thread itself, no event loop required —
  terminate heavyweight child workers (mlx/llama/Metal) and force a full
  gc pass.
- lethal ceiling: after consecutive confirmations and a hard action that
  failed to reclaim, write a tombstone with the recent samples and exit
  with a categorized status code. A clean, explained crash that a
  supervisor (or the operator) can restart beats freezing the host.

Swap exhaustion escalates the ladder: high swap with elevated managed RSS
is treated as the hard tier even if RSS alone is under the ceiling.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.process_identity import assert_owned, capture_identity
from core.runtime.resource_observation import (
    ObservationSource,
    ResourceObserver,
    get_resource_observer,
)
from core.utils.memory_monitor import process_memory_bytes

logger = logging.getLogger("Aura.Resilience.MemoryWatchdog")

_WATCHDOG_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    psutil.Error,
)

# Exit status for the lethal path. Chosen to be greppable and distinct:
# EX_SOFTWARE (70) — "internal software error" — categorized OOM abort.
MEMORY_ABORT_EXIT_CODE = 70

def _active_model_worker_handles() -> list[Any]:
    """Parent-owned model-worker handles, selected by gateway role.

    Every multiprocessing child has the same generic command line. Selecting
    by ``spawn_main`` therefore turns state vaults, sensory gates and runtime
    coordinators into collateral damage. Only the role contract attached by
    :class:`SubprocessGateway` authorizes emergency model reclamation.
    """

    try:
        from core.runtime.process_privilege import ProcessRole
        from core.runtime.subprocess_gateway import python_process_role

        children = mp.active_children()
    except (ImportError, AssertionError, OSError, RuntimeError, TypeError, ValueError):
        return []
    return [
        process
        for process in children
        if python_process_role(process) is ProcessRole.MODEL_WORKER
    ]

def _tombstone_dir() -> Path:
    """Resolved per call, not at import.

    A module-level Path("data/error_logs/memory") froze the working directory
    as it was at import time and wrote OOM tombstones relative to it. Resolving
    through the shared forensics root means the reader looks where the writer
    wrote, and an AURA_LOG_DIR set for a hermetic run is actually honoured
    rather than baked over.
    """
    from core.utils.paths import forensics_dir

    return forensics_dir("memory")
_DARWIN_CHILD_LIBPROC: Any | None = None
_DARWIN_CHILD_LIBPROC_UNAVAILABLE = False


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = str(os.environ.get(name, default) or default).strip().lower()
    return value if value in allowed else default


@dataclass(frozen=True)
class MemorySample:
    """One out-of-band measurement of the managed process tree."""

    core_rss_mb: float
    child_rss_mb: float
    swap_used_gb: float
    system_percent: float
    total_ram_gb: float
    sampled_at: float
    #: What is left in the swap file. Corroborating evidence only — on macOS
    #: this runs near zero as a matter of course. See ``_swap_is_exhausted``.
    swap_free_gb: float = 0.0
    #: RAM the host can still hand out. This is the signal that decides it.
    available_gb: float = 0.0
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""

    @property
    def managed_rss_mb(self) -> float:
        return self.core_rss_mb + self.child_rss_mb


@dataclass
class WatchdogAction:
    at: float
    tier: str
    detail: str
    managed_rss_mb: float


@dataclass
class _Thresholds:
    soft_mb: float
    hard_mb: float
    lethal_mb: float
    swap_hard_gb: float
    soft_cooldown_s: float = 30.0
    hard_cooldown_s: float = 60.0
    lethal_confirmations: int = 2
    boot_grace_s: float = 300.0

    @classmethod
    def from_environment(cls, total_ram_gb: float) -> _Thresholds:
        # Daily-use defaults come from the same host-reserve contract as model
        # admission and the governor. Explicit lower operator limits remain
        # valid recovery policy.
        total_mb = max(8192.0, total_ram_gb * 1024.0)
        try:
            from core.runtime.desktop_boot_safety import compute_desktop_memory_envelope

            envelope = compute_desktop_memory_envelope(int(total_mb * 1024 * 1024))
            soft_default = envelope.watchdog_soft_mb
            hard_default = envelope.watchdog_hard_mb
            lethal_default = envelope.watchdog_lethal_mb
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            soft_default = min(32768.0, total_mb * 0.50)
            hard_default = min(40960.0, total_mb * 0.62)
            lethal_default = min(46080.0, total_mb * 0.70)
        return cls(
            soft_mb=_env_float("AURA_MEMWATCH_SOFT_MB", soft_default),
            hard_mb=_env_float("AURA_MEMWATCH_HARD_MB", hard_default),
            lethal_mb=_env_float("AURA_MEMWATCH_LETHAL_MB", lethal_default),
            swap_hard_gb=_env_float(
                "AURA_MEMWATCH_SWAP_HARD_GB",
                min(8.0, max(2.0, total_ram_gb * 0.12)),
            ),
            soft_cooldown_s=_env_float("AURA_MEMWATCH_SOFT_COOLDOWN_S", 30.0),
            hard_cooldown_s=_env_float("AURA_MEMWATCH_HARD_COOLDOWN_S", 60.0),
            lethal_confirmations=max(
                2, int(_env_float("AURA_MEMWATCH_LETHAL_CONFIRMS", 2.0))
            ),
            boot_grace_s=_env_float("AURA_MEMWATCH_BOOT_GRACE_S", 300.0),
        )


def _phys_footprint_mb(pid: int) -> float:
    """Return the canonical RSS/phys-footprint memory sample in MB."""
    try:
        return float(process_memory_bytes(pid)) / float(1024 * 1024)
    except _WATCHDOG_RECOVERABLE_ERRORS:
        return 0.0


def _darwin_child_pids(root_pid: int, *, recursive: bool, max_children: int = 64) -> list[int]:
    """Return child pids via libproc on macOS without psutil's full ppid map.

    A live stall trace showed ``psutil.Process.children(recursive=True)`` stuck
    in the watchdog thread while the event loop was already wedged. On Darwin,
    ``proc_listchildpids`` gives a bounded direct-child query without a global
    process-table ppid map or a production raw-subprocess surface.
    """

    global _DARWIN_CHILD_LIBPROC, _DARWIN_CHILD_LIBPROC_UNAVAILABLE
    if sys.platform != "darwin" or _DARWIN_CHILD_LIBPROC_UNAVAILABLE:
        return []
    seen: set[int] = set()
    frontier = [int(root_pid)]
    deadline = time.monotonic() + 0.75
    try:
        if _DARWIN_CHILD_LIBPROC is None:
            _DARWIN_CHILD_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
            _DARWIN_CHILD_LIBPROC.proc_listchildpids.argtypes = [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            _DARWIN_CHILD_LIBPROC.proc_listchildpids.restype = ctypes.c_int
    except (AttributeError, OSError, TypeError, ValueError):
        _DARWIN_CHILD_LIBPROC_UNAVAILABLE = True
        return []

    while frontier and len(seen) < max_children and time.monotonic() < deadline:
        parent = frontier.pop(0)
        try:
            buffer = (ctypes.c_int * max_children)()
            count = int(
                _DARWIN_CHILD_LIBPROC.proc_listchildpids(
                    int(parent),
                    ctypes.byref(buffer),
                    ctypes.sizeof(buffer),
                )
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, ctypes.ArgumentError):
            _DARWIN_CHILD_LIBPROC_UNAVAILABLE = True
            break
        if count <= 0:
            break
        for raw_pid in list(buffer)[: min(count, max_children)]:
            pid = int(raw_pid)
            if pid <= 0:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            if recursive and len(seen) < max_children:
                frontier.append(pid)
    return list(seen)


def _child_processes(root_pid: int, *, recursive: bool = True) -> list[psutil.Process]:
    if sys.platform == "darwin":
        return [psutil.Process(pid) for pid in _darwin_child_pids(root_pid, recursive=recursive)]
    return psutil.Process(root_pid).children(recursive=recursive)


def default_sampler(*, observer: ResourceObserver | None = None) -> MemorySample:
    resource_observer = observer or get_resource_observer()
    memory = resource_observer.memory()
    provenance = memory.provenance
    core_rss = float(memory.process_rss_bytes) / float(1024**2)
    managed_rss = float(memory.process_tree_rss_bytes) / float(1024**2)
    child_rss = max(0.0, managed_rss - core_rss)
    if provenance.source in {ObservationSource.HOST, ObservationSource.LIVE_PRESSURE}:
        # Compression-aware enhancement for the live Darwin adapter.
        core_rss = max(core_rss, _phys_footprint_mb(os.getpid()))
        child_rss = 0.0
        try:
            for child in _child_processes(os.getpid(), recursive=True):
                try:
                    child_rss += max(
                        child.memory_info().rss / (1024 * 1024),
                        _phys_footprint_mb(child.pid),
                    )
                except _WATCHDOG_RECOVERABLE_ERRORS:
                    continue
        except _WATCHDOG_RECOVERABLE_ERRORS:
            child_rss = max(0.0, managed_rss - core_rss)
    swap_used_gb = float(memory.swap_used_bytes) / float(1024**3)
    swap_free_gb = float(getattr(memory, "swap_free_bytes", 0) or 0) / float(1024**3)
    available_gb = float(getattr(memory, "available_bytes", 0) or 0) / float(1024**3)
    system_percent = float(memory.percent) if memory.available else 100.0
    total_ram_gb = float(memory.total_bytes) / float(1024**3)
    return MemorySample(
        core_rss_mb=core_rss,
        child_rss_mb=child_rss,
        swap_used_gb=swap_used_gb,
        swap_free_gb=swap_free_gb,
        available_gb=available_gb,
        system_percent=system_percent,
        total_ram_gb=total_ram_gb,
        sampled_at=time.time(),
        observation_source=provenance.source.value,
        observation_scenario_id=provenance.scenario_id,
    )


def _swap_is_exhausted(sample: MemorySample, thresholds: _Thresholds) -> bool:
    """Is the host actually out of memory, or is this just macOS being macOS?

    Two readings had to be discarded before this one, and both were discarded
    against measurements taken on the live host while it was perfectly
    responsive.

    ``swap used`` was the original test. It is a high-water mark: pages written
    to the swap file stay accounted there long after the pressure that caused
    them is gone. It latched at 17.9 GB against a 7.7 GB threshold and declared
    an emergency on every check, for an event an hour past.

    ``swap free`` replaced it, and was better but still wrong. macOS sizes the
    swap file dynamically and runs it close to full by design — measured here
    at 1.10 GB free of 11.8 GB, having grown to 18.4 GB and shrunk back on its
    own. Low swap headroom on this platform is ordinary housekeeping.

    What decides it is RAM the host can still hand out, because that is what
    the next allocation draws on and what stalls the event loop when it runs
    out. At the moment of the loudest false alarm: 31.6 GB available, 54% used.
    Nothing was wrong.

    So availability is the gate, and swap headroom is corroboration: both have
    to be tight before this is an emergency.
    """
    free_floor_gb = max(1.0, thresholds.swap_hard_gb * 0.25)
    swap_tight = (
        sample.swap_free_gb <= free_floor_gb
        if sample.swap_free_gb > 0.0
        else sample.swap_used_gb >= thresholds.swap_hard_gb
    )
    if not swap_tight:
        return False
    if sample.available_gb <= 0.0:
        # Nothing to gate on; fall back to the swap evidence alone rather than
        # going blind. A stale signal beats a watchdog that can never fire.
        return True
    # A host with real headroom is not in a memory emergency, whatever the
    # swap file happens to look like.
    return sample.available_gb <= max(2.0, sample.total_ram_gb * 0.08)


def _shed_registered_organs() -> tuple[int, int]:
    """Pull every rung on the OOM ladder. Returns (organs_shed, bytes_freed).

    The ladder and this watchdog were two independent answers to memory pressure
    that did not know about each other. Measured live: "swap exhaustion: managed
    RSS 33868MB swap 8.3GB ... terminated 0 heavy workers and forced gc
    out-of-band" — nothing to kill, nothing reclaimed — while the prompt KV
    cache sat registered as sheddable with a bounded ~3GB footprint.

    Best-effort and never fatal: a reclaim path that raises under memory
    pressure is worse than one that frees nothing.
    """

    try:
        from core.runtime.oom_policy import get_oom_policy
    except (ImportError, AttributeError):
        return 0, 0
    try:
        policy = get_oom_policy()
        shed_all = getattr(policy, "shed_all", None)
        if callable(shed_all):
            result = shed_all()
            if isinstance(result, tuple) and len(result) == 2:
                return int(result[0]), int(result[1])
        organs = 0
        freed = 0
        for policy_entry in list(getattr(policy, "_organs", {}).values()):
            shed = getattr(policy_entry, "shed", None)
            if not callable(shed):
                continue
            try:
                released = int(shed() or 0)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                continue
            if released > 0:
                organs += 1
                freed += released
        return organs, freed
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return 0, 0


def terminate_heavy_child_workers(
    grace_s: float = 2.0, *, free_at_least_bytes: int | None = None
) -> int:
    """Terminate inference child workers out-of-band. Returns count killed.

    A WORKER IS IDENTIFIED BY ITS DECLARED ROLE AND SIZED BY ITS FOOTPRINT.

    Command-line matching first missed the resident worker because it appeared
    only as ``multiprocessing.spawn``. Broadening that marker then made every
    spawned organ killable. The gateway's parent-owned process handle carries
    the exact role declared before start; physical footprint captures Metal and
    compressed allocations that ordinary RSS omits.

    ``free_at_least_bytes`` kills largest-first and stops as soon as that much
    has been given back, so getting under the ceiling costs the fewest workers
    it can — a reload of one model instead of every child in the tree.
    """
    candidates: list[tuple[int, Any, Any, str]] = []
    for process in _active_model_worker_handles():
        try:
            pid = int(process.pid)
            alive = bool(process.is_alive())
            name = str(getattr(process, "name", "model_worker") or "model_worker")
        except (AssertionError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            continue
        if pid <= 0 or not alive:
            continue
        identity = capture_identity(process, label=name)
        if identity is None or not identity.bound:
            logger.warning(
                "🛡️ [MEMWATCH] Refused unbound model-worker reclaim pid=%s name=%s",
                pid,
                name,
            )
            continue
        footprint = max(0, int(_phys_footprint_mb(pid) * 1024 * 1024))
        candidates.append((footprint, process, identity, name))

    candidates.sort(key=lambda item: item[0], reverse=True)

    killed = 0
    freed = 0
    doomed: list[tuple[Any, Any, str]] = []
    for footprint, process, identity, name in candidates:
        if free_at_least_bytes is not None and freed >= free_at_least_bytes:
            break
        if not assert_owned(
            identity,
            process,
            action="terminate model worker",
            subsystem="memory_watchdog",
        ):
            continue
        try:
            process.terminate()
        except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
            continue
        doomed.append((process, identity, name))
        killed += 1
        freed += footprint
        logger.warning(
            "🛑 [MEMWATCH] Terminated declared model worker pid=%s "
            "footprint=%dMB name=%s",
            process.pid,
            footprint >> 20,
            name,
        )
    for process, identity, _name in doomed:
        try:
            process.join(timeout=max(0.0, float(grace_s)))
            alive = bool(process.is_alive())
        except (AssertionError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            alive = True
        if not alive or not assert_owned(
            identity,
            process,
            action="kill model worker after terminate timeout",
            subsystem="memory_watchdog",
        ):
            continue
        try:
            process.kill()
            process.join(timeout=max(0.0, min(float(grace_s), 1.0)))
        except (AssertionError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            continue
    return killed


class MemoryWatchdog(threading.Thread):
    """Daemon thread enforcing hard memory ceilings independent of the loop."""

    # Soft pressure is an incident, not a metronome. A resident model can remain
    # near a policy boundary for hours; repeatedly kicking the same in-loop
    # governor without new evidence only adds event-loop work and floods the
    # operator stream. Rearm after recovery or material worsening. Hard and
    # lethal tiers remain level-triggered and are never suppressed by this.
    SOFT_REARM_GROWTH_MB = 1024.0
    SOFT_REARM_SYSTEM_PERCENT = 3.0
    SOFT_CLEAR_MARGIN_MB = 512.0
    SOFT_CLEAR_SYSTEM_PERCENT = 90.0

    def __init__(
        self,
        *,
        loop: Any = None,
        governor: Any = None,
        sample_interval_s: float | None = None,
        thresholds: _Thresholds | None = None,
        lethal_action: str | None = None,
        sampler: Callable[[], MemorySample] | None = None,
        worker_terminator: Callable[[], int] | None = None,
        gc_collect: Callable[[], int] | None = None,
        ladder_shed: Callable[[], tuple[int, int]] | None = None,
        process_exit: Callable[[int], None] | None = None,
    ):
        super().__init__(daemon=True, name="AuraMemoryWatchdog")
        self._loop = loop
        self._governor = governor
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
        except _WATCHDOG_RECOVERABLE_ERRORS:
            total_ram_gb = 64.0
        self.thresholds = thresholds or _Thresholds.from_environment(total_ram_gb)
        self.sample_interval_s = sample_interval_s if sample_interval_s is not None else _env_float(
            "AURA_MEMWATCH_INTERVAL_S", 3.0
        )
        self.lethal_action = lethal_action or _env_choice(
            "AURA_MEMWATCH_LETHAL_ACTION", "exit", ("exit", "shed", "off")
        )
        self._sampler = sampler or default_sampler
        self._worker_terminator = worker_terminator or terminate_heavy_child_workers
        self._gc_collect = gc_collect or gc.collect
        self._ladder_shed = ladder_shed or _shed_registered_organs
        self._process_exit = process_exit or self._default_process_exit
        self._stop_event = threading.Event()
        self._started_at = time.monotonic()
        self._last_soft_action_at = 0.0
        self._soft_incident_active = False
        self._soft_incident_managed_mb = 0.0
        self._soft_incident_system_percent = 0.0
        self._last_hard_action_at = 0.0
        self._spike_count = 0
        self._spike_dumps = 0
        self._last_spike_dump_at = 0.0
        self._lethal_streak = 0
        self._hard_attempted_in_streak = False
        self._last_sample: MemorySample | None = None
        self._actions: list[WatchdogAction] = []
        self._tick_failures = 0

    # ── public surface ────────────────────────────────────────────────

    def health_snapshot(self) -> dict[str, Any]:
        sample = self._last_sample
        return {
            "running": self.is_alive(),
            "lethal_action": self.lethal_action,
            "sample_interval_s": self.sample_interval_s,
            "thresholds": asdict(self.thresholds),
            "tick_failures": self._tick_failures,
            "lethal_streak": self._lethal_streak,
            "soft_incident": {
                "active": self._soft_incident_active,
                "last_action_managed_mb": self._soft_incident_managed_mb,
                "last_action_system_percent": self._soft_incident_system_percent,
            },
            "last_sample": asdict(sample) if sample else None,
            "recent_actions": [asdict(a) for a in self._actions[-10:]],
        }

    @property
    def last_sample(self) -> MemorySample | None:
        return self._last_sample

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self is threading.current_thread() or not self.is_alive():
            return
        self.join(timeout=max(0.0, float(timeout_s)))
        if self.is_alive():
            raise TimeoutError(
                f"memory watchdog did not stop within {float(timeout_s):.1f}s"
            )

    # ── thread loop ───────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(
            "🛡️ MemoryWatchdog active (soft=%.0fMB hard=%.0fMB lethal=%.0fMB "
            "swap_hard=%.1fGB interval=%.1fs lethal_action=%s)",
            self.thresholds.soft_mb,
            self.thresholds.hard_mb,
            self.thresholds.lethal_mb,
            self.thresholds.swap_hard_gb,
            self.sample_interval_s,
            self.lethal_action,
        )
        while not self._stop_event.is_set():
            try:
                self._tick()
                self._tick_failures = 0
            except _WATCHDOG_RECOVERABLE_ERRORS as exc:
                self._tick_failures += 1
                record_degradation(
                    "memory_watchdog",
                    exc,
                    severity="warning",
                    action="kept out-of-band memory watchdog alive after tick failure",
                )
                logger.debug("MemoryWatchdog tick failed: %s", exc)
            # Adaptive cadence: past the hard ceiling every second counts —
            # a runaway in-process allocation can add gigabytes between
            # relaxed samples.
            wait_s = self.sample_interval_s
            sample = self._last_sample
            if sample is not None and sample.managed_rss_mb >= self.thresholds.hard_mb:
                wait_s = min(1.0, wait_s)
            self._stop_event.wait(wait_s)

    # ── policy ────────────────────────────────────────────────────────

    # A routine MLX generation wires ~20GB in one sample interval; without
    # a throttle the spike dumper wrote 1,568 identical stack dumps (55MB)
    # in one live afternoon. First occurrences keep full diagnostics; the
    # steady state costs one counter increment.
    SPIKE_DUMP_MIN_INTERVAL_S = 600.0
    SPIKE_DUMP_LIFETIME_CAP = 12

    def _tick(self) -> None:
        sample = self._sampler()
        previous = self._last_sample
        self._last_sample = sample
        if (
            previous is not None
            # ResourceObserver can upgrade from a conservative/unavailable
            # bootstrap estimate to the live host adapter. The resulting
            # numerical jump is a change of instrument, not an allocation.
            # Compare deltas only within one observation provenance.
            and sample.observation_source == previous.observation_source
            and sample.observation_scenario_id == previous.observation_scenario_id
            and (sample.managed_rss_mb - previous.managed_rss_mb) > 8192.0
        ):
            in_boot_grace = (
                time.monotonic() - self._started_at
            ) < self.thresholds.boot_grace_s
            if in_boot_grace and sample.managed_rss_mb < self.thresholds.soft_mb:
                logger.info(
                    "[MEMWATCH] Planned boot footprint growth %.0f→%.0fMB remained "
                    "below the %.0fMB soft ceiling; retaining samples without "
                    "incident diagnostics.",
                    previous.managed_rss_mb,
                    sample.managed_rss_mb,
                    self.thresholds.soft_mb,
                )
            else:
                self._record_footprint_spike(previous, sample)
        self._evaluate(sample, time.monotonic())

    def _log_memory_attribution(self, why: str) -> None:
        """Name what grew.

        A thread dump says where threads ARE. It cannot say what allocated,
        and on 2026-07-29 that distinction cost the diagnosis: 20.4GB
        appeared in this process in ten seconds and the only thread running
        at both samples was a MiniLM encode measured afterwards at 3.7MB per
        two thousand calls. The stacks named a bystander.

        memory-infra answers the allocation question — components declare
        what they hold, and unattributed bytes are reported as their own
        number. Dumps were already being taken on pressure and at boot; the
        diff between them was never read out, so the attribution existed and
        nobody asked for it. This asks.

        Unlike the stack dump this is not throttled: BACKGROUND providers
        self-report, which is the cheapness the module was designed around,
        and the spike that goes unattributed is the expensive one.
        """
        try:
            from core.runtime.memory_infra import DetailLevel, get_memory_infra

            infra = get_memory_infra()
            infra.dump(DetailLevel.LIGHT)
            report = infra.leak_report()
            if not report.get("available"):
                logger.warning(
                    "[MEMWATCH] %s — no attribution available (%s).",
                    why,
                    report.get("reason", "unknown"),
                )
                return
            logger.warning(
                "[MEMWATCH] %s — attribution: %s | unattributed %+.0fMB",
                why,
                report.get("narrative", ""),
                float(report.get("unattributed_growth_bytes", 0) or 0) / 1e6,
            )
        except (ImportError, *_WATCHDOG_RECOVERABLE_ERRORS):
            logger.debug("memory attribution failed", exc_info=True)

    def _record_footprint_spike(self, previous: MemorySample, sample: MemorySample) -> None:
        self._spike_count += 1
        why = (
            f"footprint spike {previous.managed_rss_mb:.0f}→"
            f"{sample.managed_rss_mb:.0f}MB in one interval "
            f"(spike #{self._spike_count} this process)"
        )
        self._log_memory_attribution(why)
        now = time.monotonic()
        if self._spike_dumps >= self.SPIKE_DUMP_LIFETIME_CAP:
            if self._spike_dumps == self.SPIKE_DUMP_LIFETIME_CAP:
                self._spike_dumps += 1
                logger.warning(
                    "[MEMWATCH] %s — lifetime stack-dump cap (%d) reached; "
                    "further spikes are counted but not dumped.",
                    why,
                    self.SPIKE_DUMP_LIFETIME_CAP,
                )
            return
        if (
            self._last_spike_dump_at
            and (now - self._last_spike_dump_at) < self.SPIKE_DUMP_MIN_INTERVAL_S
        ):
            logger.info("[MEMWATCH] %s — stack dump throttled.", why)
            return
        self._last_spike_dump_at = now
        self._spike_dumps += 1
        self._dump_thread_stacks(why)

    def _evaluate(self, sample: MemorySample, now: float) -> str:
        """Apply the escalation ladder to one sample. Returns the tier acted on."""
        managed = sample.managed_rss_mb
        t = self.thresholds

        # A resident model can legitimately sit below the host-derived soft
        # rung while still owning most of the machine. If both RAM and swap are
        # genuinely exhausted, do not wait for an arbitrary process threshold;
        # require a substantial Aura footprint so another application's
        # pressure still cannot make us kill an idle runtime.
        emergency_managed_floor = min(t.soft_mb, sample.total_ram_gb * 1024.0 * 0.65)
        swap_escalation = _swap_is_exhausted(sample, t) and managed >= emergency_managed_floor

        if managed >= t.lethal_mb:
            return self._handle_lethal(sample, now)
        self._lethal_streak = 0
        self._hard_attempted_in_streak = False

        if managed >= t.hard_mb or swap_escalation:
            return self._handle_hard(sample, now, swap_escalation=swap_escalation)

        if managed >= t.soft_mb or sample.system_percent >= 92.0:
            return self._handle_soft(sample, now)

        if (
            self._soft_incident_active
            and managed <= (t.soft_mb - self.SOFT_CLEAR_MARGIN_MB)
            and sample.system_percent <= self.SOFT_CLEAR_SYSTEM_PERCENT
        ):
            self._soft_incident_active = False
            self._soft_incident_managed_mb = 0.0
            self._soft_incident_system_percent = 0.0

        return "none"

    def _handle_soft(self, sample: MemorySample, now: float) -> str:
        materially_worse = (
            not self._soft_incident_active
            or sample.managed_rss_mb
            >= self._soft_incident_managed_mb + self.SOFT_REARM_GROWTH_MB
            or sample.system_percent
            >= self._soft_incident_system_percent + self.SOFT_REARM_SYSTEM_PERCENT
        )
        if not materially_worse:
            return "soft_stable"
        if (now - self._last_soft_action_at) < self.thresholds.soft_cooldown_s:
            return "soft_cooldown"
        self._last_soft_action_at = now
        self._soft_incident_active = True
        self._soft_incident_managed_mb = sample.managed_rss_mb
        self._soft_incident_system_percent = sample.system_percent
        self._remember("soft", sample, "scheduled governor sweep")
        logger.warning(
            "⚠️ [MEMWATCH] Soft ceiling: managed RSS %.0fMB (sys %.1f%%). "
            "Scheduling governor sweep.",
            sample.managed_rss_mb,
            sample.system_percent,
        )
        self._schedule_governor_sweep()
        return "soft"

    def _handle_hard(
        self, sample: MemorySample, now: float, *, swap_escalation: bool
    ) -> str:
        if (now - self._last_hard_action_at) < self.thresholds.hard_cooldown_s:
            return "hard_cooldown"
        self._last_hard_action_at = now
        reason = "swap exhaustion" if swap_escalation else "hard RSS ceiling"
        self._dump_thread_stacks(f"hard tier at {sample.managed_rss_mb:.0f}MB")
        logger.critical(
            "🚨 [MEMWATCH] %s: managed RSS %.0fMB swap %.1fGB. "
            "Out-of-band reclaim (terminate heavy workers + gc).",
            reason,
            sample.managed_rss_mb,
            sample.swap_used_gb,
        )
        # Shed caches BEFORE killing workers. The OOM ladder carries organs that
        # can give memory back for free — the prompt KV cache alone is bounded at
        # ~3GB — and this path did not consult it, so the live log read
        # "terminated 0 heavy workers and forced gc out-of-band" while several
        # gigabytes sat registered as sheddable. A rung nothing pulls is not a
        # rung. Killing a worker costs a model reload; dropping a cache costs a
        # re-prefill.
        shed_organs, shed_bytes = self._ladder_shed()
        killed = self._terminate_workers(sample, already_freed=shed_bytes)
        collected = self._gc_collect()
        self._remember(
            "hard",
            sample,
            f"{reason}: shed={shed_organs} organs/{shed_bytes >> 20}MB "
            f"killed={killed} gc_collected={collected}",
        )
        record_degradation(
            "memory_watchdog",
            RuntimeError(
                f"{reason}: managed RSS {sample.managed_rss_mb:.0f}MB, "
                f"swap {sample.swap_used_gb:.1f}GB"
            ),
            severity="critical",
            action=(
                f"shed {shed_organs} cache organ(s) freeing {shed_bytes >> 20}MB, "
                f"terminated {killed} heavy workers, forced gc out-of-band"
            ),
        )
        # Also nudge the graceful path in case the loop is still breathing.
        self._schedule_governor_sweep()
        return "hard"

    def _terminate_workers(self, sample: MemorySample, *, already_freed: int = 0) -> int:
        """Kill workers, preferring the fewest that clear the RSS breach.

        Only an RSS breach can say how many bytes are wanted. The hard tier
        also fires on swap exhaustion, where RSS sits under the ceiling and
        the shortfall is meaningless — asking for zero bytes there would kill
        nothing and leave the rung as empty as the bug this replaced. When
        there is no number to aim at, shed every eligible worker, which is
        what this tier did before.
        """
        shortfall = int(
            (sample.managed_rss_mb - self.thresholds.hard_mb) * (1024 * 1024)
        ) - int(already_freed)
        try:
            if shortfall > 0:
                return self._worker_terminator(free_at_least_bytes=shortfall)
            return self._worker_terminator()
        except TypeError:
            # An injected terminator (tests, older callers) that does not take
            # a budget still gets to run — it just sheds everything it knows.
            return self._worker_terminator()

    def _handle_lethal(self, sample: MemorySample, now: float) -> str:
        self._lethal_streak += 1
        in_boot_grace = (now - self._started_at) < self.thresholds.boot_grace_s

        if not self._hard_attempted_in_streak:
            # Always try reclaiming before considering the terminal action.
            self._hard_attempted_in_streak = True
            self._last_hard_action_at = now
            shed_organs, shed_bytes = self._ladder_shed()
            killed = self._terminate_workers(sample, already_freed=shed_bytes)
            collected = self._gc_collect()
            self._remember(
                "lethal_reclaim",
                sample,
                f"pre-abort reclaim: shed={shed_organs} organs/{shed_bytes >> 20}MB "
                f"killed={killed} gc_collected={collected}",
            )
            # Report all three levers, not just the kill. "Reclaimed
            # (killed=0)" reads as "nothing was reclaimed" while hiding
            # whether shedding and gc found anything — on 2026-07-29 that
            # line was the operator's only view of the last action before
            # the process exited, and it described a third of it.
            logger.critical(
                "🚨 [MEMWATCH] LETHAL ceiling: managed RSS %.0fMB ≥ %.0fMB. "
                "Reclaimed: shed=%d organs/%dMB killed=%d gc=%d. "
                "Next confirmation aborts.",
                sample.managed_rss_mb,
                self.thresholds.lethal_mb,
                shed_organs,
                shed_bytes >> 20,
                killed,
                collected,
            )
            return "lethal_reclaim"

        if self._lethal_streak < self.thresholds.lethal_confirmations + 1:
            return "lethal_pending"

        if self.lethal_action == "off" or in_boot_grace:
            self._remember(
                "lethal_suppressed",
                sample,
                "boot grace" if in_boot_grace else "lethal_action=off",
            )
            logger.critical(
                "🚨 [MEMWATCH] Lethal ceiling persists (%.0fMB) but abort is "
                "suppressed (%s).",
                sample.managed_rss_mb,
                "boot grace" if in_boot_grace else "lethal_action=off",
            )
            return "lethal_suppressed"

        if self.lethal_action == "shed":
            self._last_hard_action_at = now
            killed = self._worker_terminator()
            self._remember("lethal_shed", sample, f"repeat shed: killed={killed}")
            return "lethal_shed"

        # lethal_action == "exit": categorized abort.
        self._write_tombstone(sample)
        self._remember("lethal_exit", sample, f"exit({MEMORY_ABORT_EXIT_CODE})")
        logger.critical(
            "💀 [MEMWATCH] Managed RSS %.0fMB exceeded lethal ceiling %.0fMB "
            "after reclaim attempts. Aborting with exit code %d to protect "
            "the host (tombstone written).",
            sample.managed_rss_mb,
            self.thresholds.lethal_mb,
            MEMORY_ABORT_EXIT_CODE,
        )
        self._process_exit(MEMORY_ABORT_EXIT_CODE)
        return "lethal_exit"

    # ── helpers ───────────────────────────────────────────────────────

    def _dump_thread_stacks(self, why: str) -> None:
        """Snapshot every thread's stack at memory-spike time.

        The 78GB compressed runaway died with the allocator anonymous.
        At hard tier the allocator is, with high probability, ON one of
        these stacks — faulthandler writes them without allocating
        Python objects, safe under pressure.
        """
        try:
            import faulthandler

            from core.utils.paths import forensics_dir

            spike_log = forensics_dir("crash") / "memory_spike_stacks.log"
            try:
                # Allocation-light rotation: a stat + rename keeps the log
                # bounded (observed 54MB unrotated growth in live use).
                if spike_log.exists() and spike_log.stat().st_size > 16 * 1024 * 1024:
                    spike_log.replace(spike_log.with_suffix(".log.1"))
            except OSError:
                pass
            with open(spike_log, "a") as fh:
                fh.write(f"\n===== {why} pid={os.getpid()} at={time.time()} =====\n")
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("MemoryWatchdog stack dump failed: %s", exc)

    def _remember(self, tier: str, sample: MemorySample, detail: str) -> None:
        self._actions.append(
            WatchdogAction(
                at=time.time(),
                tier=tier,
                detail=detail[:240],
                managed_rss_mb=sample.managed_rss_mb,
            )
        )
        if len(self._actions) > 50:
            self._actions = self._actions[-50:]

    def _schedule_governor_sweep(self) -> None:
        loop = self._loop
        governor = self._governor
        if loop is None or governor is None:
            return
        try:
            if loop.is_closed():
                return

            def _kick() -> None:
                try:
                    from core.runtime.task_ownership import create_tracked_task

                    create_tracked_task(
                        governor.check(), name="memory_watchdog.governor_sweep"
                    )
                except _WATCHDOG_RECOVERABLE_ERRORS as exc:
                    logger.debug("MemoryWatchdog governor kick failed: %s", exc)

            loop.call_soon_threadsafe(_kick)
        except RuntimeError:
            return
        except _WATCHDOG_RECOVERABLE_ERRORS as exc:
            logger.debug("MemoryWatchdog could not schedule governor sweep: %s", exc)

    def _write_tombstone(self, sample: MemorySample) -> None:
        payload = {
            "schema": "aura.memory_watchdog.tombstone.v1",
            "reason": "managed RSS exceeded lethal ceiling after reclaim attempts",
            "exit_code": MEMORY_ABORT_EXIT_CODE,
            "written_at": time.time(),
            "thresholds": asdict(self.thresholds),
            "final_sample": asdict(sample),
            "recent_actions": [asdict(a) for a in self._actions[-20:]],
        }
        try:
            from core.runtime.atomic_writer import atomic_write_json

            tombstone_dir = _tombstone_dir()
            path = tombstone_dir / f"oom_tombstone_{int(time.time())}.json"
            # Approved emergency writer: atomic_writer is an audited file
            # sink with no governed-gateway machinery to starve under OOM,
            # and a torn tombstone would be worse than none.
            atomic_write_json(
                path,
                payload,
                schema_version=1,
                schema_name="aura.memory_watchdog.tombstone",
            )
            logger.critical("💀 [MEMWATCH] Tombstone written: %s", path)
        except (OSError, RuntimeError, ImportError, TypeError, ValueError) as exc:
            logger.critical("💀 [MEMWATCH] Tombstone write failed: %s", exc)

    @staticmethod
    def _default_process_exit(code: int) -> None:
        # No logging.shutdown() here: flushing handlers can block
        # indefinitely under swap thrash — observed in the 115GB crash
        # where the lethal path never reached exit. The tombstone is
        # already on disk; die immediately.
        os._exit(code)


_WATCHDOG_SINGLETON: MemoryWatchdog | None = None
_WATCHDOG_LOCK = threading.Lock()


def get_memory_watchdog() -> MemoryWatchdog | None:
    return _WATCHDOG_SINGLETON


def start_memory_watchdog(*, loop: Any = None, governor: Any = None) -> MemoryWatchdog:
    """Start (or return) the process-wide memory watchdog thread."""
    global _WATCHDOG_SINGLETON
    with _WATCHDOG_LOCK:
        existing = _WATCHDOG_SINGLETON
        if existing is not None and existing.is_alive():
            if governor is not None:
                existing._governor = governor
            if loop is not None:
                existing._loop = loop
            return existing
        watchdog = MemoryWatchdog(loop=loop, governor=governor)
        watchdog.start()
        _WATCHDOG_SINGLETON = watchdog
        return watchdog


def stop_memory_watchdog() -> None:
    global _WATCHDOG_SINGLETON
    with _WATCHDOG_LOCK:
        if _WATCHDOG_SINGLETON is not None:
            _WATCHDOG_SINGLETON.stop()
            _WATCHDOG_SINGLETON = None
