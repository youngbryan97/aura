import asyncio
import contextlib
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass

import psutil

from core.runtime.errors import record_degradation
from core.runtime.process_footprint import (
    DarwinRUsageInfoV4,
    current_darwin_footprint_bytes,
    darwin_phys_footprint_bytes,
)
from core.runtime.resource_observation import (
    ObservationSource,
    ResourceObserver,
    get_resource_observer,
)

logger = logging.getLogger("Aura.MemoryMonitor")

MEMORY_PRESSURE_NORMAL = "normal"
MEMORY_PRESSURE_WARN = "warn"
MEMORY_PRESSURE_CRITICAL = "critical"
MEMORY_PRESSURE_UNKNOWN = "unknown"

_KERNEL_PRESSURE_LEVELS = {
    1: MEMORY_PRESSURE_NORMAL,
    2: MEMORY_PRESSURE_WARN,
    4: MEMORY_PRESSURE_CRITICAL,
}
_KERNEL_PRESSURE_CACHE: tuple[float, str] = (0.0, MEMORY_PRESSURE_UNKNOWN)
_KERNEL_PRESSURE_TTL_S = 2.0
_KERNEL_PRESSURE_LOCK = threading.Lock()

_MEMORY_MONITOR_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    psutil.Error,
)

_GIB = float(1024**3)

# Compatibility names retained for focused ABI tests and existing monkeypatches.
_DarwinRUsageInfoV4 = DarwinRUsageInfoV4
_current_darwin_footprint_bytes = current_darwin_footprint_bytes
_darwin_phys_footprint_bytes = darwin_phys_footprint_bytes


_SNAPSHOT_CACHE_LOCK = threading.Lock()
_SNAPSHOT_CACHE: tuple[float, tuple[int, str, str], "MemoryPressureSnapshot"] | None = None


def _clamp_pressure(value: float) -> int:
    return max(0, min(100, int(value)))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _memory_snapshot_cache_ttl_s() -> float:
    """Return the live memory snapshot cache TTL.

    Darwin phys_footprint scans are accurate but expensive when called by many
    1Hz/10Hz background loops. In live runtime we cache for a short window so
    health/background policy probes do not stall the API event loop. Unit tests
    default to uncached unless they explicitly set the TTL.
    """

    default = 0.0 if os.environ.get("PYTEST_CURRENT_TEST") else 0.75
    return max(0.0, _env_float("AURA_MEMORY_SNAPSHOT_CACHE_TTL_S", default))


def clear_memory_pressure_snapshot_cache() -> None:
    """Testing hook and recovery hook for callers that need a fresh next sample."""

    global _SNAPSHOT_CACHE
    with _SNAPSHOT_CACHE_LOCK:
        _SNAPSHOT_CACHE = None


def _process_memory_bytes_from_process(process: psutil.Process) -> int:
    rss_bytes = int(getattr(process.memory_info(), "rss", 0) or 0)
    # The footprint lookup is an enhancement over RSS, never a gate: a
    # process object without a usable pid (test doubles, exited handles)
    # must degrade to plain RSS instead of discarding a valid sample.
    pid = int(getattr(process, "pid", 0) or 0)
    if pid <= 0:
        return rss_bytes
    return max(rss_bytes, _darwin_phys_footprint_bytes(pid))


def process_memory_bytes(pid: int | None = None) -> int:
    """Return the strongest available per-process memory pressure estimate."""

    process = psutil.Process(os.getpid() if pid is None else int(pid))
    return _process_memory_bytes_from_process(process)


def _process_tree_rss_gb() -> float:
    """Return memory footprint for Aura plus child MLX inference workers."""

    def _rss_bytes(process: psutil.Process) -> int:
        try:
            return _process_memory_bytes_from_process(process)
        except _MEMORY_MONITOR_RECOVERABLE_ERRORS:
            return 0

    process = psutil.Process(os.getpid())
    total_bytes = _rss_bytes(process)
    try:
        children = list(process.children(recursive=True))
    except _MEMORY_MONITOR_RECOVERABLE_ERRORS:
        children = []
    for child in children:
        total_bytes += _rss_bytes(child)
    return float(total_bytes) / _GIB


@dataclass(frozen=True)
class MemoryPressureSnapshot:
    pressure_pct: float
    available_gb: float
    total_gb: float
    process_rss_gb: float
    process_rss_limit_gb: float
    warning_pct: float
    high_pct: float
    critical_pct: float
    emergency_pct: float
    min_available_gb: float
    level: str
    reason: str
    observation_source: str
    observation_scenario_id: str
    host_observed: bool
    qualifies_as_live_pressure: bool
    observation_available: bool
    kernel_pressure_level: str = MEMORY_PRESSURE_UNKNOWN

    @property
    def warning(self) -> bool:
        return self.level in {"warning", "high", "critical", "emergency"}

    @property
    def high(self) -> bool:
        return self.level in {"high", "critical", "emergency"}

    @property
    def critical(self) -> bool:
        return self.level in {"critical", "emergency"}

    @property
    def emergency(self) -> bool:
        return self.level == "emergency"

    @property
    def should_gc(self) -> bool:
        return self.high

    @property
    def max_token_cap(self) -> int | None:
        if self.emergency:
            return 32
        if self.critical:
            return 64
        if self.high:
            return 192
        if self.warning:
            return 384
        return None

    @property
    def refuse_heavy_local_generation(self) -> bool:
        return (
            self.emergency
            or self.available_gb < self.min_available_gb
            or self.process_rss_gb >= self.process_rss_limit_gb
        )

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        payload = asdict(self)
        payload.update(
            {
                "warning": self.warning,
                "high": self.high,
                "critical": self.critical,
                "emergency": self.emergency,
                "should_gc": self.should_gc,
                "max_token_cap": self.max_token_cap,
                "refuse_heavy_local_generation": self.refuse_heavy_local_generation,
            }
        )
        return payload


# ── the kernel's own pressure verdict ────────────────────────────────────────
#
# LIVE 2026-08-17. A foreground turn was refused with
# "pressure=78.4% available=13.8GB (need <76.0% and >=18.0GB)" while the kernel
# reported kern.memorystatus_vm_pressure_level = 1 (NORMAL) and
# "System-wide memory free percentage: 79%". Both numbers were honestly
# computed and they measured different things.
#
# psutil's macOS `available` counts file-backed cache and compressed pages as
# consumed. They are not: the OS reclaims them on demand, which is what a cache
# is for. On a box deliberately holding a 20GB resident model plus a browser,
# that accounting reads as sustained pressure forever, so a gate keyed to it
# refuses ordinary turns in the system's normal operating state.
#
# macOS already publishes the signal it uses to tell processes to free memory.
# Ask it, instead of inferring a worse answer from page counts.


def kernel_memory_pressure_level() -> str:
    """What the OS itself says about memory pressure.

    Returns MEMORY_PRESSURE_UNKNOWN off Darwin or when the sysctl cannot be
    read, and callers must treat UNKNOWN as "no opinion" — it must never
    relax a limit on its own.
    """

    global _KERNEL_PRESSURE_CACHE
    now = time.monotonic()
    with _KERNEL_PRESSURE_LOCK:
        stamped_at, cached = _KERNEL_PRESSURE_CACHE
        if cached != MEMORY_PRESSURE_UNKNOWN and (now - stamped_at) <= _KERNEL_PRESSURE_TTL_S:
            return cached
    if sys.platform != "darwin":
        return MEMORY_PRESSURE_UNKNOWN
    try:
        import subprocess

        raw = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        level = _KERNEL_PRESSURE_LEVELS.get(
            int(str(raw.stdout).strip() or "0"), MEMORY_PRESSURE_UNKNOWN
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return MEMORY_PRESSURE_UNKNOWN
    with _KERNEL_PRESSURE_LOCK:
        _KERNEL_PRESSURE_CACHE = (now, level)
    return level


def get_memory_pressure_snapshot(
    *,
    force_refresh: bool = False,
    max_age_s: float | None = None,
    observer: ResourceObserver | None = None,
) -> MemoryPressureSnapshot:
    """Return one canonical unified-memory pressure decision for runtime gates."""

    global _SNAPSHOT_CACHE
    resource_observer = observer or get_resource_observer()
    provenance = resource_observer.provenance
    cache_key = (id(resource_observer), provenance.source.value, provenance.scenario_id)
    ttl = _memory_snapshot_cache_ttl_s() if max_age_s is None else max(0.0, float(max_age_s))
    now = time.monotonic()
    if not force_refresh and ttl > 0.0:
        with _SNAPSHOT_CACHE_LOCK:
            cached = _SNAPSHOT_CACHE
        if cached is not None and cached[1] == cache_key and (now - cached[0]) <= ttl:
            return cached[2]

    memory = resource_observer.memory()
    total_gb = float(memory.total_bytes) / float(1024**3)
    available_gb = float(memory.available_bytes) / float(1024**3)
    pressure_pct = float(memory.percent)
    if pressure_pct <= 0.0 and total_gb > 0.0:
        pressure_pct = max(0.0, min(100.0, (1.0 - (available_gb / total_gb)) * 100.0))

    if total_gb >= 60.0:
        warning_default = 78.0
        high_default = 84.0
        critical_default = 90.0
        emergency_default = 94.0
        min_available_default = 6.0
    else:
        warning_default = 72.0
        high_default = 80.0
        critical_default = 88.0
        emergency_default = 92.0
        min_available_default = 4.0

    warning_pct = _env_float("AURA_MEMORY_WARNING_PCT", warning_default)
    high_pct = _env_float("AURA_MEMORY_HIGH_PCT", high_default)
    critical_pct = _env_float("AURA_MEMORY_CRITICAL_PCT", critical_default)
    emergency_pct = _env_float("AURA_MEMORY_EMERGENCY_PCT", emergency_default)
    min_available_gb = _env_float("AURA_MEMORY_MIN_AVAILABLE_GB", min_available_default)
    if total_gb >= 60.0:
        process_rss_limit_default = min(38.0, max(30.0, total_gb * 0.56))
    elif total_gb > 0.0:
        process_rss_limit_default = min(24.0, max(10.0, total_gb * 0.70))
    else:
        process_rss_limit_default = 24.0
    process_rss_limit_gb = max(
        1.0,
        _env_float("AURA_PROCESS_RSS_LIMIT_GB", process_rss_limit_default),
    )
    process_rss_gb = float(memory.process_tree_rss_bytes) / _GIB
    if provenance.source in {ObservationSource.HOST, ObservationSource.LIVE_PRESSURE}:
        try:
            process_rss_gb = max(process_rss_gb, _process_tree_rss_gb())
        except _MEMORY_MONITOR_RECOVERABLE_ERRORS:
            pass

    system_level = "normal"
    if not memory.available:
        system_level = "emergency"
    elif pressure_pct >= emergency_pct or available_gb < max(1.0, min_available_gb / 2.0):
        system_level = "emergency"
    elif pressure_pct >= critical_pct or available_gb < min_available_gb:
        system_level = "critical"
    elif pressure_pct >= high_pct:
        system_level = "high"
    elif pressure_pct >= warning_pct:
        system_level = "warning"

    process_level = "normal"
    if process_rss_gb >= process_rss_limit_gb * 1.12:
        process_level = "emergency"
    elif process_rss_gb >= process_rss_limit_gb:
        process_level = "critical"
    elif process_rss_gb >= process_rss_limit_gb * 0.90:
        process_level = "high"
    elif process_rss_gb >= process_rss_limit_gb * 0.75:
        process_level = "warning"

    level_rank = {"normal": 0, "warning": 1, "high": 2, "critical": 3, "emergency": 4}
    level = max((system_level, process_level), key=lambda item: level_rank[item])

    reason_parts: list[str] = []
    if not memory.available:
        reason_parts.append(
            f"memory_observation_unavailable:{memory.error or 'unknown'}"
        )
    if system_level != "normal":
        reason_parts.append(
            f"memory_pressure:{pressure_pct:.1f}%/{available_gb:.1f}GB "
            f"(level={system_level}, critical>={critical_pct:.1f}%, emergency>={emergency_pct:.1f}%, "
            f"min_available={min_available_gb:.1f}GB)"
        )
    if process_level != "normal":
        reason_parts.append(
            f"process_tree_rss:{process_rss_gb:.1f}GB/{process_rss_limit_gb:.1f}GB "
            f"(level={process_level})"
        )
    reason = "; ".join(reason_parts)

    snapshot = MemoryPressureSnapshot(
        pressure_pct=pressure_pct,
        available_gb=available_gb,
        total_gb=total_gb,
        process_rss_gb=process_rss_gb,
        process_rss_limit_gb=process_rss_limit_gb,
        warning_pct=warning_pct,
        high_pct=high_pct,
        critical_pct=critical_pct,
        emergency_pct=emergency_pct,
        min_available_gb=min_available_gb,
        level=level,
        reason=reason,
        observation_source=provenance.source.value,
        kernel_pressure_level=kernel_memory_pressure_level(),
        observation_scenario_id=provenance.scenario_id,
        host_observed=provenance.host_observed,
        qualifies_as_live_pressure=provenance.qualifies_as_live_pressure,
        observation_available=bool(memory.available),
    )
    if ttl > 0.0:
        with _SNAPSHOT_CACHE_LOCK:
            _SNAPSHOT_CACHE = (now, cache_key, snapshot)
    return snapshot


class AppleSiliconMemoryMonitor:
    """Monitors Unified Memory pressure on Apple Silicon (M1/M2/M3/M4/M5).
    
    Aura uses this to throttle background reasoning (ReasoningQueue)
    when memory pressure is high to avoid system swap lag.
    """
    def __init__(
        self,
        interval: float = 2.0,
        threshold: int = 85,
        *,
        observer: ResourceObserver | None = None,
    ):
        self.interval = interval
        self.threshold = threshold
        self._observer = observer
        self.is_running = False
        self._pressure = 0
        self._loop_task = None

    async def start(self) -> None:
        self.is_running = True
        # Use our new task tracker helper (hoisted from Part 5)
        from .task_tracker import fire_and_track
        self._loop_task = fire_and_track(self._monitor_loop(), name="MemoryMonitor")
        logger.info("Apple Silicon Memory Monitor active.")

    async def stop(self) -> None:
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task

    @property
    def pressure(self) -> int:
        """Returns 0-100 indicating memory pressure."""
        return self._pressure

    async def _monitor_loop(self) -> None:
        import gc as _gc
        last_gc_at = 0.0
        last_purge_at = 0.0
        while self.is_running:
            try:
                # Sample memory pressure off the event loop so watchdogs never
                # see a shell command or psutil hiccup as a global stall.
                self._pressure = await asyncio.to_thread(self._get_pressure_sysctl)
                if self._pressure >= self.threshold:
                    logger.warning(
                        "⚠️ HIGH MEMORY PRESSURE: %s%% (Threshold: %s%%)",
                        self._pressure,
                        self.threshold,
                    )
                    import time as _time
                    now = _time.monotonic()
                    # Run a generational gc once per minute when pressure is up.
                    # Sustained-growth recovery had no eviction step between the
                    # 85% warning and the 90% VRAM purge, so RAM kept climbing
                    # through the gap.
                    if now - last_gc_at > 60.0:
                        await asyncio.to_thread(_gc.collect)
                        last_gc_at = now
                    # Trigger VRAM purge if critical (kept on its own cooldown
                    # so we never spin-purge the GPU heap).
                    if self._pressure > 90 and now - last_purge_at > 30.0:
                        from core.managers.vram_manager import get_vram_manager
                        await asyncio.to_thread(get_vram_manager().purge)
                        last_purge_at = now

                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except _MEMORY_MONITOR_RECOVERABLE_ERRORS as e:
                record_degradation('memory_monitor', e)
                logger.error("Memory monitor error: %s", e)
                await asyncio.sleep(5)

    def _get_pressure_sysctl(self) -> int:
        """Return a safe, attributable system memory pressure sample."""
        try:
            mem = (self._observer or get_resource_observer()).memory()
            if not mem.available:
                raise RuntimeError(mem.error or "memory observation unavailable")
            if mem.percent > 0.0:
                return _clamp_pressure(float(mem.percent))

            total = int(mem.total_bytes)
            available = int(mem.available_bytes)
            if total > 0:
                return _clamp_pressure((1.0 - (available / total)) * 100.0)
            return 0
        except _MEMORY_MONITOR_RECOVERABLE_ERRORS as exc:
            record_degradation("memory_monitor", exc)
            logger.debug("Memory pressure sample failed: %s", exc)
            return 100
