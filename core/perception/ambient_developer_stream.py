"""Bounded ambient developer-environment sensory stream.

This service gives Aura a continuous, low-cost stream of local runtime context:
repository changes, active watched directories, and recent log warnings. It is
not a prompt wrapper. It publishes compact frames into WorldState and the
TimescaleBridge so foreground cognition can be grounded in verified background
evidence without inventing idle-time events.
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime import resource_psutil as psutil
from core.runtime.background_policy import (
    background_loop_start_reason,
    constitutive_compute_budget_async,
)
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.AmbientDeveloperStream")

_RUNTIME_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    ImportError,
    TimeoutError,
    asyncio.TimeoutError,
)
_DEFAULT_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "htmlcov",
    "llm_data",
    "models",
    "node_modules",
    "venv",
}
_CODE_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
_LOG_PATTERN = re.compile(
    r"\b(ERROR|CRITICAL|Traceback|Exception|DEGRADATION|memory_pressure|OOM|unhealthy|blocked)\b",
    re.IGNORECASE,
)


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value))


def _bounded_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\x00", "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() or text[:limit]


def _project_root() -> Path:
    try:
        from core.config import config

        return config.paths.project_root.resolve()
    except _RUNTIME_ERRORS:
        return Path(__file__).resolve().parents[2]


def _log_roots(project_root: Path) -> tuple[Path, ...]:
    roots = [project_root / "logs"]
    try:
        from core.config import config

        roots.append(config.paths.log_dir)
    except _RUNTIME_ERRORS:
        pass
    return tuple(dict.fromkeys(path.resolve() for path in roots))


def _default_watch_roots(project_root: Path) -> tuple[Path, ...]:
    configured = os.environ.get("AURA_AMBIENT_WATCH_DIRS", "").strip()
    if configured:
        raw = [part.strip() for part in configured.split(os.pathsep) if part.strip()]
        return tuple((Path(part).expanduser() if Path(part).is_absolute() else project_root / part).resolve() for part in raw)
    names = ("core", "interface", "tools", "training", "tests", "config", "docs")
    return tuple((project_root / name).resolve() for name in names if (project_root / name).exists())


@dataclass(frozen=True)
class AmbientFileEvent:
    path: str
    kind: str
    mtime: float
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AmbientLogEvent:
    path: str
    line: str
    file_mtime: float = 0.0
    observed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AmbientTerminalEvent:
    path: str
    line: str
    file_mtime: float = 0.0
    observed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AmbientNetworkEvent:
    kind: str
    count: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AmbientResourceInterrupt:
    kind: str
    severity: str
    value: float
    threshold: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _TextReadCursor:
    device: int
    inode: int
    offset: int
    mtime_ns: int
    remainder: bytes = b""
    last_seen: float = field(default_factory=time.time)


@dataclass(frozen=True)
class AmbientDeveloperFrame:
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    repo_root: str = ""
    git_dirty_count: int = 0
    git_status: tuple[str, ...] = ()
    recent_files: tuple[AmbientFileEvent, ...] = ()
    log_events: tuple[AmbientLogEvent, ...] = ()
    terminal_events: tuple[AmbientTerminalEvent, ...] = ()
    network_events: tuple[AmbientNetworkEvent, ...] = ()
    resource_interrupts: tuple[AmbientResourceInterrupt, ...] = ()
    repair_candidates: tuple[str, ...] = ()
    throttled_reason: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["git_status"] = list(self.git_status)
        data["recent_files"] = [event.to_dict() for event in self.recent_files]
        data["log_events"] = [event.to_dict() for event in self.log_events]
        data["terminal_events"] = [event.to_dict() for event in self.terminal_events]
        data["network_events"] = [event.to_dict() for event in self.network_events]
        data["resource_interrupts"] = [event.to_dict() for event in self.resource_interrupts]
        data["repair_candidates"] = list(self.repair_candidates)
        return data

    @property
    def event_count(self) -> int:
        return (
            self.git_dirty_count
            + len(self.recent_files)
            + len(self.log_events)
            + len(self.terminal_events)
            + len(self.network_events)
            + len(self.resource_interrupts)
        )


class AmbientDeveloperStream:
    """Continuously sample local developer/runtime context with hard bounds."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        watch_roots: tuple[Path, ...] | None = None,
        log_roots: tuple[Path, ...] | None = None,
        terminal_roots: tuple[Path, ...] | None = None,
        sample_interval_s: float | None = None,
        max_scan_files: int | None = None,
        recent_window_s: float | None = None,
    ) -> None:
        self.project_root = (project_root or _project_root()).resolve()
        self.watch_roots = (
            watch_roots
            if watch_roots is not None
            else _default_watch_roots(self.project_root)
        )
        self.log_roots = (
            log_roots if log_roots is not None else _log_roots(self.project_root)
        )
        self.terminal_roots = (
            terminal_roots
            if terminal_roots is not None
            else self._default_terminal_roots()
        )
        self.sample_interval_s = (
            sample_interval_s
            if sample_interval_s is not None
            else _env_float("AURA_AMBIENT_STREAM_INTERVAL_S", 30.0, minimum=5.0, maximum=900.0)
        )
        self.max_scan_files = max_scan_files or _env_int(
            "AURA_AMBIENT_STREAM_MAX_SCAN_FILES",
            3500,
            minimum=100,
            maximum=20000,
        )
        self.recent_window_s = (
            recent_window_s
            if recent_window_s is not None
            else _env_float("AURA_AMBIENT_STREAM_RECENT_WINDOW_S", 180.0, minimum=15.0, maximum=3600.0)
        )
        self.running = False
        self._task: asyncio.Task | None = None
        self._frame_id = 0
        self._errors = 0
        self._started_at = 0.0
        self._latest_frame: AmbientDeveloperFrame | None = None
        self._frames: deque[AmbientDeveloperFrame] = deque(maxlen=120)
        self._source_state_lock = threading.RLock()
        self._recent_file_fingerprints: dict[str, tuple[int, int, int, int]] = {}
        self._text_read_cursors: dict[tuple[str, str], _TextReadCursor] = {}
        self._text_bytes_dropped = 0

    async def start(self) -> None:
        if self.running:
            return
        reason = background_loop_start_reason("ambient_developer_stream")
        if reason:
            ServiceContainer.register_instance("ambient_developer_stream", self, required=False)
            logger.info("AmbientDeveloperStream not started: %s", reason)
            return
        self.running = True
        self._started_at = time.time()
        ServiceContainer.register_instance("ambient_developer_stream", self, required=False)
        self._task = create_tracked_task(
            self._run_loop(),
            name="Aura.AmbientDeveloperStream",
        )
        logger.info("AmbientDeveloperStream ONLINE — %ss interval", self.sample_interval_s)

    async def stop(self) -> None:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def latest_frame(self) -> AmbientDeveloperFrame | None:
        return self._latest_frame

    async def _run_loop(self) -> None:
        while self.running:
            try:
                budget = await constitutive_compute_budget_async(
                    "ambient_developer_stream",
                    base_hz=0.1,
                    foreground_hz=0.1,
                    memory_high_hz=0.1,
                    memory_critical_hz=0.1,
                )
                await self.sample_once(
                    throttled_reason=budget.reason if self._budget_is_throttled(budget) else ""
                )
                await asyncio.sleep(max(self.sample_interval_s, budget.interval_s))
            except asyncio.CancelledError:
                raise
            except _RUNTIME_ERRORS as exc:
                self._errors += 1
                record_degradation("ambient_developer_stream", exc)
                logger.debug("AmbientDeveloperStream tick failed: %s", exc)
                await asyncio.sleep(self.sample_interval_s)

    @staticmethod
    def _budget_is_throttled(budget: Any) -> bool:
        reason = str(getattr(budget, "reason", "") or "")
        return reason not in {"", "nominal", "component_override", "global_override"}

    async def sample_once(self, *, throttled_reason: str = "") -> AmbientDeveloperFrame:
        frame = await asyncio.to_thread(self._collect_frame, throttled_reason=throttled_reason)
        self._publish_frame(frame)
        return frame

    def _collect_frame(self, *, throttled_reason: str = "") -> AmbientDeveloperFrame:
        self._frame_id += 1
        if throttled_reason.startswith("memory") or throttled_reason == "foreground_generation_active":
            frame = AmbientDeveloperFrame(
                frame_id=self._frame_id,
                repo_root=str(self.project_root),
                throttled_reason=throttled_reason,
                summary=f"ambient stream throttled: {throttled_reason}",
            )
            return frame

        git_status = self._collect_git_status()
        recent_files = self._collect_recent_files()
        log_events = self._collect_log_events()
        terminal_events = self._collect_terminal_events()
        network_events = self._collect_network_events()
        resource_interrupts = self._collect_resource_interrupts()
        repair_candidates = self._build_repair_candidates(
            git_status,
            recent_files,
            log_events,
            terminal_events,
            network_events,
            resource_interrupts,
        )
        summary = self._summarize(
            git_status,
            recent_files,
            log_events,
            terminal_events,
            network_events,
            resource_interrupts,
            repair_candidates,
        )
        return AmbientDeveloperFrame(
            frame_id=self._frame_id,
            repo_root=str(self.project_root),
            git_dirty_count=len(git_status),
            git_status=tuple(git_status),
            recent_files=tuple(recent_files),
            log_events=tuple(log_events),
            terminal_events=tuple(terminal_events),
            network_events=tuple(network_events),
            resource_interrupts=tuple(resource_interrupts),
            repair_candidates=tuple(repair_candidates),
            summary=summary,
        )

    def _default_terminal_roots(self) -> tuple[Path, ...]:
        configured = os.environ.get("AURA_AMBIENT_TERMINAL_DIRS", "").strip()
        if configured:
            raw = [part.strip() for part in configured.split(os.pathsep) if part.strip()]
            return tuple((Path(part).expanduser() if Path(part).is_absolute() else self.project_root / part).resolve() for part in raw)
        roots = [
            state_root() / "data" / "terminal",
            self.project_root / "logs" / "terminal",
        ]
        return tuple(path.resolve() for path in roots)

    def _collect_git_status(self) -> list[str]:
        try:
            from core.runtime.subprocess_gateway import get_subprocess_gateway

            result = get_subprocess_gateway().run(
                ["git", "status", "--porcelain=v1", "-uno"],
                cwd=str(self.project_root),
                capture_output=True,
                timeout=2.0,
                read_only=True,
                source="ambient_developer_stream.git_status",
                accelerator_capability="none",
            )
            if result.returncode != 0:
                return []
            lines = [
                _bounded_text(line, 180)
                for line in str(result.stdout or "").splitlines()
                if line.strip()
            ]
            return lines[:40]
        except _RUNTIME_ERRORS as exc:
            record_degradation("ambient_developer_stream.git_status", exc)
            return []

    def _collect_recent_files(self) -> list[AmbientFileEvent]:
        cutoff = time.time() - self.recent_window_s
        events: list[AmbientFileEvent] = []
        scanned = 0
        stack = [root for root in self.watch_roots if root.exists()]
        while stack and scanned < self.max_scan_files:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if scanned >= self.max_scan_files:
                            break
                        name = entry.name
                        if name in _DEFAULT_SKIP_DIRS or name.startswith(".#"):
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            scanned += 1
                            path = Path(entry.path)
                            if path.suffix.lower() not in _CODE_SUFFIXES:
                                continue
                            stat = entry.stat(follow_symlinks=False)
                            if stat.st_mtime < cutoff:
                                continue
                            source_key = str(path.resolve())
                            fingerprint = (
                                int(stat.st_dev),
                                int(stat.st_ino),
                                int(stat.st_mtime_ns),
                                int(stat.st_size),
                            )
                            with self._source_state_lock:
                                if self._recent_file_fingerprints.get(source_key) == fingerprint:
                                    continue
                                self._recent_file_fingerprints[source_key] = fingerprint
                            events.append(
                                AmbientFileEvent(
                                    path=self._relative(path),
                                    kind="modified",
                                    mtime=round(float(stat.st_mtime), 3),
                                    size=int(stat.st_size),
                                )
                            )
                        except _RUNTIME_ERRORS:
                            continue
            except _RUNTIME_ERRORS:
                continue
        events.sort(key=lambda event: event.mtime, reverse=True)
        return events[:25]

    def _recent_text_candidates(
        self,
        roots: tuple[Path, ...],
        *,
        stream: str,
        suffixes: tuple[str, ...],
        limit: int = 4,
    ) -> list[tuple[Path, os.stat_result]]:
        cutoff = time.time() - self.recent_window_s
        candidates: dict[str, tuple[Path, os.stat_result]] = {}
        for root in roots:
            if not root.exists():
                continue
            for suffix in suffixes:
                try:
                    paths = root.glob(f"*{suffix}")
                    for path in paths:
                        try:
                            stat = path.stat()
                        except _RUNTIME_ERRORS:
                            continue
                        if not path.is_file() or stat.st_size <= 0 or stat.st_mtime < cutoff:
                            continue
                        if not self._text_source_needs_read(path, stat, stream=stream):
                            continue
                        candidates[str(path.resolve())] = (path, stat)
                except _RUNTIME_ERRORS:
                    continue
        return sorted(
            candidates.values(),
            key=lambda item: (float(item[1].st_mtime), str(item[0])),
            reverse=True,
        )[:limit]

    def _text_source_needs_read(
        self,
        path: Path,
        stat: os.stat_result,
        *,
        stream: str,
    ) -> bool:
        cursor_key = (stream, str(path.resolve()))
        identity = (int(stat.st_dev), int(stat.st_ino))
        with self._source_state_lock:
            cursor = self._text_read_cursors.get(cursor_key)
            if cursor is None or (cursor.device, cursor.inode) != identity:
                return True
            return (
                int(stat.st_size) != cursor.offset
                or int(stat.st_mtime_ns) != cursor.mtime_ns
            )

    def _read_incremental_lines(
        self,
        path: Path,
        stat: os.stat_result,
        *,
        stream: str,
        byte_budget: int,
    ) -> tuple[list[str], float, float]:
        """Read each source byte at most once while handling rotation/truncation."""

        observed_at = time.time()
        resolved = str(path.resolve())
        cursor_key = (stream, resolved)
        identity = (int(stat.st_dev), int(stat.st_ino))
        with self._source_state_lock:
            cursor = self._text_read_cursors.get(cursor_key)
            reset = (
                cursor is None
                or (cursor.device, cursor.inode) != identity
                or int(stat.st_size) < cursor.offset
                or (
                    int(stat.st_size) == cursor.offset
                    and int(stat.st_mtime_ns) != cursor.mtime_ns
                )
            )
            if (
                not reset
                and int(stat.st_size) == cursor.offset
                and int(stat.st_mtime_ns) == cursor.mtime_ns
            ):
                cursor.last_seen = observed_at
                return [], float(stat.st_mtime), observed_at

            remainder = b"" if reset else cursor.remainder
            start = max(0, int(stat.st_size) - byte_budget) if reset else cursor.offset
            unread = max(0, int(stat.st_size) - start)
            if unread > byte_budget:
                dropped = unread - byte_budget
                self._text_bytes_dropped += dropped
                start = int(stat.st_size) - byte_budget
                remainder = b""

            with path.open("rb") as handle:
                handle.seek(start)
                chunk = handle.read(byte_budget)
                final_stat = os.fstat(handle.fileno())
            next_offset = start + len(chunk)

            if reset and start > 0:
                newline = chunk.find(b"\n")
                chunk = chunk[newline + 1 :] if newline >= 0 else b""
            combined = (b"" if reset else remainder) + chunk
            newline = combined.rfind(b"\n")
            if newline < 0:
                self._text_bytes_dropped += max(0, len(combined) - 4096)
                complete = b""
                next_remainder = combined[-4096:]
            else:
                complete = combined[: newline + 1]
                trailing = combined[newline + 1 :]
                self._text_bytes_dropped += max(0, len(trailing) - 4096)
                next_remainder = trailing[-4096:]

            self._text_read_cursors[cursor_key] = _TextReadCursor(
                device=int(final_stat.st_dev),
                inode=int(final_stat.st_ino),
                offset=next_offset,
                mtime_ns=int(final_stat.st_mtime_ns),
                remainder=next_remainder,
                last_seen=observed_at,
            )
            if len(self._text_read_cursors) > 256:
                oldest = sorted(
                    self._text_read_cursors,
                    key=lambda key: self._text_read_cursors[key].last_seen,
                )[: len(self._text_read_cursors) - 256]
                for key in oldest:
                    self._text_read_cursors.pop(key, None)

        lines = complete.decode("utf-8", errors="replace").splitlines()
        return lines, float(final_stat.st_mtime), observed_at

    def _collect_log_events(self) -> list[AmbientLogEvent]:
        events: list[AmbientLogEvent] = []
        for path, stat in self._recent_text_candidates(
            self.log_roots,
            stream="log",
            suffixes=(".log",),
        ):
            try:
                lines, file_mtime, observed_at = self._read_incremental_lines(
                    path,
                    stat,
                    stream="log",
                    byte_budget=16000,
                )
                for line in lines[-120:]:
                    if _LOG_PATTERN.search(line):
                        events.append(
                            AmbientLogEvent(
                                path=self._relative(path),
                                line=_bounded_text(line, 260),
                                file_mtime=round(file_mtime, 3),
                                observed_at=round(observed_at, 3),
                            )
                        )
                        if len(events) >= 12:
                            return events
            except _RUNTIME_ERRORS:
                continue
        return events

    def _collect_terminal_events(self) -> list[AmbientTerminalEvent]:
        events: list[AmbientTerminalEvent] = []
        for path, stat in self._recent_text_candidates(
            self.terminal_roots,
            stream="terminal",
            suffixes=(".log", ".txt"),
        ):
            try:
                lines, file_mtime, observed_at = self._read_incremental_lines(
                    path,
                    stat,
                    stream="terminal",
                    byte_budget=12000,
                )
                for line in lines[-80:]:
                    if _LOG_PATTERN.search(line) or "Traceback" in line:
                        events.append(
                            AmbientTerminalEvent(
                                path=self._relative(path),
                                line=_bounded_text(line, 260),
                                file_mtime=round(file_mtime, 3),
                                observed_at=round(observed_at, 3),
                            )
                        )
                        if len(events) >= 10:
                            return events
            except _RUNTIME_ERRORS:
                continue
        return events

    def _collect_network_events(self) -> list[AmbientNetworkEvent]:
        try:
            conns = psutil.net_connections(kind="inet")
        except psutil.AccessDenied:
            # macOS can deny process-wide socket enumeration even when Aura's
            # own network access is healthy. That is a sensor capability
            # boundary, not a degraded runtime. Preserve the fact in the
            # perceptual stream without poisoning health or nociception.
            return [
                AmbientNetworkEvent(
                    kind="socket_visibility_unavailable",
                    count=0,
                    detail="host denied process-wide socket enumeration",
                )
            ]
        except (psutil.NoSuchProcess, OSError, RuntimeError) as exc:
            record_degradation("ambient_developer_stream.network", exc)
            return []
        by_status: dict[str, int] = {}
        local_listeners = 0
        external_established = 0
        for conn in conns[:1000]:
            status = str(getattr(conn, "status", "") or "UNKNOWN")
            by_status[status] = by_status.get(status, 0) + 1
            laddr = getattr(conn, "laddr", None)
            raddr = getattr(conn, "raddr", None)
            if status == "LISTEN" and laddr:
                local_listeners += 1
            if status == "ESTABLISHED" and raddr:
                external_established += 1
        events: list[AmbientNetworkEvent] = []
        if local_listeners:
            events.append(AmbientNetworkEvent(kind="listening_sockets", count=local_listeners))
        if external_established:
            events.append(AmbientNetworkEvent(kind="established_connections", count=external_established))
        if len(conns) >= 1000:
            events.append(AmbientNetworkEvent(kind="socket_scan_truncated", count=len(conns), detail="first_1000_connections_sampled"))
        return events[:6]

    def _collect_resource_interrupts(self) -> list[AmbientResourceInterrupt]:
        interrupts: list[AmbientResourceInterrupt] = []
        try:
            vm = psutil.virtual_memory()
            if float(vm.percent) >= 85.0:
                interrupts.append(
                    AmbientResourceInterrupt(
                        kind="memory_pressure",
                        severity="critical" if float(vm.percent) >= 92.0 else "warning",
                        value=round(float(vm.percent), 2),
                        threshold=85.0,
                        detail=f"available_gb={round(float(vm.available) / 1024**3, 2)}",
                    )
                )
            cpu = float(psutil.cpu_percent(interval=None))
            if cpu >= 90.0:
                interrupts.append(
                    AmbientResourceInterrupt(
                        kind="cpu_pressure",
                        severity="warning",
                        value=round(cpu, 2),
                        threshold=90.0,
                    )
                )
            battery = psutil.sensors_battery()
            if battery and battery.percent <= 15.0 and not battery.power_plugged:
                interrupts.append(
                    AmbientResourceInterrupt(
                        kind="battery_low",
                        severity="warning",
                        value=round(float(battery.percent), 2),
                        threshold=15.0,
                    )
                )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation("ambient_developer_stream.resources", exc)

        # psutil does not expose macOS thermal pressure; surface the platform so
        # callers know whether this was directly measured or unavailable.
        if platform.system() == "Darwin":
            try:
                from core.runtime.pressure import get_pressure_snapshot

                snapshot = get_pressure_snapshot()
                thermal = str(getattr(snapshot, "thermal_pressure", "") or "")
                if thermal and thermal.lower() not in {"nominal", "0", "none"}:
                    interrupts.append(
                        AmbientResourceInterrupt(
                            kind="thermal_pressure",
                            severity="warning",
                            value=1.0,
                            threshold=0.0,
                            detail=thermal,
                        )
                    )
            except _RUNTIME_ERRORS:
                pass
        return interrupts[:8]

    def _build_repair_candidates(
        self,
        git_status: list[str],
        recent_files: list[AmbientFileEvent],
        log_events: list[AmbientLogEvent],
        terminal_events: list[AmbientTerminalEvent],
        network_events: list[AmbientNetworkEvent],
        resource_interrupts: list[AmbientResourceInterrupt],
    ) -> list[str]:
        candidates: list[str] = []
        if log_events:
            candidates.append("review_recent_log_errors")
        if terminal_events:
            candidates.append("review_recent_terminal_errors")
        if any("DEGRADATION" in event.line.upper() for event in log_events):
            candidates.append("triage_degradation_events")
        if any("MEMORY_PRESSURE" in event.line.upper() or "OOM" in event.line.upper() for event in log_events):
            candidates.append("check_memory_pressure_guard")
        if any(event.kind == "memory_pressure" for event in resource_interrupts):
            candidates.append("reduce_background_compute_until_memory_recovers")
        if any(event.kind == "thermal_pressure" for event in resource_interrupts):
            candidates.append("throttle_nonessential_background_work")
        if any(event.kind == "established_connections" and event.count > 25 for event in network_events):
            candidates.append("audit_network_activity")
        if git_status or recent_files:
            candidates.append("run_targeted_tests_for_recent_changes")
        return candidates[:6]

    def _summarize(
        self,
        git_status: list[str],
        recent_files: list[AmbientFileEvent],
        log_events: list[AmbientLogEvent],
        terminal_events: list[AmbientTerminalEvent],
        network_events: list[AmbientNetworkEvent],
        resource_interrupts: list[AmbientResourceInterrupt],
        repair_candidates: list[str],
    ) -> str:
        parts = []
        if git_status:
            parts.append(f"{len(git_status)} tracked repo change(s)")
        if recent_files:
            parts.append(f"{len(recent_files)} recent watched file event(s)")
        if log_events:
            parts.append(f"{len(log_events)} recent warning/error log line(s)")
        if terminal_events:
            parts.append(f"{len(terminal_events)} terminal warning/error line(s)")
        if network_events:
            parts.append(f"{len(network_events)} network telemetry signal(s)")
        if resource_interrupts:
            parts.append(f"{len(resource_interrupts)} resource interrupt(s)")
        if repair_candidates:
            parts.append("repair candidates: " + ", ".join(repair_candidates[:3]))
        return "; ".join(parts) if parts else "ambient developer stream observed no material changes"

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except _RUNTIME_ERRORS:
            return str(path)

    def _publish_frame(self, frame: AmbientDeveloperFrame) -> None:
        self._latest_frame = frame
        self._frames.append(frame)
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws is not None and hasattr(ws, "record_event") and frame.event_count:
                ws.record_event(
                    f"Ambient developer stream: {frame.summary}",
                    source="ambient_developer_stream",
                    salience=0.45 if frame.log_events else 0.25,
                    ttl=300,
                )
        except _RUNTIME_ERRORS as exc:
            record_degradation("ambient_developer_stream.world_state", exc)
        try:
            from core.runtime.timescale_bridge import get_timescale_bridge

            get_timescale_bridge().ingest_ambient_developer_frame(frame)
        except _RUNTIME_ERRORS as exc:
            record_degradation("ambient_developer_stream.timescale_bridge", exc)

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "schema": "aura.ambient_developer_stream.status.v1",
            "sample_interval_s": self.sample_interval_s,
            "frames": len(self._frames),
            "errors": self._errors,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0.0,
            "latest_frame": self._latest_frame.to_dict() if self._latest_frame else None,
            "watch_roots": [str(path) for path in self.watch_roots],
            "terminal_roots": [str(path) for path in self.terminal_roots],
            "recent_window_s": self.recent_window_s,
            "text_cursor_count": len(self._text_read_cursors),
            "text_bytes_dropped": self._text_bytes_dropped,
        }

    status = get_status


_AMBIENT_STREAM: AmbientDeveloperStream | None = None


def get_ambient_developer_stream() -> AmbientDeveloperStream:
    global _AMBIENT_STREAM
    existing = ServiceContainer.get("ambient_developer_stream", default=None)
    if isinstance(existing, AmbientDeveloperStream):
        _AMBIENT_STREAM = existing
        return existing
    if _AMBIENT_STREAM is None:
        _AMBIENT_STREAM = AmbientDeveloperStream()
    ServiceContainer.register_instance("ambient_developer_stream", _AMBIENT_STREAM, required=False)
    return _AMBIENT_STREAM


def render_ambient_developer_prompt_block(frame: dict[str, Any] | AmbientDeveloperFrame | None) -> str:
    if frame is None:
        return ""
    data = frame.to_dict() if isinstance(frame, AmbientDeveloperFrame) else dict(frame or {})
    summary = str(data.get("summary") or "").strip()
    if not summary:
        return ""
    lines = ["## AMBIENT DEVELOPER STREAM"]
    lines.append(f"- Summary: {summary}")
    candidates = data.get("repair_candidates") if isinstance(data.get("repair_candidates"), list) else []
    if candidates:
        lines.append("- Repair candidates: " + ", ".join(str(item) for item in candidates[:4]))
    resources = data.get("resource_interrupts") if isinstance(data.get("resource_interrupts"), list) else []
    if resources:
        lines.append("- Resource interrupts: " + ", ".join(str(item.get("kind")) for item in resources[:4] if isinstance(item, dict)))
    lines.append("Use as verified background evidence; do not invent file/log events beyond this frame.")
    return "\n".join(lines)


__all__ = [
    "AmbientDeveloperFrame",
    "AmbientDeveloperStream",
    "AmbientFileEvent",
    "AmbientLogEvent",
    "AmbientNetworkEvent",
    "AmbientResourceInterrupt",
    "AmbientTerminalEvent",
    "get_ambient_developer_stream",
    "render_ambient_developer_prompt_block",
]
