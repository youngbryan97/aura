"""Typed, attributable host-resource observations.

Resource policy must not reach into ``psutil`` (or platform APIs) at the point
where it makes a decision.  Doing so made otherwise deterministic tests depend
on whichever models, processes, disks, and thermal load happened to exist on
the developer host.  This module is the adapter boundary between host facts and
policy:

* production uses :class:`HostResourceObserver`;
* tests install :class:`SimulatedResourceObserver` with an explicit scenario;
* bounded live-pressure proofs use the distinct ``live_pressure`` source.

Every observation carries provenance.  A simulated observation therefore
cannot be mistaken for live evidence merely because its numbers look real.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import psutil


class ObservationSource(StrEnum):
    """Where a resource fact came from."""

    HOST = "host"
    SIMULATED = "simulated"
    LIVE_PRESSURE = "live_pressure"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ObservationProvenance:
    source: ObservationSource
    scenario_id: str
    captured_at: float = field(default_factory=time.time)
    observer: str = ""

    @property
    def host_observed(self) -> bool:
        return self.source in {
            ObservationSource.HOST,
            ObservationSource.LIVE_PRESSURE,
        }

    @property
    def qualifies_as_live_pressure(self) -> bool:
        return self.source is ObservationSource.LIVE_PRESSURE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload["host_observed"] = self.host_observed
        payload["qualifies_as_live_pressure"] = self.qualifies_as_live_pressure
        return payload


@dataclass(frozen=True)
class MemoryObservation:
    provenance: ObservationProvenance
    total_bytes: int
    available_bytes: int
    used_bytes: int
    free_bytes: int
    active_bytes: int
    percent: float
    process_rss_bytes: int
    process_tree_rss_bytes: int
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    swap_free_bytes: int = 0
    swap_percent: float = 0.0
    available: bool = True
    error: str = ""

    # psutil-compatible read-only aliases ease migration of legacy telemetry.
    @property
    def total(self) -> int:
        return self.total_bytes

    @property
    def used(self) -> int:
        return self.used_bytes

    @property
    def free(self) -> int:
        return self.free_bytes

    @property
    def active(self) -> int:
        return self.active_bytes

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class DiskObservation:
    provenance: ObservationProvenance
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent: float
    available: bool = True
    error: str = ""

    @property
    def total(self) -> int:
        return self.total_bytes

    @property
    def used(self) -> int:
        return self.used_bytes

    @property
    def free(self) -> int:
        return self.free_bytes

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class ThermalObservation:
    provenance: ObservationProvenance
    level: int
    provider: str
    detail: str = ""
    available: bool = True

    @property
    def blind(self) -> bool:
        return not self.available or self.provider == "blind"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        payload["blind"] = self.blind
        return payload


@dataclass(frozen=True)
class AcceleratorObservation:
    provenance: ObservationProvenance
    provider: str
    active_bytes: int = 0
    cache_bytes: int = 0
    peak_bytes: int = 0
    available: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class ProcessObservation:
    provenance: ObservationProvenance
    pid: int
    ppid: int
    create_time: float
    status: str
    name: str
    cmdline: tuple[str, ...]
    rss_bytes: int
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    cpu_user_seconds: float = 0.0
    cpu_system_seconds: float = 0.0
    num_threads: int = 0
    num_fds: int = 0
    ancestor_pids: tuple[int, ...] = ()
    exe: str = ""
    username: str = ""
    cwd: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        payload["cmdline"] = list(self.cmdline)
        payload["ancestor_pids"] = list(self.ancestor_pids)
        return payload


@dataclass(frozen=True)
class ProcessTableObservation:
    provenance: ObservationProvenance
    processes: tuple[ProcessObservation, ...]
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "available": self.available,
            "error": self.error,
            "processes": [process.to_dict() for process in self.processes],
        }


@dataclass(frozen=True)
class ProcessIdsObservation:
    """Lightweight process identity census without metadata enrichment."""

    provenance: ObservationProvenance
    pids: tuple[int, ...]
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "available": self.available,
            "error": self.error,
            "pids": list(self.pids),
        }


@dataclass(frozen=True)
class NetworkConnectionObservation:
    provenance: ObservationProvenance
    pid: int
    fd: int
    family: str
    socket_type: str
    local_host: str
    local_port: int
    remote_host: str = ""
    remote_port: int = 0
    status: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class ConnectionTableObservation:
    provenance: ObservationProvenance
    connections: tuple[NetworkConnectionObservation, ...]
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "available": self.available,
            "error": self.error,
            "connections": [connection.to_dict() for connection in self.connections],
        }


@dataclass(frozen=True)
class OpenFileIdentityObservation:
    """Kernel-derived identity for one process-owned file descriptor."""

    path: str
    fd: int
    device: int
    inode: int
    byte_length: int
    mtime_ns: int
    mode: int
    provider: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenFilesObservation:
    provenance: ObservationProvenance
    pid: int
    paths: tuple[str, ...]
    identities: tuple[OpenFileIdentityObservation, ...] = ()
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "pid": self.pid,
            "paths": list(self.paths),
            "identities": [identity.to_dict() for identity in self.identities],
            "available": self.available,
            "error": self.error,
        }


@dataclass(frozen=True)
class PowerObservation:
    provenance: ObservationProvenance
    battery_percent: float
    plugged: bool
    seconds_left: int = -2
    provider: str = "psutil"
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class ComputeObservation:
    provenance: ObservationProvenance
    cpu_percent: float
    cpu_count: int
    load_1m: float
    load_5m: float = 0.0
    load_15m: float = 0.0
    boot_time: float = 0.0
    cpu_user_seconds: float = 0.0
    cpu_system_seconds: float = 0.0
    cpu_idle_seconds: float = 0.0
    available: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance.to_dict()
        return payload


@dataclass(frozen=True)
class ResourceObservation:
    provenance: ObservationProvenance
    memory: MemoryObservation
    disk: DiskObservation
    thermal: ThermalObservation
    accelerator: AcceleratorObservation
    compute: ComputeObservation
    power: PowerObservation
    processes: tuple[ProcessObservation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "memory": self.memory.to_dict(),
            "disk": self.disk.to_dict(),
            "thermal": self.thermal.to_dict(),
            "accelerator": self.accelerator.to_dict(),
            "compute": self.compute.to_dict(),
            "power": self.power.to_dict(),
            "processes": [process.to_dict() for process in self.processes],
        }


@runtime_checkable
class ResourceObserver(Protocol):
    @property
    def provenance(self) -> ObservationProvenance: ...

    def memory(
        self,
        *,
        root_pid: int | None = None,
        include_process_tree: bool = True,
    ) -> MemoryObservation: ...

    def disk(self, path: str | os.PathLike[str] = "/") -> DiskObservation: ...

    def thermal(self, *, max_age_s: float = 5.0) -> ThermalObservation: ...

    def accelerator(self) -> AcceleratorObservation: ...

    def compute(self) -> ComputeObservation: ...

    def power(self) -> PowerObservation: ...

    def process_ids(self) -> ProcessIdsObservation: ...

    def process_table(self) -> ProcessTableObservation: ...

    def processes(self) -> tuple[ProcessObservation, ...]: ...

    def process(self, pid: int) -> ProcessObservation | None: ...

    def process_tree(
        self,
        root_pid: int,
        *,
        recursive: bool = True,
    ) -> ProcessTableObservation: ...

    def connection_table(
        self,
        *,
        kind: str = "inet",
        pid: int | None = None,
    ) -> ConnectionTableObservation: ...

    def connections(
        self,
        *,
        kind: str = "inet",
        pid: int | None = None,
    ) -> tuple[NetworkConnectionObservation, ...]: ...

    def open_file_table(self, *, pid: int | None = None) -> OpenFilesObservation: ...

    def open_files(self, *, pid: int | None = None) -> tuple[str, ...]: ...

    def snapshot(
        self,
        *,
        path: str | os.PathLike[str] = "/",
        include_processes: bool = False,
    ) -> ResourceObservation: ...


_DARWIN_MAXPATHLEN = 1024
_DARWIN_PROC_PIDFDVNODEPATHINFO = 2


class _DarwinProcFileInfo(ctypes.Structure):
    _fields_ = [
        ("fi_openflags", ctypes.c_uint32),
        ("fi_status", ctypes.c_uint32),
        ("fi_offset", ctypes.c_int64),
        ("fi_type", ctypes.c_int32),
        ("fi_guardflags", ctypes.c_uint32),
    ]


class _DarwinVInfoStat(ctypes.Structure):
    _fields_ = [
        ("vst_dev", ctypes.c_uint32),
        ("vst_mode", ctypes.c_uint16),
        ("vst_nlink", ctypes.c_uint16),
        ("vst_ino", ctypes.c_uint64),
        ("vst_uid", ctypes.c_uint32),
        ("vst_gid", ctypes.c_uint32),
        ("vst_atime", ctypes.c_int64),
        ("vst_atimensec", ctypes.c_int64),
        ("vst_mtime", ctypes.c_int64),
        ("vst_mtimensec", ctypes.c_int64),
        ("vst_ctime", ctypes.c_int64),
        ("vst_ctimensec", ctypes.c_int64),
        ("vst_birthtime", ctypes.c_int64),
        ("vst_birthtimensec", ctypes.c_int64),
        ("vst_size", ctypes.c_int64),
        ("vst_blocks", ctypes.c_int64),
        ("vst_blksize", ctypes.c_int32),
        ("vst_flags", ctypes.c_uint32),
        ("vst_gen", ctypes.c_uint32),
        ("vst_rdev", ctypes.c_uint32),
        ("vst_qspare", ctypes.c_int64 * 2),
    ]


class _DarwinVNodeInfo(ctypes.Structure):
    _fields_ = [
        ("vi_stat", _DarwinVInfoStat),
        ("vi_type", ctypes.c_int),
        ("vi_pad", ctypes.c_int),
        ("vi_fsid", ctypes.c_int32 * 2),
    ]


class _DarwinVNodeInfoPath(ctypes.Structure):
    _fields_ = [
        ("vip_vi", _DarwinVNodeInfo),
        ("vip_path", ctypes.c_char * _DARWIN_MAXPATHLEN),
    ]


class _DarwinVNodeFdInfoWithPath(ctypes.Structure):
    _fields_ = [
        ("pfi", _DarwinProcFileInfo),
        ("pvip", _DarwinVNodeInfoPath),
    ]


def _darwin_open_file_identity(
    *,
    pid: int,
    fd: int,
) -> OpenFileIdentityObservation | None:
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidfdinfo = libproc.proc_pidfdinfo
        proc_pidfdinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidfdinfo.restype = ctypes.c_int
        buffer = _DarwinVNodeFdInfoWithPath()
        received = proc_pidfdinfo(
            pid,
            fd,
            _DARWIN_PROC_PIDFDVNODEPATHINFO,
            ctypes.byref(buffer),
            ctypes.sizeof(buffer),
        )
        if received != ctypes.sizeof(buffer):
            return None
        vnode = buffer.pvip.vip_vi.vi_stat
        raw_path = bytes(buffer.pvip.vip_path).split(b"\0", 1)[0]
        path = os.fsdecode(raw_path)
        if not path:
            return None
        return OpenFileIdentityObservation(
            path=path,
            fd=fd,
            device=int(vnode.vst_dev),
            inode=int(vnode.vst_ino),
            byte_length=int(vnode.vst_size),
            mtime_ns=(int(vnode.vst_mtime) * 1_000_000_000 + int(vnode.vst_mtimensec)),
            mode=int(vnode.vst_mode),
            provider="proc_pidfdinfo",
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _posix_open_file_identity(
    *,
    pid: int,
    fd: int,
    path: str,
) -> OpenFileIdentityObservation | None:
    try:
        if sys.platform == "darwin":
            return _darwin_open_file_identity(pid=pid, fd=fd)
        if sys.platform.startswith("linux"):
            descriptor_path = Path(f"/proc/{pid}/fd/{fd}")
            metadata = descriptor_path.stat()
            observed_path = os.readlink(descriptor_path)
            return OpenFileIdentityObservation(
                path=observed_path,
                fd=fd,
                device=int(metadata.st_dev),
                inode=int(metadata.st_ino),
                byte_length=int(metadata.st_size),
                mtime_ns=int(metadata.st_mtime_ns),
                mode=int(metadata.st_mode),
                provider="procfs_fd",
            )
        if pid == os.getpid():
            metadata = os.fstat(fd)
            return OpenFileIdentityObservation(
                path=path,
                fd=fd,
                device=int(metadata.st_dev),
                inode=int(metadata.st_ino),
                byte_length=int(metadata.st_size),
                mtime_ns=int(metadata.st_mtime_ns),
                mode=int(metadata.st_mode),
                provider="self_fstat",
            )
    except (OSError, TypeError, ValueError):
        return None
    return None


class HostResourceObserver:
    """Read actual host resources through bounded platform adapters."""

    def __init__(
        self,
        *,
        source: ObservationSource = ObservationSource.HOST,
        scenario_id: str = "runtime",
    ) -> None:
        if source not in {
            ObservationSource.HOST,
            ObservationSource.LIVE_PRESSURE,
        }:
            raise ValueError("HostResourceObserver requires a host-backed source")
        self._source = source
        self._scenario_id = str(scenario_id or "runtime")
        # Short-TTL cache for the full process table. A psutil process_iter
        # over the whole host costs seconds of syscalls on macOS, and the
        # Jul 24 boot-stall dumps caught it running ON the event loop
        # (resource_observation.processes -> process_table). Observation
        # freshness of a couple of seconds is well inside every consumer's
        # tolerance — the scan itself takes that long.
        self._process_table_cache: tuple[float, ProcessTableObservation] | None = None
        self._process_table_cache_ttl_s = 2.0
        # The same treatment for the process-TREE walk in memory(), which the
        # Jul 24 pass missed because it reaches the host through
        # Process.children(recursive=True) rather than process_iter. It is the
        # same host-wide enumeration and it kept the same habit of running on
        # the event loop: on Jul 29 it turned a 5.2s lag into a 63.5s freeze,
        # because record_degradation sampled RSS and every record bought the
        # next one. Only the children's total is cached — this process's own
        # RSS is one cheap call and stays live.
        self._tree_children_rss_cache: dict[int, tuple[float, int]] = {}
        self._tree_children_rss_ttl_s = 2.0

    @property
    def provenance(self) -> ObservationProvenance:
        return ObservationProvenance(
            source=self._source,
            scenario_id=self._scenario_id,
            observer=type(self).__name__,
        )

    def _process_from_handle(self, process: psutil.Process) -> ProcessObservation | None:
        provenance = self.provenance
        try:
            with process.oneshot():
                pid = int(process.pid)
                ppid = int(process.ppid())
                create_time = float(process.create_time())
                status = str(process.status())
                name = str(process.name() or "")
                cmdline = tuple(str(part) for part in (process.cmdline() or ()))
                rss_bytes = int(getattr(process.memory_info(), "rss", 0) or 0)
                try:
                    memory_percent = float(process.memory_percent() or 0.0)
                except (psutil.Error, OSError, RuntimeError, TypeError, ValueError):
                    memory_percent = 0.0
                try:
                    cpu_percent = float(process.cpu_percent(interval=None) or 0.0)
                except (psutil.Error, OSError, RuntimeError, TypeError, ValueError):
                    cpu_percent = 0.0
                try:
                    cpu_times = process.cpu_times()
                    cpu_user_seconds = float(getattr(cpu_times, "user", 0.0) or 0.0)
                    cpu_system_seconds = float(getattr(cpu_times, "system", 0.0) or 0.0)
                except (psutil.Error, OSError, RuntimeError, TypeError, ValueError):
                    cpu_user_seconds = 0.0
                    cpu_system_seconds = 0.0
                try:
                    num_threads = int(process.num_threads() or 0)
                except (psutil.Error, OSError, RuntimeError, TypeError, ValueError):
                    num_threads = 0
                try:
                    num_fds_reader = getattr(process, "num_fds", None)
                    num_fds = int(num_fds_reader() or 0) if callable(num_fds_reader) else 0
                except (psutil.Error, OSError, RuntimeError, TypeError, ValueError):
                    num_fds = 0
                try:
                    exe = str(process.exe() or "")
                except (psutil.Error, OSError, RuntimeError, ValueError):
                    exe = ""
                try:
                    username = str(process.username() or "")
                except (psutil.Error, OSError, RuntimeError, ValueError):
                    username = ""
                try:
                    cwd = str(process.cwd() or "")
                except (psutil.Error, OSError, RuntimeError, ValueError):
                    cwd = ""
            try:
                ancestors = tuple(int(parent.pid) for parent in process.parents())
            except (psutil.Error, OSError, RuntimeError, ValueError):
                ancestors = ()
            return ProcessObservation(
                provenance=provenance,
                pid=pid,
                ppid=ppid,
                create_time=create_time,
                status=status,
                name=name,
                cmdline=cmdline,
                rss_bytes=rss_bytes,
                memory_percent=max(0.0, memory_percent),
                cpu_percent=max(0.0, cpu_percent),
                cpu_user_seconds=max(0.0, cpu_user_seconds),
                cpu_system_seconds=max(0.0, cpu_system_seconds),
                num_threads=max(0, num_threads),
                num_fds=max(0, num_fds),
                ancestor_pids=ancestors,
                exe=exe,
                username=username,
                cwd=cwd,
            )
        except (psutil.Error, OSError, RuntimeError, SystemError, TypeError, ValueError):
            return None

    def process(self, pid: int) -> ProcessObservation | None:
        try:
            return self._process_from_handle(psutil.Process(int(pid)))
        except (psutil.Error, OSError, RuntimeError, SystemError, TypeError, ValueError):
            return None

    def process_ids(self) -> ProcessIdsObservation:
        """Return only host PIDs for count/existence telemetry.

        ``process_table`` intentionally enriches every process and can take
        seconds on a busy workstation. Callers that only need a census must
        never pay that cost or run it on an async owner loop.
        """

        try:
            observed = tuple(sorted({int(pid) for pid in psutil.pids()}))
            return ProcessIdsObservation(
                provenance=self.provenance,
                pids=observed,
            )
        except (
            psutil.Error,
            OSError,
            RuntimeError,
            SystemError,
            TypeError,
            ValueError,
        ) as exc:
            return ProcessIdsObservation(
                provenance=self.provenance,
                pids=(),
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )

    def _lightweight_process_from_handle(
        self,
        process: psutil.Process,
    ) -> ProcessObservation | None:
        """Observe identity and RSS without expensive whole-process metadata."""

        try:
            with process.oneshot():
                pid = int(process.pid)
                ppid = int(process.ppid())
                create_time = float(process.create_time())
                status = str(process.status())
        except (psutil.Error, OSError, RuntimeError, SystemError, TypeError, ValueError):
            return None
        try:
            rss_bytes = int(getattr(process.memory_info(), "rss", 0) or 0)
        except (psutil.Error, OSError, RuntimeError, SystemError, TypeError, ValueError):
            # Identity is authoritative for process-lifetime decisions. A
            # transient RSS read failure must degrade one metric, not make a
            # live guarded process appear to have exited.
            rss_bytes = 0
        return ProcessObservation(
            provenance=self.provenance,
            pid=pid,
            ppid=ppid,
            create_time=create_time,
            status=status,
            name="",
            cmdline=(),
            rss_bytes=rss_bytes,
        )

    def process_tree(
        self,
        root_pid: int,
        *,
        recursive: bool = True,
    ) -> ProcessTableObservation:
        """Observe one process tree without enumerating every host process.

        This is the hot-path counterpart to :meth:`process_table`. It collects
        only identity, status, and RSS because watchdogs do not need command
        lines, CPU sampling, filesystem identity, or parent-chain expansion on
        every tick.
        """

        try:
            root = psutil.Process(int(root_pid))
            handles = (root, *root.children(recursive=bool(recursive)))
        except (
            psutil.Error,
            OSError,
            RuntimeError,
            SystemError,
            TypeError,
            ValueError,
        ) as exc:
            return ProcessTableObservation(
                provenance=self.provenance,
                processes=(),
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )

        observed: list[ProcessObservation] = []
        seen_pids: set[int] = set()
        for handle in handles:
            item = self._lightweight_process_from_handle(handle)
            if item is None or item.pid in seen_pids:
                continue
            seen_pids.add(item.pid)
            observed.append(item)
        root_present = bool(observed and observed[0].pid == int(root_pid))
        return ProcessTableObservation(
            provenance=self.provenance,
            processes=tuple(observed),
            available=root_present,
            error="" if root_present else "root_process_unobservable",
        )

    def process_table(self) -> ProcessTableObservation:
        cached = self._process_table_cache
        now = time.monotonic()
        if cached is not None and now - cached[0] < self._process_table_cache_ttl_s:
            return cached[1]
        observed: list[ProcessObservation] = []
        try:
            iterator = psutil.process_iter()
            for process in iterator:
                item = self._process_from_handle(process)
                if item is not None:
                    observed.append(item)
        except (
            psutil.Error,
            OSError,
            RuntimeError,
            SystemError,
            TypeError,
            ValueError,
        ) as exc:
            # Failures are never cached: the next caller retries the scan.
            return ProcessTableObservation(
                provenance=self.provenance,
                processes=tuple(observed),
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )
        result = ProcessTableObservation(
            provenance=self.provenance,
            processes=tuple(observed),
        )
        self._process_table_cache = (now, result)
        return result

    def processes(self) -> tuple[ProcessObservation, ...]:
        return self.process_table().processes

    def connection_table(
        self,
        *,
        kind: str = "inet",
        pid: int | None = None,
    ) -> ConnectionTableObservation:
        provenance = self.provenance
        try:
            target_pid = None if pid is None else int(pid)
            raw_connections = (
                psutil.net_connections(kind=kind)
                if target_pid is None
                else psutil.Process(target_pid).net_connections(kind=kind)
            )
        except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            return ConnectionTableObservation(
                provenance=provenance,
                connections=(),
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )
        observed: list[NetworkConnectionObservation] = []
        for connection in raw_connections:
            try:
                local = getattr(connection, "laddr", None)
                remote = getattr(connection, "raddr", None)
                raw_fd = getattr(connection, "fd", -1)
                observed.append(
                    NetworkConnectionObservation(
                        provenance=provenance,
                        pid=int(getattr(connection, "pid", 0) or target_pid or 0),
                        fd=-1 if raw_fd is None else int(raw_fd),
                        family=str(getattr(connection, "family", "") or ""),
                        socket_type=str(getattr(connection, "type", "") or ""),
                        local_host=str(getattr(local, "ip", "") or ""),
                        local_port=int(getattr(local, "port", 0) or 0),
                        remote_host=str(getattr(remote, "ip", "") or ""),
                        remote_port=int(getattr(remote, "port", 0) or 0),
                        status=str(getattr(connection, "status", "") or ""),
                    )
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                continue
        return ConnectionTableObservation(
            provenance=provenance,
            connections=tuple(observed),
        )

    def connections(
        self,
        *,
        kind: str = "inet",
        pid: int | None = None,
    ) -> tuple[NetworkConnectionObservation, ...]:
        return self.connection_table(kind=kind, pid=pid).connections

    def open_file_table(self, *, pid: int | None = None) -> OpenFilesObservation:
        target_pid = os.getpid() if pid is None else int(pid)
        try:
            open_files = tuple(psutil.Process(target_pid).open_files())
            paths = tuple(
                str(getattr(item, "path", "") or "")
                for item in open_files
                if str(getattr(item, "path", "") or "")
            )
            identities = tuple(
                identity
                for item in open_files
                if (
                    identity := _posix_open_file_identity(
                        pid=target_pid,
                        fd=int(getattr(item, "fd", -1)),
                        path=str(getattr(item, "path", "") or ""),
                    )
                )
                is not None
            )
            return OpenFilesObservation(
                provenance=self.provenance,
                pid=target_pid,
                paths=paths,
                identities=identities,
            )
        except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            return OpenFilesObservation(
                provenance=self.provenance,
                pid=target_pid,
                paths=(),
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )

    def open_files(self, *, pid: int | None = None) -> tuple[str, ...]:
        return self.open_file_table(pid=pid).paths

    def _children_rss_bytes(self, process: Any, root: int) -> int:
        """Summed RSS of this process's descendants, at most once per TTL.

        The walk is the expensive half of a tree observation and several
        subsystems ask for one per tick. Failures are never cached, so the
        next caller retries the scan rather than inheriting a zero.
        """
        now = time.monotonic()
        cached = self._tree_children_rss_cache.get(root)
        if cached is not None and now - cached[0] < self._tree_children_rss_ttl_s:
            return cached[1]
        total = 0
        for child in process.children(recursive=True):
            try:
                total += int(getattr(child.memory_info(), "rss", 0) or 0)
            except (psutil.Error, OSError, RuntimeError, ValueError):
                continue
        self._tree_children_rss_cache[root] = (now, total)
        return total

    def memory(
        self,
        *,
        root_pid: int | None = None,
        include_process_tree: bool = True,
    ) -> MemoryObservation:
        provenance = self.provenance
        try:
            vm = psutil.virtual_memory()
            total = int(getattr(vm, "total", 0) or 0)
            available = int(getattr(vm, "available", 0) or 0)
            used = int(getattr(vm, "used", max(0, total - available)) or 0)
            free = int(getattr(vm, "free", available) or 0)
            active = int(getattr(vm, "active", used) or 0)
            percent = float(getattr(vm, "percent", 0.0) or 0.0)
            if percent <= 0.0 and total > 0:
                percent = max(0.0, min(100.0, (1.0 - available / total) * 100.0))
        except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            return MemoryObservation(
                provenance=provenance,
                total_bytes=0,
                available_bytes=0,
                used_bytes=0,
                free_bytes=0,
                active_bytes=0,
                percent=0.0,
                process_rss_bytes=0,
                process_tree_rss_bytes=0,
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )

        # OUR OWN RSS IS CHEAP. THE TREE'S IS THE WHOLE MACHINE'S.
        #
        # One memory_info() is a single mach call (~3us). children(recursive)
        # is not a walk of our children — psutil reaches it through
        # _ppid_map(), which enumerates EVERY pid on the host and builds a
        # Process for each. Measured here at 7ms against 751 pids while idle,
        # and far worse while a 32B worker is spawning, because the pid
        # enumeration contends with process creation.
        #
        # include_process_tree=False used to skip both and report an RSS of
        # zero, so a caller that wanted only this process either paid the
        # machine's bill or was told it occupied no memory. Sample the cheap
        # one always; gate only the scan.
        process_rss = 0
        tree_rss = 0
        root = os.getpid() if root_pid is None else int(root_pid)
        try:
            process = psutil.Process(root)
            process_rss = int(getattr(process.memory_info(), "rss", 0) or 0)
            tree_rss = process_rss
            if include_process_tree:
                tree_rss += self._children_rss_bytes(process, root)
        except (psutil.Error, OSError, RuntimeError, TypeError, ValueError):
            pass
        swap_total = 0
        swap_used = 0
        swap_free = 0
        swap_percent = 0.0
        try:
            swap = psutil.swap_memory()
            swap_total = int(getattr(swap, "total", 0) or 0)
            swap_used = int(getattr(swap, "used", 0) or 0)
            swap_free = int(getattr(swap, "free", 0) or 0)
            swap_percent = float(getattr(swap, "percent", 0.0) or 0.0)
        except (psutil.Error, OSError, RuntimeError, TypeError, ValueError):
            pass
        return MemoryObservation(
            provenance=provenance,
            total_bytes=total,
            available_bytes=available,
            used_bytes=used,
            free_bytes=free,
            active_bytes=active,
            percent=max(0.0, min(100.0, percent)),
            process_rss_bytes=process_rss,
            process_tree_rss_bytes=tree_rss,
            swap_total_bytes=swap_total,
            swap_used_bytes=swap_used,
            swap_free_bytes=swap_free,
            swap_percent=max(0.0, min(100.0, swap_percent)),
        )

    def disk(self, path: str | os.PathLike[str] = "/") -> DiskObservation:
        provenance = self.provenance
        normalized = str(Path(path).expanduser())
        try:
            usage = shutil.disk_usage(normalized)
            total = int(usage.total)
            used = int(usage.used)
            free = int(usage.free)
            percent = 0.0 if total <= 0 else used / total * 100.0
            return DiskObservation(
                provenance=provenance,
                path=normalized,
                total_bytes=total,
                used_bytes=used,
                free_bytes=free,
                percent=max(0.0, min(100.0, percent)),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return DiskObservation(
                provenance=provenance,
                path=normalized,
                total_bytes=0,
                used_bytes=0,
                free_bytes=0,
                percent=0.0,
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )

    def thermal(self, *, max_age_s: float = 5.0) -> ThermalObservation:
        provenance = self.provenance
        try:
            from core.runtime.thermal import thermal_state

            reading = thermal_state(max_age_s=max_age_s)
            return ThermalObservation(
                provenance=provenance,
                level=max(0, min(3, int(reading.level))),
                provider=str(reading.source),
                detail=str(reading.detail or ""),
                available=not bool(reading.blind),
            )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return ThermalObservation(
                provenance=provenance,
                level=0,
                provider="blind",
                detail=f"{type(exc).__name__}:{exc}",
                available=False,
            )

    def accelerator(self) -> AcceleratorObservation:
        provenance = self.provenance
        # Sampling must never import/initialize Metal just to answer a health
        # query.  Observe it only when the runtime already owns the module.
        mx = sys.modules.get("mlx.core")
        metal = getattr(mx, "metal", None) if mx is not None else None
        if metal is None:
            return AcceleratorObservation(
                provenance=provenance,
                provider="mlx_metal",
                available=False,
                detail="mlx.core not resident",
            )
        try:
            active_reader = getattr(mx, "get_active_memory", None) or getattr(
                metal,
                "get_active_memory",
                None,
            )
            cache_reader = getattr(mx, "get_cache_memory", None) or getattr(
                metal,
                "get_cache_memory",
                None,
            )
            peak_reader = getattr(mx, "get_peak_memory", None) or getattr(
                metal,
                "get_peak_memory",
                None,
            )
            return AcceleratorObservation(
                provenance=provenance,
                provider="mlx_metal",
                active_bytes=int(active_reader() if callable(active_reader) else 0),
                cache_bytes=int(cache_reader() if callable(cache_reader) else 0),
                peak_bytes=int(peak_reader() if callable(peak_reader) else 0),
                available=any(callable(reader) for reader in (active_reader, cache_reader, peak_reader)),
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return AcceleratorObservation(
                provenance=provenance,
                provider="mlx_metal",
                available=False,
                detail=f"{type(exc).__name__}:{exc}",
            )

    def compute(self) -> ComputeObservation:
        provenance = self.provenance
        try:
            try:
                load_1m, load_5m, load_15m = (float(value) for value in os.getloadavg())
            except (AttributeError, OSError):
                load_1m = load_5m = load_15m = 0.0
            cpu_times = psutil.cpu_times()
            return ComputeObservation(
                provenance=provenance,
                cpu_percent=float(psutil.cpu_percent(interval=None) or 0.0),
                cpu_count=max(1, int(psutil.cpu_count() or os.cpu_count() or 1)),
                load_1m=max(0.0, load_1m),
                load_5m=max(0.0, load_5m),
                load_15m=max(0.0, load_15m),
                boot_time=max(0.0, float(psutil.boot_time() or 0.0)),
                cpu_user_seconds=max(0.0, float(getattr(cpu_times, "user", 0.0) or 0.0)),
                cpu_system_seconds=max(
                    0.0,
                    float(getattr(cpu_times, "system", 0.0) or 0.0),
                ),
                cpu_idle_seconds=max(0.0, float(getattr(cpu_times, "idle", 0.0) or 0.0)),
            )
        except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            return ComputeObservation(
                provenance=provenance,
                cpu_percent=0.0,
                cpu_count=max(1, int(os.cpu_count() or 1)),
                load_1m=0.0,
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )

    def power(self) -> PowerObservation:
        provenance = self.provenance
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return PowerObservation(
                    provenance=provenance,
                    battery_percent=100.0,
                    plugged=True,
                    available=False,
                    error="battery_not_reported",
                )
            return PowerObservation(
                provenance=provenance,
                battery_percent=max(
                    0.0,
                    min(100.0, float(getattr(battery, "percent", 100.0) or 0.0)),
                ),
                plugged=bool(getattr(battery, "power_plugged", False)),
                seconds_left=int(getattr(battery, "secsleft", -2) or -2),
            )
        except (psutil.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            return PowerObservation(
                provenance=provenance,
                battery_percent=100.0,
                plugged=True,
                available=False,
                error=f"{type(exc).__name__}:{exc}",
            )

    def snapshot(
        self,
        *,
        path: str | os.PathLike[str] = "/",
        include_processes: bool = False,
    ) -> ResourceObservation:
        provenance = self.provenance
        return ResourceObservation(
            provenance=provenance,
            memory=self.memory(),
            disk=self.disk(path),
            thermal=self.thermal(),
            accelerator=self.accelerator(),
            compute=self.compute(),
            power=self.power(),
            processes=self.processes() if include_processes else (),
        )


class SimulatedResourceObserver:
    """Deterministic observer for unit/integration tests and simulations."""

    def __init__(
        self,
        *,
        scenario_id: str = "deterministic-normal",
        total_memory_bytes: int = 64 * 1024**3,
        available_memory_bytes: int = 32 * 1024**3,
        memory_percent: float = 50.0,
        process_rss_bytes: int = 1024**3,
        process_tree_rss_bytes: int = 1024**3,
        swap_total_bytes: int = 0,
        swap_used_bytes: int = 0,
        disk_total_bytes: int = 1024**4,
        disk_free_bytes: int = 768 * 1024**3,
        thermal_level: int = 0,
        thermal_provider: str = "simulated",
        accelerator_active_bytes: int = 0,
        accelerator_cache_bytes: int = 0,
        accelerator_peak_bytes: int = 0,
        accelerator_available: bool = False,
        cpu_percent: float = 10.0,
        cpu_count: int = 8,
        load_1m: float = 0.5,
        load_5m: float = 0.5,
        load_15m: float = 0.5,
        boot_time: float = 1_700_000_000.0,
        cpu_user_seconds: float = 0.0,
        cpu_system_seconds: float = 0.0,
        cpu_idle_seconds: float = 0.0,
        battery_percent: float = 100.0,
        power_plugged: bool = True,
        battery_seconds_left: int = -2,
        processes: Sequence[ProcessObservation] = (),
        connections: Sequence[NetworkConnectionObservation] = (),
        open_files: Sequence[str] = (),
    ) -> None:
        self._scenario_id = str(scenario_id or "deterministic-normal")
        provenance = self.provenance
        total = max(0, int(total_memory_bytes))
        available = max(0, min(total, int(available_memory_bytes)))
        used = max(0, total - available)
        swap_total = max(0, int(swap_total_bytes))
        swap_used = max(0, min(swap_total, int(swap_used_bytes)))
        self._memory = MemoryObservation(
            provenance=provenance,
            total_bytes=total,
            available_bytes=available,
            used_bytes=used,
            free_bytes=available,
            active_bytes=used,
            percent=max(0.0, min(100.0, float(memory_percent))),
            process_rss_bytes=max(0, int(process_rss_bytes)),
            process_tree_rss_bytes=max(0, int(process_tree_rss_bytes)),
            swap_total_bytes=swap_total,
            swap_used_bytes=swap_used,
            swap_free_bytes=max(0, swap_total - swap_used),
            swap_percent=(
                0.0
                if swap_total <= 0
                else swap_used / swap_total * 100.0
            ),
        )
        disk_total = max(0, int(disk_total_bytes))
        disk_free = max(0, min(disk_total, int(disk_free_bytes)))
        disk_used = max(0, disk_total - disk_free)
        self._disk = DiskObservation(
            provenance=provenance,
            path="/",
            total_bytes=disk_total,
            used_bytes=disk_used,
            free_bytes=disk_free,
            percent=0.0 if disk_total <= 0 else disk_used / disk_total * 100.0,
        )
        self._thermal = ThermalObservation(
            provenance=provenance,
            level=max(0, min(3, int(thermal_level))),
            provider=str(thermal_provider or "simulated"),
        )
        self._accelerator = AcceleratorObservation(
            provenance=provenance,
            provider="simulated",
            active_bytes=max(0, int(accelerator_active_bytes)),
            cache_bytes=max(0, int(accelerator_cache_bytes)),
            peak_bytes=max(0, int(accelerator_peak_bytes)),
            available=bool(accelerator_available),
        )
        self._compute = ComputeObservation(
            provenance=provenance,
            cpu_percent=max(0.0, min(100.0, float(cpu_percent))),
            cpu_count=max(1, int(cpu_count)),
            load_1m=max(0.0, float(load_1m)),
            load_5m=max(0.0, float(load_5m)),
            load_15m=max(0.0, float(load_15m)),
            boot_time=max(0.0, float(boot_time)),
            cpu_user_seconds=max(0.0, float(cpu_user_seconds)),
            cpu_system_seconds=max(0.0, float(cpu_system_seconds)),
            cpu_idle_seconds=max(0.0, float(cpu_idle_seconds)),
        )
        self._power = PowerObservation(
            provenance=provenance,
            battery_percent=max(0.0, min(100.0, float(battery_percent))),
            plugged=bool(power_plugged),
            seconds_left=int(battery_seconds_left),
            provider="simulated",
        )
        supplied_processes = tuple(processes)
        if not supplied_processes:
            supplied_processes = (
                ProcessObservation(
                    provenance=provenance,
                    pid=os.getpid(),
                    ppid=os.getppid(),
                    create_time=1.0,
                    status="running",
                    name="simulated-current-process",
                    cmdline=(),
                    rss_bytes=max(0, int(process_rss_bytes)),
                ),
            )
        self._processes = tuple(
            self._with_provenance(process) for process in supplied_processes
        )
        self._connections = tuple(
            replace(connection, provenance=provenance) for connection in connections
        )
        self._open_files = tuple(str(path) for path in open_files)
        self._lock = threading.RLock()

    @property
    def provenance(self) -> ObservationProvenance:
        return ObservationProvenance(
            source=ObservationSource.SIMULATED,
            scenario_id=self._scenario_id,
            observer=type(self).__name__,
        )

    def _with_provenance(self, process: ProcessObservation) -> ProcessObservation:
        return replace(process, provenance=self.provenance)

    def configure_memory(
        self,
        *,
        total_bytes: int | None = None,
        available_bytes: int | None = None,
        percent: float | None = None,
        process_rss_bytes: int | None = None,
        process_tree_rss_bytes: int | None = None,
        swap_total_bytes: int | None = None,
        swap_used_bytes: int | None = None,
        observation_available: bool | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            current = self._memory
            total = current.total_bytes if total_bytes is None else max(0, int(total_bytes))
            available = (
                current.available_bytes
                if available_bytes is None
                else max(0, min(total, int(available_bytes)))
            )
            used = max(0, total - available)
            self._memory = replace(
                current,
                provenance=self.provenance,
                total_bytes=total,
                available_bytes=available,
                used_bytes=used,
                free_bytes=available,
                active_bytes=used,
                percent=current.percent if percent is None else max(0.0, min(100.0, float(percent))),
                process_rss_bytes=(
                    current.process_rss_bytes
                    if process_rss_bytes is None
                    else max(0, int(process_rss_bytes))
                ),
                process_tree_rss_bytes=(
                    current.process_tree_rss_bytes
                    if process_tree_rss_bytes is None
                    else max(0, int(process_tree_rss_bytes))
                ),
                swap_total_bytes=(
                    current.swap_total_bytes
                    if swap_total_bytes is None
                    else max(0, int(swap_total_bytes))
                ),
                swap_used_bytes=(
                    current.swap_used_bytes
                    if swap_used_bytes is None
                    else max(0, int(swap_used_bytes))
                ),
                available=(
                    current.available
                    if observation_available is None
                    else bool(observation_available)
                ),
                error=current.error if error is None else str(error),
            )
            swap_total = self._memory.swap_total_bytes
            swap_used = min(swap_total, self._memory.swap_used_bytes)
            self._memory = replace(
                self._memory,
                swap_used_bytes=swap_used,
                swap_free_bytes=max(0, swap_total - swap_used),
                swap_percent=(
                    0.0 if swap_total <= 0 else swap_used / swap_total * 100.0
                ),
            )

    def configure_disk(
        self,
        *,
        total_bytes: int | None = None,
        free_bytes: int | None = None,
        observation_available: bool | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            current = self._disk
            total = current.total_bytes if total_bytes is None else max(0, int(total_bytes))
            free = current.free_bytes if free_bytes is None else max(0, min(total, int(free_bytes)))
            used = max(0, total - free)
            self._disk = replace(
                current,
                provenance=self.provenance,
                total_bytes=total,
                used_bytes=used,
                free_bytes=free,
                percent=0.0 if total <= 0 else used / total * 100.0,
                available=(
                    current.available
                    if observation_available is None
                    else bool(observation_available)
                ),
                error=current.error if error is None else str(error),
            )

    def configure_thermal(self, level: int, *, provider: str = "simulated") -> None:
        with self._lock:
            self._thermal = replace(
                self._thermal,
                provenance=self.provenance,
                level=max(0, min(3, int(level))),
                provider=str(provider or "simulated"),
            )

    def configure_accelerator(
        self,
        *,
        active_bytes: int,
        cache_bytes: int = 0,
        peak_bytes: int = 0,
        available: bool = True,
    ) -> None:
        with self._lock:
            self._accelerator = replace(
                self._accelerator,
                provenance=self.provenance,
                active_bytes=max(0, int(active_bytes)),
                cache_bytes=max(0, int(cache_bytes)),
                peak_bytes=max(0, int(peak_bytes)),
                available=bool(available),
            )

    def configure_compute(
        self,
        *,
        cpu_percent: float | None = None,
        cpu_count: int | None = None,
        load_1m: float | None = None,
        load_5m: float | None = None,
        load_15m: float | None = None,
        boot_time: float | None = None,
    ) -> None:
        with self._lock:
            self._compute = replace(
                self._compute,
                provenance=self.provenance,
                cpu_percent=(
                    self._compute.cpu_percent
                    if cpu_percent is None
                    else max(0.0, min(100.0, float(cpu_percent)))
                ),
                cpu_count=(
                    self._compute.cpu_count if cpu_count is None else max(1, int(cpu_count))
                ),
                load_1m=(
                    self._compute.load_1m if load_1m is None else max(0.0, float(load_1m))
                ),
                load_5m=(
                    self._compute.load_5m if load_5m is None else max(0.0, float(load_5m))
                ),
                load_15m=(
                    self._compute.load_15m
                    if load_15m is None
                    else max(0.0, float(load_15m))
                ),
                boot_time=(
                    self._compute.boot_time
                    if boot_time is None
                    else max(0.0, float(boot_time))
                ),
            )

    def configure_power(
        self,
        *,
        battery_percent: float | None = None,
        plugged: bool | None = None,
        seconds_left: int | None = None,
    ) -> None:
        with self._lock:
            self._power = replace(
                self._power,
                provenance=self.provenance,
                battery_percent=(
                    self._power.battery_percent
                    if battery_percent is None
                    else max(0.0, min(100.0, float(battery_percent)))
                ),
                plugged=self._power.plugged if plugged is None else bool(plugged),
                seconds_left=(
                    self._power.seconds_left
                    if seconds_left is None
                    else int(seconds_left)
                ),
            )

    def configure_processes(self, processes: Sequence[ProcessObservation]) -> None:
        with self._lock:
            self._processes = tuple(self._with_provenance(process) for process in processes)

    def configure_connections(
        self,
        connections: Sequence[NetworkConnectionObservation],
    ) -> None:
        with self._lock:
            self._connections = tuple(
                replace(connection, provenance=self.provenance)
                for connection in connections
            )

    def configure_open_files(self, paths: Sequence[str | os.PathLike[str]]) -> None:
        with self._lock:
            self._open_files = tuple(str(path) for path in paths)

    def memory(
        self,
        *,
        root_pid: int | None = None,
        include_process_tree: bool = True,
    ) -> MemoryObservation:
        del root_pid, include_process_tree
        with self._lock:
            return replace(self._memory, provenance=self.provenance)

    def disk(self, path: str | os.PathLike[str] = "/") -> DiskObservation:
        with self._lock:
            return replace(self._disk, provenance=self.provenance, path=str(path))

    def thermal(self, *, max_age_s: float = 5.0) -> ThermalObservation:
        del max_age_s
        with self._lock:
            return replace(self._thermal, provenance=self.provenance)

    def accelerator(self) -> AcceleratorObservation:
        with self._lock:
            return replace(self._accelerator, provenance=self.provenance)

    def compute(self) -> ComputeObservation:
        with self._lock:
            return replace(self._compute, provenance=self.provenance)

    def power(self) -> PowerObservation:
        with self._lock:
            return replace(self._power, provenance=self.provenance)

    def processes(self) -> tuple[ProcessObservation, ...]:
        with self._lock:
            return tuple(self._with_provenance(process) for process in self._processes)

    def process_ids(self) -> ProcessIdsObservation:
        with self._lock:
            pids = tuple(sorted({int(process.pid) for process in self._processes}))
        return ProcessIdsObservation(
            provenance=self.provenance,
            pids=pids,
        )

    def process_table(self) -> ProcessTableObservation:
        return ProcessTableObservation(
            provenance=self.provenance,
            processes=self.processes(),
        )

    def process(self, pid: int) -> ProcessObservation | None:
        for process in self.processes():
            if process.pid == int(pid):
                return process
        return None

    def process_tree(
        self,
        root_pid: int,
        *,
        recursive: bool = True,
    ) -> ProcessTableObservation:
        root = int(root_pid)
        processes = self.processes()
        root_process = next((process for process in processes if process.pid == root), None)
        if root_process is None:
            return ProcessTableObservation(
                provenance=self.provenance,
                processes=(),
                available=False,
                error="root_process_unobservable",
            )
        if recursive:
            descendants = [
                process
                for process in processes
                if process.pid != root and root in process.ancestor_pids
            ]
        else:
            descendants = [
                process
                for process in processes
                if process.pid != root and process.ppid == root
            ]
        return ProcessTableObservation(
            provenance=self.provenance,
            processes=(root_process, *descendants),
        )

    def connections(
        self,
        *,
        kind: str = "inet",
        pid: int | None = None,
    ) -> tuple[NetworkConnectionObservation, ...]:
        del kind
        with self._lock:
            observed = tuple(
                replace(connection, provenance=self.provenance)
                for connection in self._connections
            )
        if pid is None:
            return observed
        target_pid = int(pid)
        return tuple(connection for connection in observed if connection.pid == target_pid)

    def connection_table(
        self,
        *,
        kind: str = "inet",
        pid: int | None = None,
    ) -> ConnectionTableObservation:
        return ConnectionTableObservation(
            provenance=self.provenance,
            connections=self.connections(kind=kind, pid=pid),
        )

    def open_files(self, *, pid: int | None = None) -> tuple[str, ...]:
        del pid
        with self._lock:
            return tuple(self._open_files)

    def open_file_table(self, *, pid: int | None = None) -> OpenFilesObservation:
        target_pid = os.getpid() if pid is None else int(pid)
        return OpenFilesObservation(
            provenance=self.provenance,
            pid=target_pid,
            paths=self.open_files(pid=target_pid),
        )

    def snapshot(
        self,
        *,
        path: str | os.PathLike[str] = "/",
        include_processes: bool = False,
    ) -> ResourceObservation:
        provenance = self.provenance
        return ResourceObservation(
            provenance=provenance,
            memory=self.memory(),
            disk=self.disk(path),
            thermal=self.thermal(),
            accelerator=self.accelerator(),
            compute=self.compute(),
            power=self.power(),
            processes=self.processes() if include_processes else (),
        )


_OBSERVER_LOCK = threading.RLock()
_HOST_OBSERVER = HostResourceObserver()
_PROCESS_OBSERVER: ResourceObserver | None = None


def get_resource_observer() -> ResourceObserver:
    with _OBSERVER_LOCK:
        return _PROCESS_OBSERVER or _HOST_OBSERVER


def set_resource_observer_for_test(observer: ResourceObserver | None) -> ResourceObserver | None:
    """Install a process-wide observer and return the previous override.

    The override is intentionally explicit and test-named. Production callers
    should inject an observer into their owner or use the host default.
    """

    if observer is not None and not isinstance(observer, ResourceObserver):
        raise TypeError("observer does not implement ResourceObserver")
    global _PROCESS_OBSERVER
    with _OBSERVER_LOCK:
        previous = _PROCESS_OBSERVER
        _PROCESS_OBSERVER = observer
        return previous


@contextlib.contextmanager
def resource_observer_scope(observer: ResourceObserver) -> Iterator[ResourceObserver]:
    previous = set_resource_observer_for_test(observer)
    try:
        yield observer
    finally:
        set_resource_observer_for_test(previous)


def assert_live_pressure_observer(observer: ResourceObserver) -> None:
    """Reject simulated or ordinary host samples as live-pressure evidence."""

    provenance = observer.provenance
    if not isinstance(observer, HostResourceObserver):
        raise ValueError("live pressure proof requires HostResourceObserver")
    if not provenance.qualifies_as_live_pressure:
        raise ValueError(
            "live pressure proof requires source=live_pressure; "
            f"received {provenance.source.value}"
        )


__all__ = [
    "AcceleratorObservation",
    "ConnectionTableObservation",
    "ComputeObservation",
    "DiskObservation",
    "HostResourceObserver",
    "MemoryObservation",
    "NetworkConnectionObservation",
    "ObservationProvenance",
    "ObservationSource",
    "OpenFileIdentityObservation",
    "OpenFilesObservation",
    "PowerObservation",
    "ProcessIdsObservation",
    "ProcessObservation",
    "ProcessTableObservation",
    "ResourceObservation",
    "ResourceObserver",
    "SimulatedResourceObserver",
    "ThermalObservation",
    "assert_live_pressure_observer",
    "get_resource_observer",
    "resource_observer_scope",
    "set_resource_observer_for_test",
]
