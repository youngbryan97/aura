"""Compatibility facade for legacy read-only ``psutil`` resource probes.

New policy owners should inject :class:`ResourceObserver` directly.  Older
telemetry and embodiment modules still use the familiar psutil call shape; this
facade routes those resource reads through the canonical observer so pytest's
simulated observer applies consistently.  Non-resource psutil APIs are exposed
through ``__getattr__`` during migration and must not be used for policy.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import psutil as _psutil

from core.runtime.resource_observation import (
    ConnectionTableObservation,
    ProcessObservation,
    get_resource_observer,
)

logger = logging.getLogger(__name__)

_RESOURCE_READ_APIS = frozenset(
    {
        "boot_time",
        "cpu_count",
        "cpu_freq",
        "cpu_percent",
        "cpu_stats",
        "cpu_times",
        "disk_io_counters",
        "disk_usage",
        "getloadavg",
        "net_connections",
        "net_if_addrs",
        "net_io_counters",
        "pid_exists",
        "pids",
        "process_iter",
        "sensors_battery",
        "sensors_temperatures",
        "swap_memory",
        "virtual_memory",
    }
)


class _AttributedMapping(dict[str, Any]):
    def __init__(self, *args: Any, provenance: Any, observation_available: bool, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.provenance = provenance
        self.observation_available = observation_available


def _counter_namespace(raw: Any, fields: tuple[str, ...], *, available: bool = True) -> Any:
    observer = get_resource_observer()
    return SimpleNamespace(
        **{field: getattr(raw, field, 0) if raw is not None else 0 for field in fields},
        provenance=observer.provenance,
        observation_available=available,
    )


def _host_counter(api: str, fields: tuple[str, ...]) -> Any:
    observer = get_resource_observer()
    if not observer.provenance.host_observed:
        return _counter_namespace(None, fields)
    try:
        raw = getattr(_psutil, api)()
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        _psutil.Error,
    ) as exc:
        # None here is read by callers as "this host counter does not exist",
        # which is not the same as a counter that raised on this call.
        logger.debug("host counter %s raised (%s: %s)", api, type(exc).__name__, exc)
        return None
    return None if raw is None else _counter_namespace(raw, fields)


def virtual_memory() -> Any:
    # The psutil compatibility API promises host memory, not recursive process
    # accounting. The latter walks every process parent on macOS and was being
    # paid by hot-path callers every few seconds, including on the event loop.
    # Explicit process-tree consumers still use ResourceObserver.memory().
    memory = get_resource_observer().memory(include_process_tree=False)
    if memory.available:
        return SimpleNamespace(
            total=memory.total_bytes,
            available=memory.available_bytes,
            used=memory.used_bytes,
            free=memory.free_bytes,
            active=memory.active_bytes,
            percent=memory.percent,
            provenance=memory.provenance,
            observation_available=True,
            error="",
        )
    # Existing legacy guards generally compare percent/available without
    # checking provenance. Preserve fail-closed behavior at this adapter.
    return SimpleNamespace(
        total=memory.total_bytes,
        available=0,
        used=memory.total_bytes,
        free=0,
        active=memory.total_bytes,
        percent=100.0,
        provenance=memory.provenance,
        observation_available=False,
        error=memory.error,
    )


def disk_usage(path: str) -> Any:
    disk = get_resource_observer().disk(path)
    if disk.available:
        return disk
    return SimpleNamespace(
        total=disk.total_bytes,
        used=disk.total_bytes,
        free=0,
        percent=100.0,
        provenance=disk.provenance,
        observation_available=False,
        error=disk.error,
    )


def swap_memory() -> Any:
    memory = get_resource_observer().memory()
    return SimpleNamespace(
        total=memory.swap_total_bytes,
        used=memory.swap_used_bytes,
        free=memory.swap_free_bytes,
        percent=memory.swap_percent,
        provenance=memory.provenance,
        observation_available=memory.available,
        sin=0,
        sout=0,
    )


def cpu_percent(
    interval: float | None = None,
    percpu: bool = False,
) -> Any:
    del interval
    compute = get_resource_observer().compute()
    value = float(compute.cpu_percent) if compute.available else 100.0
    if percpu:
        return [value for _index in range(max(1, int(compute.cpu_count)))]
    return value


def cpu_count(logical: bool = True) -> int:
    del logical
    return int(get_resource_observer().compute().cpu_count)


def cpu_times() -> Any:
    compute = get_resource_observer().compute()
    return SimpleNamespace(
        user=compute.cpu_user_seconds,
        system=compute.cpu_system_seconds,
        idle=compute.cpu_idle_seconds,
        provenance=compute.provenance,
        observation_available=compute.available,
    )


def cpu_freq(*_args: Any, **_kwargs: Any) -> Any:
    return _host_counter("cpu_freq", ("current", "min", "max"))


def cpu_stats() -> Any:
    return _host_counter("cpu_stats", ("ctx_switches", "interrupts", "soft_interrupts", "syscalls"))


def getloadavg() -> tuple[float, float, float]:
    compute = get_resource_observer().compute()
    return (compute.load_1m, compute.load_5m, compute.load_15m)


def boot_time() -> float:
    return float(get_resource_observer().compute().boot_time)


def disk_io_counters(*_args: Any, **_kwargs: Any) -> Any:
    return _host_counter(
        "disk_io_counters",
        (
            "read_count",
            "write_count",
            "read_bytes",
            "write_bytes",
            "read_time",
            "write_time",
            "read_merged_count",
            "write_merged_count",
            "busy_time",
        ),
    )


def net_io_counters(*_args: Any, **_kwargs: Any) -> Any:
    return _host_counter(
        "net_io_counters",
        (
            "bytes_sent",
            "bytes_recv",
            "packets_sent",
            "packets_recv",
            "errin",
            "errout",
            "dropin",
            "dropout",
        ),
    )


def pids() -> list[int]:
    observation = get_resource_observer().process_ids()
    if not observation.available:
        raise _psutil.AccessDenied(pid=os.getpid(), msg=observation.error)
    return list(observation.pids)


def pid_exists(pid: int) -> bool:
    return get_resource_observer().process(int(pid)) is not None


def process_iter(*_args: Any, **kwargs: Any) -> Iterator[Process]:
    attrs = kwargs.get("attrs")
    for pid in pids():
        process = Process(pid)
        if attrs is not None:
            process.info = process.as_dict(attrs=list(attrs))
        yield process


def _converted_connections(
    table: ConnectionTableObservation,
    *,
    owner_pid: int,
) -> list[Any]:
    if not table.available:
        raise _psutil.AccessDenied(pid=owner_pid, msg=table.error)
    converted: list[Any] = []
    for connection in table.connections:
        local = (
            SimpleNamespace(ip=connection.local_host, port=connection.local_port)
            if connection.local_host or connection.local_port
            else None
        )
        remote = (
            SimpleNamespace(ip=connection.remote_host, port=connection.remote_port)
            if connection.remote_host or connection.remote_port
            else None
        )
        converted.append(
            SimpleNamespace(
                fd=connection.fd,
                family=connection.family,
                type=connection.socket_type,
                laddr=local,
                raddr=remote,
                status=connection.status,
                pid=connection.pid,
                provenance=connection.provenance,
            )
        )
    return converted


def net_connections(kind: str = "inet") -> list[Any]:
    table = get_resource_observer().connection_table(kind=kind)
    return _converted_connections(table, owner_pid=os.getpid())


def net_if_addrs() -> dict[str, Any]:
    observer = get_resource_observer()
    if not observer.provenance.host_observed:
        return _AttributedMapping(
            provenance=observer.provenance,
            observation_available=True,
        )
    try:
        raw = _psutil.net_if_addrs()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, _psutil.Error):
        return _AttributedMapping(
            provenance=observer.provenance,
            observation_available=False,
        )
    return _AttributedMapping(
        raw,
        provenance=observer.provenance,
        observation_available=True,
    )


class Process:
    """Observer-backed process reads with explicit native action methods."""

    def __init__(self, pid: int | None = None) -> None:
        self.pid = os.getpid() if pid is None else int(pid)
        self.info: dict[str, Any] = {}

    def _observation(self) -> ProcessObservation:
        observed = get_resource_observer().process(self.pid)
        if observed is None:
            raise _psutil.NoSuchProcess(self.pid)
        return observed

    def _native(self) -> _psutil.Process:
        return _psutil.Process(self.pid)

    def oneshot(self) -> contextlib.AbstractContextManager[Process]:
        return contextlib.nullcontext(self)

    def memory_info(self) -> Any:
        observed = self._observation()
        return SimpleNamespace(rss=observed.rss_bytes, vms=observed.rss_bytes)

    def memory_full_info(self) -> Any:
        return self.memory_info()

    def as_dict(self, attrs: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
        requested = tuple(
            attrs
            or (
                "pid",
                "ppid",
                "name",
                "cmdline",
                "create_time",
                "status",
                "memory_info",
                "memory_percent",
                "cpu_percent",
                "num_threads",
            )
        )
        values: dict[str, Any] = {}
        for name in requested:
            if name == "pid":
                values[name] = self.pid
                continue
            reader = getattr(self, name)
            values[name] = reader() if callable(reader) else reader
        return values

    def memory_percent(self) -> float:
        return self._observation().memory_percent

    def cpu_percent(self, interval: float | None = None) -> float:
        del interval
        return self._observation().cpu_percent

    def cpu_times(self) -> Any:
        observed = self._observation()
        return SimpleNamespace(
            user=observed.cpu_user_seconds,
            system=observed.cpu_system_seconds,
        )

    def create_time(self) -> float:
        return self._observation().create_time

    def status(self) -> str:
        return self._observation().status

    def name(self) -> str:
        return self._observation().name

    def cmdline(self) -> list[str]:
        return list(self._observation().cmdline)

    def cwd(self) -> str:
        return self._observation().cwd

    def exe(self) -> str:
        return self._observation().exe

    def username(self) -> str:
        return self._observation().username

    def ppid(self) -> int:
        return self._observation().ppid

    def num_threads(self) -> int:
        return self._observation().num_threads

    def num_fds(self) -> int:
        return self._observation().num_fds

    def is_running(self) -> bool:
        observed = get_resource_observer().process(self.pid)
        return observed is not None and observed.status not in {"dead", "zombie"}

    def children(self, recursive: bool = False) -> list[Process]:
        table = get_resource_observer().process_tree(self.pid, recursive=recursive)
        if not table.available:
            raise _psutil.AccessDenied(pid=self.pid, msg=table.error)
        observed = [item for item in table.processes if item.pid != self.pid]
        return [Process(item.pid) for item in observed]

    def parent(self) -> Process | None:
        observed = self._observation()
        return None if observed.ppid <= 0 else Process(observed.ppid)

    def parents(self) -> list[Process]:
        return [Process(pid) for pid in self._observation().ancestor_pids]

    def open_files(self) -> list[Any]:
        table = get_resource_observer().open_file_table(pid=self.pid)
        if not table.available:
            raise _psutil.AccessDenied(pid=self.pid, msg=table.error)
        return [SimpleNamespace(path=path, fd=-1) for path in table.paths]

    def net_connections(self, kind: str = "inet") -> list[Any]:
        table = get_resource_observer().connection_table(kind=kind, pid=self.pid)
        return _converted_connections(table, owner_pid=self.pid)

    connections = net_connections

    def uids(self) -> Any:
        return self._native().uids()

    def terminate(self) -> None:
        self._native().terminate()

    def kill(self) -> None:
        self._native().kill()

    def wait(self, timeout: float | None = None) -> int | None:
        result = self._native().wait(timeout=timeout)
        return None if result is None else int(result)


def wait_procs(
    processes: list[Process],
    timeout: float | None = None,
    callback: Callable[[Process], Any] | None = None,
) -> tuple[list[Process], list[Process]]:
    pairs = [(process, process._native()) for process in processes]
    by_pid = {native.pid: wrapped for wrapped, native in pairs}
    gone, alive = _psutil.wait_procs(
        [native for _wrapped, native in pairs],
        timeout=timeout,
        callback=None,
    )
    wrapped_gone = [by_pid[process.pid] for process in gone]
    wrapped_alive = [by_pid[process.pid] for process in alive]
    if callback is not None:
        for process in wrapped_gone:
            callback(process)
    return wrapped_gone, wrapped_alive


def sensors_temperatures(*_args: Any, **_kwargs: Any) -> dict[str, list[Any]]:
    thermal = get_resource_observer().thermal()
    # Compatibility consumers expect Celsius. These representative values keep
    # their existing threshold arithmetic while the canonical level remains the
    # source of truth.
    current_c = {0: 45.0, 1: 70.0, 2: 82.0, 3: 95.0}.get(int(thermal.level), 95.0)
    if not thermal.available:
        current_c = 100.0
    return {
        "canonical": [
            SimpleNamespace(
                current=current_c,
                label=thermal.provider,
                provenance=thermal.provenance,
            )
        ]
    }


def sensors_battery() -> Any:
    power = get_resource_observer().power()
    return SimpleNamespace(
        percent=power.battery_percent,
        power_plugged=power.plugged,
        secsleft=power.seconds_left,
        provenance=power.provenance,
        observation_available=power.available,
        error=power.error,
    )


def __getattr__(name: str) -> Any:
    if name in _RESOURCE_READ_APIS:
        raise AttributeError(f"resource read API {name!r} has no observer-backed implementation")
    return getattr(_psutil, name)


__all__ = [
    "cpu_count",
    "cpu_freq",
    "cpu_percent",
    "cpu_stats",
    "cpu_times",
    "boot_time",
    "disk_io_counters",
    "disk_usage",
    "getloadavg",
    "net_connections",
    "net_if_addrs",
    "net_io_counters",
    "pid_exists",
    "pids",
    "Process",
    "process_iter",
    "sensors_temperatures",
    "sensors_battery",
    "swap_memory",
    "virtual_memory",
    "wait_procs",
]
