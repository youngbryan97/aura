"""Bounded platform adapter for current per-process physical footprint.

macOS ``phys_footprint`` includes compressed and IOKit-mapped memory that RSS
can omit. All guards use this one ABI definition so a Darwin layout change
cannot make watchdogs disagree or write past a duplicated ctypes structure.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)


class DarwinRUsageInfoV4(ctypes.Structure):
    """Oversized, forward-tolerant representation of ``rusage_info_v4``."""

    _fields_ = [
        ("ri_uuid", ctypes.c_ubyte * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
        ("ri_flags", ctypes.c_uint64),
        # rusage_info_v4 is currently 304 bytes. Spare capacity prevents a
        # future flavor/layout extension from overrunning this allocation.
        ("_ri_spare", ctypes.c_uint64 * 16),
    ]


_RUSAGE_INFO_V4 = 4
_LIBPROC: Any | None = None
_LIBPROC_UNAVAILABLE = False
_LIBPROC_LOCK = threading.Lock()


def current_darwin_footprint_bytes(info: DarwinRUsageInfoV4) -> int:
    """Return current footprint, never the process-lifetime high-water mark."""

    current = int(getattr(info, "ri_phys_footprint", 0) or 0)
    if current > 0:
        return current
    return int(getattr(info, "ri_resident_size", 0) or 0)


def _load_libproc() -> Any | None:
    global _LIBPROC, _LIBPROC_UNAVAILABLE
    if sys.platform != "darwin" or _LIBPROC_UNAVAILABLE:
        return None
    with _LIBPROC_LOCK:
        if _LIBPROC is not None:
            return _LIBPROC
        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib")
            library.proc_pid_rusage.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            library.proc_pid_rusage.restype = ctypes.c_int
            _LIBPROC = library
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            # Every footprint reading falls back to 0 from here, once, for
            # the life of the process.
            logger.warning(
                "libproc is unavailable (%s: %s); darwin footprints will read 0",
                type(exc).__name__,
                exc,
            )
            _LIBPROC_UNAVAILABLE = True
            return None
    return _LIBPROC


def darwin_phys_footprint_bytes(pid: int) -> int:
    """Return current Darwin physical footprint, or zero when unavailable."""

    global _LIBPROC_UNAVAILABLE
    if int(pid) <= 0:
        return 0
    library = _load_libproc()
    if library is None:
        return 0
    try:
        info = DarwinRUsageInfoV4()
        rc = library.proc_pid_rusage(
            int(pid),
            _RUSAGE_INFO_V4,
            ctypes.byref(info),
        )
        return 0 if rc != 0 else current_darwin_footprint_bytes(info)
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        ctypes.ArgumentError,
    ) as exc:
        logger.warning(
            "libproc rusage for pid %s failed (%s: %s); footprints will read 0",
            pid,
            type(exc).__name__,
            exc,
        )
        _LIBPROC_UNAVAILABLE = True
        return 0


__all__ = [
    "DarwinRUsageInfoV4",
    "current_darwin_footprint_bytes",
    "darwin_phys_footprint_bytes",
]
