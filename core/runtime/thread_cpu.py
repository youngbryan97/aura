"""The CPU time of another thread, read from outside it.

``time.thread_time()`` answers only for the calling thread. A watchdog that
wants to know whether the event loop is stuck or merely starved needs the
loop thread's clock from its own thread, while the loop is not running. On
macOS the kernel keeps it per thread and ``thread_info`` hands it over; on
Linux it is in ``/proc/self/task/<tid>/stat``. Where neither applies the
reading is None and callers fall back to the wall clock alone.

Why it matters: a thread that is blocked accrues no CPU at all. A thread the
host is not scheduling accrues a little, in bursts. A thread doing work
accrues about as much as the wall clock advances. Three different conditions,
one reading, and the wall clock alone cannot tell them apart.

Proved on this host 2026-09-16 at load 30: a busy thread read 0.55s of CPU
over 1.0s of wall from another thread, a sleeping one 0.001s, and a thread
reading itself matched ``time.thread_time()`` to a hundred microseconds.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

__all__ = ["every_thread_cpu", "thread_cpu_seconds", "thread_cpu_share", "who_ran_meanwhile"]

_libc: ctypes.CDLL | None = None
_MACH_UNAVAILABLE = False


class _TimeValue(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_int), ("microseconds", ctypes.c_int)]


class _ThreadBasicInfo(ctypes.Structure):
    _fields_ = [
        ("user_time", _TimeValue),
        ("system_time", _TimeValue),
        ("cpu_usage", ctypes.c_int),
        ("policy", ctypes.c_int),
        ("run_state", ctypes.c_int),
        ("flags", ctypes.c_int),
        ("suspend_count", ctypes.c_int),
        ("sleep_time", ctypes.c_int),
    ]


_THREAD_BASIC_INFO = 3
_THREAD_BASIC_INFO_COUNT = ctypes.sizeof(_ThreadBasicInfo) // ctypes.sizeof(ctypes.c_int)


def _mach_libc() -> ctypes.CDLL | None:
    global _libc, _MACH_UNAVAILABLE
    if _libc is not None or _MACH_UNAVAILABLE:
        return _libc
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.pthread_mach_thread_np.restype = ctypes.c_uint
        libc.pthread_mach_thread_np.argtypes = [ctypes.c_void_p]
        libc.thread_info.restype = ctypes.c_int
        libc.thread_info.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
    except (AttributeError, OSError, TypeError) as exc:
        # Every per-thread CPU reading is unavailable from here on, once, for
        # the life of the process — which is what lockdep uses to tell a
        # section that is working from one that is blocked.
        logger.warning(
            "mach thread_info is unavailable (%s: %s); per-thread CPU will "
            "not be measured",
            type(exc).__name__,
            exc,
        )
        _MACH_UNAVAILABLE = True
        return None
    _libc = libc
    return libc


def _mach_thread_cpu_seconds(ident: int) -> float | None:
    libc = _mach_libc()
    if libc is None:
        return None
    # CPython's thread ident on POSIX is pthread_self(), which is what the
    # mach port lookup wants.
    port = libc.pthread_mach_thread_np(ctypes.c_void_p(int(ident)))
    if not port:
        return None
    info = _ThreadBasicInfo()
    count = ctypes.c_uint(_THREAD_BASIC_INFO_COUNT)
    if libc.thread_info(port, _THREAD_BASIC_INFO, ctypes.byref(info), ctypes.byref(count)) != 0:
        return None
    return (
        info.user_time.seconds
        + info.user_time.microseconds / 1e6
        + info.system_time.seconds
        + info.system_time.microseconds / 1e6
    )


def _proc_thread_cpu_seconds(native_id: int) -> float | None:
    try:
        with open(f"/proc/self/task/{int(native_id)}/stat", encoding="ascii") as handle:
            stat = handle.read()
        # The command name is in parentheses and may hold spaces; fields
        # start after the closing one. utime and stime are fields 14 and 15.
        fields = stat[stat.rindex(")") + 2 :].split()
        ticks = float(fields[11]) + float(fields[12])
        return ticks / float(os.sysconf("SC_CLK_TCK"))
    # not a failure: no /proc means this reader has nothing to offer, and the
    # caller reports per-thread CPU as unmeasured.
    except (OSError, ValueError, IndexError):
        return None


def thread_cpu_seconds(ident: int, *, native_id: int | None = None) -> float | None:
    """CPU seconds (user + system) the thread with this ident has used, or None."""
    if sys.platform == "darwin":
        return _mach_thread_cpu_seconds(ident)
    if sys.platform.startswith("linux"):
        if native_id is None:
            for thread in threading.enumerate():
                if thread.ident == ident:
                    native_id = thread.native_id
                    break
        if native_id is None:
            return None
        return _proc_thread_cpu_seconds(native_id)
    return None


def thread_cpu_share(
    cpu_before: float | None, cpu_after: float | None, wall_s: float
) -> float | None:
    """The fraction of ``wall_s`` a thread spent on a CPU, or None if unread."""
    if cpu_before is None or cpu_after is None or wall_s <= 0.0:
        return None
    return max(0.0, cpu_after - cpu_before) / wall_s


def every_thread_cpu() -> dict[int, float]:
    """CPU seconds of every live thread, by ident. A thread the kernel will not read is left out."""
    seen: dict[int, float] = {}
    for thread in threading.enumerate():
        if thread.ident is None:
            continue
        cpu = thread_cpu_seconds(thread.ident, native_id=thread.native_id)
        if cpu is not None:
            seen[thread.ident] = cpu
    return seen


def who_ran_meanwhile(before: dict[int, float], wall_s: float, *, leave_out: int | None = None, most: int = 3) -> str:
    """The threads that used the most CPU since ``before``, each with its share of a core and where it is now.

    The loop's starvation detectors said only "its own process held it": on 14
    November 2026 her loop had 10.49 s of lag on 7% of a core while her other
    threads used 1.13 cores, and nothing said which. By the time the lag is
    read the stall is over, so a stack taken then may be the wrong one; the
    CPU each thread spent over the window is not.
    """
    if wall_s <= 0.0:
        return ""
    after = every_thread_cpu()
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    frames = sys._current_frames()
    # A thread started during the window spent all it shows inside it.
    spent = sorted(
        ((after[ident] - before.get(ident, 0.0), ident) for ident in after if ident != leave_out),
        reverse=True,
    )
    said = []
    for cpu, ident in spent[:most]:
        if cpu <= 0.0:
            break
        said.append(f"{names.get(ident, ident)} {cpu / wall_s:.0%} of a core{_where(frames.get(ident))}")
    return "Ran meanwhile: " + "; ".join(said) + "." if said else ""


def _where(frame: object) -> str:
    """Where a thread is: the innermost frame, and the innermost in this repository when that is not it."""
    if frame is None:
        return ""
    innermost = frame
    ours = None
    while frame is not None and ours is None:
        if f"{os.sep}core{os.sep}" in frame.f_code.co_filename or f"{os.sep}interface{os.sep}" in frame.f_code.co_filename:
            ours = frame
        frame = frame.f_back

    def said(at: object) -> str:
        return f"{os.path.basename(at.f_code.co_filename)}:{at.f_lineno} {at.f_code.co_name}"

    if ours is None or ours is innermost:
        return f" at {said(innermost)}"
    return f" at {said(innermost)}, from {said(ours)}"
