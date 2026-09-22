"""core/utils/executor.py
Global executors for Aura to manage GIL contention and blocking ops.
"""
import asyncio
import concurrent.futures
import multiprocessing
import threading
from collections.abc import Callable
from functools import partial
from typing import Any

_executor_lock = threading.RLock()
_cpu_executor: concurrent.futures.ProcessPoolExecutor | None = None
_io_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _shutdown_requested() -> bool:
    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        return is_shutdown_requested()
    # not a failure: the module is optional here, and its absence is the answer.
    except ImportError:
        return False


def _register_executor(
    executor: concurrent.futures.Executor,
    *,
    name: str,
) -> None:
    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        get_runtime_hygiene().register_shutdown_resource(
            executor,
            kind="executor",
            name=name,
            source="core.utils.executor",
            closer=partial(executor.shutdown, wait=False, cancel_futures=True),
            timeout_s=1.0,
            required=True,
        )
    # not a failure: the module is optional here, and its absence is the answer.
    except ImportError:
        return


def get_cpu_executor() -> concurrent.futures.ProcessPoolExecutor:
    global _cpu_executor
    with _executor_lock:
        if _shutdown_requested():
            raise RuntimeError("runtime_shutdown")
        if _cpu_executor is None:
            _cpu_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(multiprocessing.cpu_count(), 2),
                mp_context=multiprocessing.get_context("spawn"),
            )
        executor = _cpu_executor
        _register_executor(executor, name="legacy_cpu_process_pool")
        return executor


def get_io_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _io_executor
    with _executor_lock:
        if _shutdown_requested():
            raise RuntimeError("runtime_shutdown")
        if _io_executor is None:
            _io_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(multiprocessing.cpu_count() * 4, 32),
                thread_name_prefix="Aura_IO",
            )
        executor = _io_executor
        _register_executor(executor, name="legacy_io_thread_pool")
        return executor

async def run_in_process(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run a CPU-bound function in a separate process."""
    loop = asyncio.get_running_loop()
    if kwargs:
        from functools import partial
        func = partial(func, **kwargs)
    try:
        return await loop.run_in_executor(get_cpu_executor(), func, *args)
    except RuntimeError as e:
        if "shutdown" in str(e).lower():
            return None
        raise

async def run_in_thread(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run an I/O-bound function in a separate thread."""
    loop = asyncio.get_running_loop()
    if kwargs:
        from functools import partial
        func = partial(func, **kwargs)
    try:
        return await loop.run_in_executor(get_io_executor(), func, *args)
    except RuntimeError as e:
        if "shutdown" in str(e).lower():
            return None
        raise


def shutdown_executors() -> None:
    """Cleanup on exit."""
    global _cpu_executor, _io_executor
    with _executor_lock:
        cpu_executor = _cpu_executor
        io_executor = _io_executor
        _cpu_executor = None
        _io_executor = None
    if cpu_executor is not None:
        cpu_executor.shutdown(wait=False, cancel_futures=True)
    if io_executor is not None:
        io_executor.shutdown(wait=False, cancel_futures=True)
