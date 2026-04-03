"""core/utils/executor.py
Global executors for Aura to manage GIL contention and blocking ops.
"""
import asyncio
import concurrent.futures
import multiprocessing
import os
from typing import Any, Callable

_cpu_executor = None
_io_executor = None

def get_cpu_executor():
    global _cpu_executor
    if _cpu_executor is None:
        _cpu_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(multiprocessing.cpu_count(), 2),
            mp_context=multiprocessing.get_context('spawn')
        )
    return _cpu_executor

def get_io_executor():
    global _io_executor
    if _io_executor is None:
        _io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(multiprocessing.cpu_count() * 4, 32)
        )
    return _io_executor

async def run_in_process(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a CPU-bound function in a separate process."""
    loop = asyncio.get_running_loop()
    if kwargs:
        from functools import partial
        func = partial(func, **kwargs)
    try:
        return await loop.run_in_executor(get_cpu_executor(), func, *args)
    except RuntimeError as e:
        if "shutdown" in str(e).lower(): return None
        raise

async def run_in_thread(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run an I/O-bound function in a separate thread."""
    loop = asyncio.get_running_loop()
    if kwargs:
        from functools import partial
        func = partial(func, **kwargs)
    try:
        return await loop.run_in_executor(get_io_executor(), func, *args)
    except RuntimeError as e:
        if "shutdown" in str(e).lower(): return None
        raise

def shutdown_executors():
    """Cleanup on exit."""
    global _cpu_executor, _io_executor
    if _cpu_executor:
        _cpu_executor.shutdown(wait=False)
        _cpu_executor = None
    if _io_executor:
        _io_executor.shutdown(wait=False)
        _io_executor = None