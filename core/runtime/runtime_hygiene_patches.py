"""Standing between the runtime and the four things that leak.

A thread, a subprocess, a multiprocessing child and an event loop are all
created by functions this process does not own. Wrapping them is how anything
gets counted at all, and unwrapping them on the way out is how a test that
imported this module does not leave the wrappers behind for the next one.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import subprocess
import threading

from core.runtime.errors import record_degradation
from core.utils.task_tracker import (
    shutdown_resource_creation_allowed,
)


class _WatchesWhatTheRuntimeCreates:
    """Lifted whole from RuntimeHygieneManager; see runtime_hygiene.py."""

    def _patch_asyncio_new_event_loop(self) -> None:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .runtime_hygiene import (
            logger,
        )

        if self._original_new_event_loop is not None:
            return

        self._original_new_event_loop = asyncio.new_event_loop
        tracker = self._task_tracker

        def _patched_new_event_loop():
            loop = self._original_new_event_loop()
            try:
                tracker.install_loop_hygiene(loop)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('runtime_hygiene', exc)
                logger.debug("RuntimeHygiene: failed to install task factory on new loop: %s", exc)
            return loop

        asyncio.new_event_loop = _patched_new_event_loop

    def _patch_threading(self) -> None:
        if self._original_thread_start is not None:
            return

        self._original_thread_start = threading.Thread.start
        manager = self

        def _patched_start(thread: threading.Thread, *args, **kwargs):
            executor_teardown = manager._is_executor_shutdown_thread(thread)
            cleanup_critical = (
                shutdown_resource_creation_allowed() or executor_teardown
            )
            if not cleanup_critical and manager._shutdown_blocks_resource_start(
                operation=f"thread.start:{thread.name}", resource_kind="thread"
            ):
                manager._run_thread_suppression_cleanup(thread)
                raise RuntimeError("runtime_shutdown")
            if executor_teardown and manager._runtime_shutdown_latched():
                manager._record_creation_boundary(
                    operation=f"thread.start:{thread.name}",
                    resource_kind="thread",
                    outcome="allowed_teardown",
                    detail="asyncio_default_executor_shutdown",
                )
            thread._aura_shutdown_critical = cleanup_critical
            manager._register_thread(thread, source="thread.start")
            result = manager._original_thread_start(thread, *args, **kwargs)
            if manager._runtime_shutdown_latched() and not cleanup_critical:
                manager._record_creation_boundary(
                    operation=f"thread.start:{thread.name}",
                    resource_kind="thread",
                    outcome="crossed",
                    detail=f"ident={thread.ident}",
                )
                if not thread.is_alive():
                    manager._record_creation_boundary(
                        operation=f"thread.start:{thread.name}",
                        resource_kind="thread",
                        outcome="reaped",
                        detail="target_exited_at_shutdown_boundary",
                    )
            return result

        threading.Thread.start = _patched_start

    def _patch_subprocess(self) -> None:
        if self._original_popen_init is not None:
            return

        self._original_popen_init = subprocess.Popen.__init__
        manager = self

        def _patched_init(proc_self, *args, **kwargs):
            cleanup_critical = shutdown_resource_creation_allowed()
            command = kwargs.get("args") or (args[0] if args else "unknown")
            if manager._shutdown_blocks_resource_start(
                operation=f"subprocess.Popen:{str(command)[:160]}",
                resource_kind="subprocess",
            ):
                proc_self._child_created = False
                raise RuntimeError("runtime_shutdown")
            manager._original_popen_init(proc_self, *args, **kwargs)
            manager._register_subprocess(proc_self, args=args, kwargs=kwargs)
            if manager._runtime_shutdown_latched() and not cleanup_critical:
                operation = f"subprocess.Popen:{str(command)[:160]}"
                manager._record_creation_boundary(
                    operation=operation,
                    resource_kind="subprocess",
                    outcome="crossed",
                    detail=f"pid={getattr(proc_self, 'pid', None)}",
                )
                reaped = manager._reap_crossed_subprocess(proc_self)
                manager._record_creation_boundary(
                    operation=operation,
                    resource_kind="subprocess",
                    outcome="reaped" if reaped else "survived",
                    detail=f"pid={getattr(proc_self, 'pid', None)}",
                )
                raise RuntimeError("runtime_shutdown_after_subprocess_start")

        subprocess.Popen.__init__ = _patched_init

    def _patch_multiprocessing(self) -> None:
        if self._original_mp_start is not None:
            return

        self._original_mp_start = mp.process.BaseProcess.start
        manager = self

        def _patched_start(proc_self, *args, **kwargs):
            cleanup_critical = shutdown_resource_creation_allowed()
            if manager._shutdown_blocks_resource_start(
                operation=f"multiprocessing.start:{getattr(proc_self, 'name', 'unknown')}",
                resource_kind="multiprocessing",
            ):
                raise RuntimeError("runtime_shutdown")
            result = manager._original_mp_start(proc_self, *args, **kwargs)
            manager._register_multiprocessing_process(proc_self)
            if manager._runtime_shutdown_latched() and not cleanup_critical:
                operation = (
                    f"multiprocessing.start:{getattr(proc_self, 'name', 'unknown')}"
                )
                manager._record_creation_boundary(
                    operation=operation,
                    resource_kind="multiprocessing",
                    outcome="crossed",
                    detail=f"pid={getattr(proc_self, 'pid', None)}",
                )
                reaped = manager._reap_crossed_multiprocessing(proc_self)
                manager._record_creation_boundary(
                    operation=operation,
                    resource_kind="multiprocessing",
                    outcome="reaped" if reaped else "survived",
                    detail=f"pid={getattr(proc_self, 'pid', None)}",
                )
                raise RuntimeError("runtime_shutdown_after_multiprocessing_start")
            return result

        mp.process.BaseProcess.start = _patched_start

    @staticmethod
    def _is_executor_shutdown_thread(thread: threading.Thread) -> bool:
        target = getattr(thread, "_target", None)
        module = str(getattr(target, "__module__", "") or "")
        qualname = str(getattr(target, "__qualname__", "") or "")
        return module == "asyncio.base_events" and qualname.endswith(
            "BaseEventLoop._do_shutdown"
        )

    def _restore_patches(self) -> None:
        if self._original_thread_start is not None:
            threading.Thread.start = self._original_thread_start
            self._original_thread_start = None
        if self._original_popen_init is not None:
            subprocess.Popen.__init__ = self._original_popen_init
            self._original_popen_init = None
        if self._original_mp_start is not None:
            mp.process.BaseProcess.start = self._original_mp_start
            self._original_mp_start = None
        if self._original_new_event_loop is not None:
            asyncio.new_event_loop = self._original_new_event_loop
            self._original_new_event_loop = None
