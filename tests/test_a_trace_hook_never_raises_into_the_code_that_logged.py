"""Her trace hook reads the process inside logging, so nothing it meets may raise into the code that logged.

Q09, 13 November 2026: `test_orphan_reclaimed_before_memory_admission` passed alone and failed in a batch. Once an
earlier test had installed the trace hook, the worker spawn's own log line read the process through psutil, which the
test had stood in for with a process that has no memory_info, and AttributeError came out of the spawn.
"""
from __future__ import annotations

import os

from core.resilience import omni_tracer


class _AProcessWithoutMemory:
    pid = 999999


def test_a_stand_in_process_leaves_the_trace_its_pid(monkeypatch):
    monkeypatch.setattr(omni_tracer.psutil, "Process", lambda pid: _AProcessWithoutMemory())
    assert omni_tracer._get_system_context() == {"pid": os.getpid()}
