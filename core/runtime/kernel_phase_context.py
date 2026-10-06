"""Whether the running code is a kernel phase, holding the kernel's state lock.

A kernel tick runs its phases one at a time under the state lock, and every
tick behind it waits for that lock. A phase that asks for a background
generation while another generation holds the model's gate would wait with
the lock held. LIVE 2026-10-06 a background UnitaryResponsePhase waited 5.16 s
behind the narrative stream's generation, the router's whole background
wait, and then deferred anyway; a turn arriving in those seconds waited too.

The kernel marks the phase it is running; the router reads the mark and
lets a background request from inside a phase defer at once instead.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["phase_holding_the_kernel", "running_a_kernel_phase"]

_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_kernel_phase_holding_the_lock", default=""
)


@contextmanager
def running_a_kernel_phase(phase_name: str) -> Iterator[None]:
    """Mark the code inside as the kernel's phase ``phase_name``."""
    token = _PHASE.set(str(phase_name or "phase"))
    try:
        yield
    finally:
        _PHASE.reset(token)


def phase_holding_the_kernel() -> str:
    """The kernel phase this code runs in, or "" outside one."""
    return _PHASE.get()
