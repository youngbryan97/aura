"""Aura runtime package.

Lazy attribute access so importing a sibling like
``core.runtime.atomic_writer`` doesn't drag in ``core_runtime`` (which pulls
``core.config`` → cycle when ``core.config`` is what triggered the import).
``from core.runtime import CoreRuntime`` still works via PEP 562.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core_runtime import CoreRuntime
    from .loop_guard import LoopLagMonitor

__all__ = ["CoreRuntime", "LoopLagMonitor"]


def __getattr__(name: str):
    if name == "CoreRuntime":
        from .core_runtime import CoreRuntime
        return CoreRuntime
    if name == "LoopLagMonitor":
        from .loop_guard import LoopLagMonitor
        return LoopLagMonitor
    raise AttributeError(f"module 'core.runtime' has no attribute {name!r}")
