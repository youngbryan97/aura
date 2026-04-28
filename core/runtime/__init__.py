"""Runtime package exports.

Keep this module lazy.  Core boot paths import small helpers such as
``core.runtime.atomic_writer`` while ``core.container`` is still defining
``ServiceContainer``; importing ``CoreRuntime`` eagerly from here re-enters the
container and creates a partial-initialization circular import.
"""

__all__ = ["CoreRuntime", "LoopLagMonitor"]


def __getattr__(name):
    if name == "CoreRuntime":
        from .core_runtime import CoreRuntime

        return CoreRuntime
    if name == "LoopLagMonitor":
        from .loop_guard import LoopLagMonitor

        return LoopLagMonitor
    raise AttributeError(f"module 'core.runtime' has no attribute {name!r}")
