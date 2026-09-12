#!/usr/bin/env python3
"""How many cores a tool may use, as the runtime observes them.

Two gates size a process pool from this. They read it through the resource
observer rather than `os.cpu_count` for the reason every other reading does:
a run under a simulated observer has to see the machine it was told it was
on, and the ownership audit counts a direct read as a finding.

Getting that wrong is silent and expensive. A tool run as `tools/x.py` has
only `tools/` on `sys.path`, so `import core...` raises ModuleNotFoundError,
the helper falls back to one core, and the gate runs serially while every
measurement of it says it is parallel — 21s of work taking 80s, which is how
the enterprise gate came to time out against its own 60s limit. So the repo
root goes on the path here, once, where the reason for it is written down.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

__all__ = ["cores_available"]


def cores_available(ceiling: int = 8) -> int:
    """Cores this run may use, at most ``ceiling``. One if the runtime is absent.

    One core is the safe direction when the observer cannot be reached: a
    serial scan is slow, a pool sized from a machine that is not there is
    wrong.
    """
    try:
        from core.runtime.resource_psutil import cpu_count
    except ImportError:
        return 1
    try:
        observed = int(cpu_count(logical=True))
    except (OSError, RuntimeError, TypeError, ValueError):
        return 1
    return max(1, min(observed, max(1, ceiling)))
