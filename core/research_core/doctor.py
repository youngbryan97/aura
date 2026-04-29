"""F5 doctor-bundle integration for the research core.

The diagnostics bundle (``aura doctor --bundle``) calls
``collect_research_core_status()`` to grab a compact JSON snapshot of
the research core for the tarball.  The collector is fail-safe: it
returns a structured error blob rather than raising when the core
isn't registered or is in a degraded state.
"""
from __future__ import annotations

from typing import Any, Dict


def collect_research_core_status() -> Dict[str, Any]:
    try:
        from core.container import ServiceContainer

        core = ServiceContainer.get("research_core", default=None)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    if core is None:
        return {"available": False, "reason": "research_core not registered"}

    try:
        return {
            "available": True,
            "status": core.status(),
            "recent_cycles": core.cycle_history()[-5:],
        }
    except Exception as exc:  # noqa: BLE001 - last-resort safety
        return {"available": True, "error": f"status_failed: {exc}"}
