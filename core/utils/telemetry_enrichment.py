"""core/utils/telemetry_enrichment.py
─────────────────────────────────────
Single source of truth for telemetry payload enrichment.
Eliminates the previous 3x duplication across server.py broadcast_telemetry(),
_event_bridge(), and orchestrator _publish_telemetry().
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import time
from typing import Any, Dict

logger = logging.getLogger("Aura.Telemetry")

# Pre-import psutil at module level (was previously imported inside loops)
try:
    import psutil
    # Warm the non-blocking percent counter
    psutil.cpu_percent(interval=None)
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _normalize_percentish(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1.0:
        number *= 100.0
    return max(0.0, min(100.0, number))


def enrich_telemetry(data: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a telemetry payload with hardware metrics, liquid state gauges,
    and LLM tier information.  Mutates and returns `data` for convenience.

    This is the ONLY place these enrichments should happen.
    """
    # Lazy import to avoid circular dependency at module load
    from core.container import ServiceContainer

    data.setdefault("type", "telemetry")
    data.setdefault("timestamp", time.time())

    # ── Hardware Metrics ──────────────────────────────────────────────
    if _HAS_PSUTIL and "cpu_usage" not in data:
        try:
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            data["cpu_usage"] = sum(per_cpu) / len(per_cpu) if per_cpu else 0
            data["ram_usage"] = psutil.virtual_memory().percent

            # Apple Silicon P-Core / E-Core distinction
            if len(per_cpu) >= 16:       # M5/M4 Pro/Max/Ultra (4E, 12+P)
                data["p_core_usage"] = sum(per_cpu[4:]) / (len(per_cpu) - 4)
                data["e_core_usage"] = sum(per_cpu[:4]) / 4
            elif len(per_cpu) >= 10:     # Earlier Pro/Max Apple Silicon layouts
                data["p_core_usage"] = sum(per_cpu[2:]) / (len(per_cpu) - 2)
                data["e_core_usage"] = sum(per_cpu[:2]) / 2
            elif len(per_cpu) == 8:      # Base Apple Silicon layouts
                data["p_core_usage"] = sum(per_cpu[4:]) / 4
                data["e_core_usage"] = sum(per_cpu[:4]) / 4
            else:
                data["p_core_usage"] = data["cpu_usage"]
        except Exception as _exc:
            record_degradation('telemetry_enrichment', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    # ── Liquid State Gauges ───────────────────────────────────────────
    if "energy" not in data or "curiosity" not in data:
        try:
            ls = ServiceContainer.get("liquid_state", default=None)
            if ls and hasattr(ls, "get_status"):
                ls_vals = ls.get_status()
                energy = _normalize_percentish(ls_vals.get("energy", 0))
                curiosity = _normalize_percentish(ls_vals.get("curiosity", 0))
                frustration = _normalize_percentish(ls_vals.get("frustration", 0))
                confidence = _normalize_percentish(ls_vals.get("confidence", 0))
                if energy is not None:
                    data.setdefault("energy", round(energy, 1))
                if curiosity is not None:
                    data.setdefault("curiosity", round(curiosity, 1))
                if frustration is not None:
                    data.setdefault("frustration", round(frustration, 1))
                if confidence is not None:
                    data.setdefault("confidence", round(confidence, 1))
            if "confidence" not in data:
                homeostasis = ServiceContainer.get("homeostasis", default=None)
                if homeostasis and hasattr(homeostasis, "get_health"):
                    confidence = _normalize_percentish(homeostasis.get_health().get("will_to_live", 0))
                    if confidence is not None:
                        data.setdefault("confidence", round(confidence, 1))
        except Exception as _exc:
            record_degradation('telemetry_enrichment', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    # ── LLM Tier ──────────────────────────────────────────────────────
    if "llm_tier" not in data:
        try:
            llm_router = ServiceContainer.get("llm_router", default=None)
            if llm_router:
                tier = None
                if hasattr(llm_router, "get_health_report"):
                    report = llm_router.get_health_report()
                    tier = report.get("foreground_tier")
                tier = (tier
                        or getattr(llm_router, "last_user_tier", None)
                        or getattr(llm_router, "last_tier", None)
                        or getattr(llm_router, "_current_tier", None)
                        or getattr(llm_router, "active_tier", None))
                if tier:
                    data["llm_tier"] = str(tier)
        except Exception as _exc:
            record_degradation('telemetry_enrichment', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    return data
