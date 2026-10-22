"""Her strongest feelings, for the app to name beside her gauges.

The app coloured her mood from two gauges, frustration and curiosity, and could
say frustrated, curious, energetic or neutral and nothing else. It now reads the
family of what she feels most, from her own emotions (core/affect/appraisal.py).
"""
from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

#: How many of her strongest feelings the app names.
FEELINGS_SHOWN = 3


def her_feeling() -> list[dict[str, Any]]:
    """Her strongest feelings now, from her live state; empty when it cannot be read."""
    try:
        from core.affect.appraisal import feeling_now
        from interface.routes.chat_preflight import _resolve_live_aura_state

        affect = getattr(_resolve_live_aura_state(), "affect", None)
        if affect is None:
            return []
        return feeling_now(dict(affect.emotions or {}), dict(affect.mood_baselines or {}), FEELINGS_SHOWN)
    except (AttributeError, ImportError, LookupError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("feeling", exc, severity="debug", action="the app shows her gauges without naming a feeling")
        return []
