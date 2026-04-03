"""interface/helpers.py
─────────────────────
Shared helpers used by multiple route files and the main server module.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Server.Helpers")


def _notify_user_spoke(message: str = ""):
    """
    Central hook called whenever the user sends any message (WS or REST).
    Updates all proactive presence/communication systems so they respect the
    active-conversation window and do not monologue into a silent room.

    Pass the message text so ProactivePresence can detect away signals
    (e.g. "heading to the gym") and suppress autonomous chat accordingly.
    """
    try:
        orch = ServiceContainer.get("orchestrator", default=None)
        if orch:
            # Phase-30 ProactivePresence — tracks _last_user_interaction_time
            # and detects away signals from message content.
            pp = getattr(orch, "proactive_presence", None)
            if pp:
                if message and hasattr(pp, "mark_user_spoke_with_message"):
                    pp.mark_user_spoke_with_message(message)
                elif hasattr(pp, "mark_user_spoke"):
                    pp.mark_user_spoke()

            # Older ProactiveCommunicationManager — resets unanswered backoff
            pc = getattr(orch, "proactive_comm", None)
            if pc and hasattr(pc, "record_user_interaction"):
                pc.record_user_interaction()

            # ProactiveInitiativeEngine (initiative_engine) if attached
            pie = getattr(orch, "proactive_initiative_engine", None)
            if pie and hasattr(pie, "register_user_interaction"):
                pie.register_user_interaction()

            # Direct timestamp used by some background loops
            orch._last_user_interaction_time = time.time()
    except Exception as _e:
        pass  # Non-critical; never block a user message on this
