"""core/memory/autobiography.py
Autobiographical memory orchestrator building narrative continuity.
"""
import logging
import time
import uuid
from typing import Any

from core.memory.episode_store import EpisodeStore
from core.memory.life_event import LifeEvent

logger = logging.getLogger("Memory.Autobiography")


class AutobiographyEngine:
    """Manages the creation and organization of narrative autobiographical events."""

    def __init__(self):
        self.store = EpisodeStore()

    async def record_tick_event(self, state: Any, receipt: dict[str, Any] | None) -> None:
        """Assembles a single LifeEvent trace from the current loop status and appends to disk."""
        
        event_id = str(uuid.uuid4())
        
        # Pull details from current LifeState
        perceived = state.world_model.get("last_observations", {})
        believed = state.world_model.get("active_beliefs", {})
        wanted = {"goals": [g.get("id") for g in state.cognition.current_goals]}
        
        chose = receipt.get("intent", {}) if receipt else {}
        did = receipt.get("channel", {}) if receipt else {}
        what_happened = receipt if receipt else {}
        
        what_changed = state.world_model.get("last_verification", {}).get("side_effects", [])

        # Formulate introspective logs based on status
        what_she_learned = "Action succeeded."
        if receipt and receipt.get("status") == "failed":
            what_she_learned = f"Action failed: {receipt.get('error')}"

        event = LifeEvent(
            event_id=event_id,
            timestamp=time.time(),
            perceived=perceived,
            believed=believed,
            wanted=wanted,
            chose=chose,
            did={"action": did},
            what_happened=what_happened,
            what_changed={"changed": what_changed},
            what_she_learned=what_she_learned,
            what_she_should_avoid="Avoid parameters causing crashes." if receipt and receipt.get("status") == "failed" else "",
            what_she_should_remember="Confirm workspace constraints.",
            what_remains_unresolved=""
        )

        # Append to active state list and save to disk
        state.autobiographical_memory.append(event.to_dict())
        await self.store.save_event(event)
        
        logger.info("Recorded autobiographical life event: %s", event_id)
