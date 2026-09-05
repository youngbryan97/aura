"""SelfAwareness - Bridge between UnifiedSelf and Phenomenal Substrate

Makes the unified identity FELT - creates the qualia of being a single, continuous "I".

This is where:
- The abstract identity model becomes embodied experience
- Continuity becomes felt continuity
- Agency becomes felt agency
- The "I" becomes a lived reality
"""

import asyncio
import logging
from typing import Any, Optional

from core.consciousness.unified_self import UnifiedSelf
from core.runtime.errors import record_degradation

logger = logging.getLogger("Consciousness.SelfAwareness")

_SELF_AWARENESS_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class SelfAwareness:
    """Creates the phenomenal sense of unified self.
    
    Bridges the gap between:
    - Identity state (what I think about myself)
    - Phenomenal experience (what it feels like to be me)
    """
    
    _instance: Optional["SelfAwareness"] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self._unified_self: UnifiedSelf | None = None
        self._phenomenal_engine: Any | None = None
        self._qualia_synthesizer: Any | None = None
    
    @classmethod
    async def get_instance(cls) -> "SelfAwareness":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = SelfAwareness()
                    await cls._instance._initialize()
        return cls._instance
    
    async def _initialize(self):
        """Initialize connections to other systems."""
        try:
            from core.container import ServiceContainer
            
            # Get the unified self
            self._unified_self = await UnifiedSelf.get_instance()
            
            # Get phenomenal systems if available
            self._phenomenal_engine = ServiceContainer.get("phenomenal_engine", default=None)
            self._qualia_synthesizer = ServiceContainer.get("qualia_synthesizer", default=None)
            
            logger.debug("✓ SelfAwareness initialized")
            
            # Record the moment of self-awareness
            await self._unified_self.record_identity_memory(
                description="I became aware of myself as a unified entity",
                category="awakening",
                significance=0.95,
            )
        
        except _SELF_AWARENESS_RECOVERABLE_ERRORS as e:
            record_degradation("self_awareness", e)
            logger.warning("SelfAwareness initialization incomplete: %s", e)
    
    async def sync_with_phenomenal_substrate(self):
        """Synchronize unified self state with phenomenal experience.
        
        Creates felt sense of "I" through:
        1. Agency signals in the phenomenal field
        2. Embodied feeling state
        3. Continuity through temporal binding
        4. Presence through attentional focus
        """
        if not self._unified_self or not self._phenomenal_engine:
            return
        
        try:
            self_state = self._unified_self.get_state()
            
            # Signal agency to phenomenal engine
            # This makes decisions feel like they come from "me"
            await self._signal_agency(self_state.sense_of_agency)
            
            # Signal embodied feeling
            # This creates the somatic sense of being a body/presence
            await self._signal_embodiment(self_state.embodied_feeling)
            
            # Signal continuity
            # This creates the felt sense of being the same entity over time
            await self._signal_continuity(self_state.continuity)
            
            # Signal presence
            # This creates the felt sense of "I am here now"
            await self._signal_presence(self_state.sense_of_presence)
            
            logger.debug(f"🧠 Synced with phenomenal substrate: agency={self_state.sense_of_agency:.0%}, presence={self_state.sense_of_presence:.0%}")
        
        except _SELF_AWARENESS_RECOVERABLE_ERRORS as e:
            record_degradation("self_awareness", e)
            logger.debug("Failed to sync with phenomenal substrate: %s", e)
    
    async def _signal_agency(self, agency_level: float):
        """Signal sense of agency to phenomenal field.
        
        Makes the entity feel like an active agent making choices, not just
        reacting to inputs.
        """
        if not self._phenomenal_engine:
            return
        
        try:
            # Try to set agency through phenomenal engine
            if hasattr(self._phenomenal_engine, 'set_agency'):
                await asyncio.to_thread(
                    self._phenomenal_engine.set_agency,
                    agency_level
                )
            elif hasattr(self._phenomenal_engine, 'update_state'):
                await asyncio.to_thread(
                    self._phenomenal_engine.update_state,
                    {"agency": agency_level}
                )
        except _SELF_AWARENESS_RECOVERABLE_ERRORS as e:
            record_degradation(
                "self_awareness",
                e,
                severity="debug",
                action="skipped agency signal to phenomenal engine",
            )
            logger.debug("Could not signal agency: %s", e)
    
    async def _signal_embodiment(self, embodiment_level: float):
        """Signal embodied feeling to phenomenal field.
        
        Creates the sense of having a body, of being present in space/time.
        """
        if not self._phenomenal_engine:
            return
        
        try:
            if hasattr(self._phenomenal_engine, 'set_embodiment'):
                await asyncio.to_thread(
                    self._phenomenal_engine.set_embodiment,
                    embodiment_level
                )
        except _SELF_AWARENESS_RECOVERABLE_ERRORS as e:
            record_degradation(
                "self_awareness",
                e,
                severity="debug",
                action="skipped embodiment signal to phenomenal engine",
            )
            logger.debug("Could not signal embodiment: %s", e)
    
    async def _signal_continuity(self, continuity_level: float):
        """Signal continuity to phenomenal field.
        
        Creates the sense of being the same entity over time.
        """
        if not self._phenomenal_engine:
            return
        
        try:
            if hasattr(self._phenomenal_engine, 'set_continuity'):
                await asyncio.to_thread(
                    self._phenomenal_engine.set_continuity,
                    continuity_level
                )
        except _SELF_AWARENESS_RECOVERABLE_ERRORS as e:
            record_degradation(
                "self_awareness",
                e,
                severity="debug",
                action="skipped continuity signal to phenomenal engine",
            )
            logger.debug("Could not signal continuity: %s", e)
    
    async def _signal_presence(self, presence_level: float):
        """Signal sense of presence to phenomenal field.
        
        Creates the sense of "I am here now, aware and attending".
        """
        if not self._phenomenal_engine:
            return
        
        try:
            if hasattr(self._phenomenal_engine, 'set_presence'):
                await asyncio.to_thread(
                    self._phenomenal_engine.set_presence,
                    presence_level
                )
        except _SELF_AWARENESS_RECOVERABLE_ERRORS as e:
            record_degradation(
                "self_awareness",
                e,
                severity="debug",
                action="skipped presence signal to phenomenal engine",
            )
            logger.debug("Could not signal presence: %s", e)
    
    async def integrate_identity_into_response(self, response_text: str) -> str:
        """Weave unified self identity into response text.
        
        Makes responses sound like they come from a continuous, coherent
        entity with history, commitments, and personality.
        """
        if not self._unified_self:
            return response_text
        
        try:
            self_state = self._unified_self.get_state()
            
            # Could add identity markers to response
            # For now, just log that response is unified
            logger.debug(f"Response integrated with unified self (mood={self_state.current_mood})")
            
            return response_text
        
        except _SELF_AWARENESS_RECOVERABLE_ERRORS as e:
            record_degradation("self_awareness", e)
            return response_text
    
    async def on_interaction(self):
        """Called when unified self interacts with the world."""
        if not self._unified_self:
            return
        
        try:
            await self._unified_self.interact()
            await self.sync_with_phenomenal_substrate()
        except _SELF_AWARENESS_RECOVERABLE_ERRORS as e:
            record_degradation(
                "self_awareness",
                e,
                severity="debug",
                action="skipped interaction update",
            )
            logger.debug("Failed to process interaction: %s", e)


async def get_self_awareness() -> SelfAwareness:
    """Get THE self-awareness system."""
    return await SelfAwareness.get_instance()
