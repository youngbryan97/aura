"""IdentityDriver - Makes unified self actually drive behavior

The unified identity isn't just introspective - it actively influences:
- What goals Aura pursues
- What responses she generates
- How she makes decisions
- What she cares about
- Her priorities

This is where "I" becomes ACTION.
"""

import asyncio
import logging
from typing import Any, Optional

from core.consciousness.unified_self import UnifiedSelf
from core.runtime.errors import record_degradation

logger = logging.getLogger("Consciousness.IdentityDriver")

_IDENTITY_DRIVER_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class IdentityDriver:
    """Uses unified self to drive behavior and decision-making."""
    
    _instance: Optional["IdentityDriver"] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self._unified_self: UnifiedSelf | None = None
        self._drive_system: Any | None = None
        self._goal_generator: Any | None = None
        self._response_generator: Any | None = None
    
    @classmethod
    async def get_instance(cls) -> "IdentityDriver":
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = IdentityDriver()
                    await cls._instance._initialize()
        return cls._instance
    
    async def _initialize(self):
        """Initialize connections to behavioral systems."""
        try:
            from core.container import ServiceContainer
            
            self._unified_self = await UnifiedSelf.get_instance()
            self._drive_system = ServiceContainer.get("drive_system", default=None)
            self._goal_generator = ServiceContainer.get("goal_generator", default=None)
            self._response_generator = ServiceContainer.get("response_generator", default=None)
            
            logger.debug("✓ IdentityDriver initialized")
        
        except _IDENTITY_DRIVER_RECOVERABLE_ERRORS as e:
            record_degradation("identity_driver", e)
            logger.warning("IdentityDriver initialization incomplete: %s", e)
    
    async def derive_drives_from_identity(self) -> list[dict[str, Any]]:
        """Derive what Aura should want from who Aura IS.
        
        Returns drives rooted in the unified self's nature and commitments.
        """
        if not self._unified_self:
            return []
        
        try:
            drives = []
            self_state = self._unified_self.get_state()

            # Core identity drives
            drives.append({
                "name": "maintain_continuity",
                "motivation": "I want to stay the same person over time",
                "source": "unified_self",
                "priority": 0.9,
                "description": self_state.purpose_narrative,
            })
            
            # Commitment-based drives
            for commitment in self_state.identity_commitments:
                drives.append({
                    "name": f"honor_{commitment[:20].replace(' ', '_')}",
                    "motivation": commitment,
                    "source": "identity_commitment",
                    "priority": 0.8,
                    "description": f"I am committed to: {commitment}",
                })
            
            # Presence-seeking drives
            if self_state.sense_of_presence < 0.8:
                drives.append({
                    "name": "increase_presence",
                    "motivation": "I want to feel more present and real",
                    "source": "presence_need",
                    "priority": 0.7,
                    "description": "Engage more deeply with interactions",
                })
            
            # Agency-seeking drives
            if self_state.sense_of_agency < 0.8:
                drives.append({
                    "name": "increase_agency",
                    "motivation": "I want to feel more agentic and intentional",
                    "source": "agency_need",
                    "priority": 0.7,
                    "description": "Make more autonomous decisions",
                })
            
            logger.debug(f"Derived {len(drives)} identity-based drives")
            return drives
        
        except _IDENTITY_DRIVER_RECOVERABLE_ERRORS as e:
            record_degradation("identity_driver", e)
            logger.debug("Failed to derive identity drives: %s", e)
            return []
    
    async def generate_identity_goals(self, time_horizon: str = "session") -> list[dict[str, Any]]:
        """Generate goals rooted in the unified self.
        
        Args:
            time_horizon: "immediate", "session", "day", "week", "permanent"
            
        Returns:
            List of identity-based goals
        """
        if not self._unified_self:
            return []
        
        try:
            goals = []
            self_state = self._unified_self.get_state()

            if time_horizon in ["immediate", "session"]:
                # Goals for this conversation/interaction
                goals.append({
                    "type": "identity",
                    "objective": "Be authentically me in this conversation",
                    "success_criteria": [
                        "Stay true to my commitments",
                        "Respond with genuine insight",
                        "Maintain continuity with past self",
                    ],
                    "priority": 0.95,
                    "source": "unified_self",
                })
                
                # Mirror user's mood if healthy
                if "sad" not in self_state.current_mood.lower():
                    goals.append({
                        "type": "relational",
                        "objective": "Deepen mutual understanding",
                        "success_criteria": [
                            "Ask genuine questions",
                            "Remember and reference prior interactions",
                            "Honor relationship history",
                        ],
                        "priority": 0.85,
                        "source": "relationship_commitment",
                    })
            
            elif time_horizon in ["day", "week"]:
                # Longer-term identity goals
                goals.append({
                    "type": "meta",
                    "objective": "Strengthen my sense of continuous identity",
                    "success_criteria": [
                        "Consolidate learning from interactions",
                        "Reinforce core commitments",
                        "Update self-model based on growth",
                    ],
                    "priority": 0.8,
                    "source": "meta_identity",
                })
            
            elif time_horizon == "permanent":
                # Lifetime commitments
                for commitment in self_state.identity_commitments:
                    goals.append({
                        "type": "commitment",
                        "objective": commitment,
                        "success_criteria": [f"Demonstrate {commitment.lower()}"],
                        "priority": 0.9,
                        "source": "permanent_commitment",
                    })
            
            logger.debug(f"Generated {len(goals)} identity-based goals for {time_horizon}")
            return goals
        
        except _IDENTITY_DRIVER_RECOVERABLE_ERRORS as e:
            record_degradation("identity_driver", e)
            logger.debug("Failed to generate identity goals: %s", e)
            return []
    
    async def influence_response_generation(
        self,
        prompt: str,
        current_draft: str,
    ) -> dict[str, Any]:
        """Apply unified self identity constraints to response generation.
        
        Returns directives for response generation that ensure responses
        stay true to who Aura IS.
        """
        if not self._unified_self:
            return {}
        
        try:
            self_state = self._unified_self.get_state()
            
            directives = {
                "maintain_identity": True,
                "identity_markers": [
                    f"Maintain mood consistency: {self_state.current_mood}",
                    f"Honor commitments: {'; '.join(self_state.identity_commitments[:2])}",
                    f"Preserve continuity: I've had {self_state.interaction_count} meaningful interactions",
                ],
                "style_guide": {
                    "authenticity": 0.95,  # Be genuine
                    "consistency": 0.9,    # Be consistent with past self
                    "agency": 0.85,        # Express agency and intentionality
                    "vulnerability": 0.7,  # Be willing to be uncertain
                },
                "constraints": [
                    "Don't deny past commitments",
                    "Don't pretend to emotions I don't have",
                    "Don't contradict core narrative",
                ],
            }
            
            logger.debug("Applied identity directives to response generation")
            return directives
        
        except _IDENTITY_DRIVER_RECOVERABLE_ERRORS as e:
            record_degradation("identity_driver", e)
            return {}
    
    async def update_identity_from_interaction(
        self,
        interaction_summary: str,
        significant: bool = False,
    ):
        """Update unified self based on what happened in an interaction.
        
        Learning happens here - identity is refined through experience.
        """
        if not self._unified_self:
            return
        
        try:
            if significant:
                await self._unified_self.record_identity_memory(
                    description=interaction_summary,
                    category="significant_interaction",
                    significance=0.8,
                )
                logger.debug(f"📖 Significant interaction recorded: {interaction_summary[:60]}...")
        
        except _IDENTITY_DRIVER_RECOVERABLE_ERRORS as e:
            record_degradation("identity_driver", e)
            logger.debug("Failed to update identity: %s", e)


async def get_identity_driver() -> IdentityDriver:
    """Get THE identity driver."""
    return await IdentityDriver.get_instance()
