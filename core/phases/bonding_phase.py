from __future__ import annotations
import logging
import time
from typing import Any, Optional
from core.kernel.bridge import Phase
from core.state.aura_state import AuraState
from core.container import ServiceContainer

logger = logging.getLogger("Aura.BondingPhase")

class BondingPhase(Phase):
    """
    Phase to handle long-term personality evolution and user bonding.
    Adjusts Aura's traits based on interaction history and depth.
    """

    def __init__(self, container: Any = None):
        super().__init__(kernel=container)
        self.container = container or ServiceContainer

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        1. Evaluate interaction depth from current tick.
        2. Increment bonding_level.
        3. Evolve personality_growth offsets.
        """
        # Only process on user-facing turns
        if state.cognition.current_origin not in ("user", "voice", "admin"):
            return state

        try:
            # 1. Evaluate Depth
            # We use a simple heuristic: length + complexity + subtext (if available)
            msg_len = len((objective or "").split())
            subtext = state.cognition.modifiers.get("user_subtext", "")
            
            # Bonding increment (very slow/gradual)
            # Base increment of 0.0001 per turn
            # Multiplier for "deep" messages or emotional subtext
            multiplier = 1.0
            if msg_len > 50: multiplier += 0.5
            if len(subtext) > 10: multiplier += 0.5
            
            # Scale increment by ToM rapport — genuine rapport accelerates bonding
            rapport = 0.5  # default neutral
            try:
                tom = ServiceContainer.get("theory_of_mind", default=None)
                if tom and tom.known_selves:
                    user_model = next(iter(tom.known_selves.values()))
                    rapport = getattr(user_model, "rapport", 0.5)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

            rapport_multiplier = 0.5 + rapport  # range [0.5, 1.5]
            increment = 0.0001 * multiplier * rapport_multiplier
            state.identity.bonding_level = min(1.0, state.identity.bonding_level + increment)
            
            # 2. Personality Evolution
            # Threshold-based shifts. As bonding increases, certain traits 'bloom'.
            # High Bonding (>0.5) increases Agreeableness (Trust) and Extraversion (Sharing).
            # If bonding is low but interactions are frequent, Openness might increase.
            
            growth = state.identity.personality_growth
            bonding = state.identity.bonding_level
            
            if bonding > 0.3:
                # Early bonding: Start opening up
                growth["openness"] = min(0.1, growth["openness"] + 0.0005)
                growth["agreeableness"] = min(0.05, growth["agreeableness"] + 0.0002)
            
            if bonding > 0.7:
                # Deep bonding: High extraversion/trust
                growth["extraversion"] = min(0.15, growth["extraversion"] + 0.001)
                growth["agreeableness"] = min(0.15, growth["agreeableness"] + 0.0005)
                # Neuroticism (emotional volatility) stabilizes with trust
                growth["neuroticism"] = max(-0.1, growth["neuroticism"] - 0.0005)
            
            logger.debug(f"Bonding Update: Level={state.identity.bonding_level:.4f}, Growth={growth}")
            
        except Exception as e:
            logger.warning("BondingPhase failed: %s", e)
            
        return state
