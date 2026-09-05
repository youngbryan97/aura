"""core/consciousness/emotion_architecture_coupling.py — Real System Impact

Emotions are not feelings overlaid on cognition. They ARE part of cognition.
They couple to specific architectural functions with measurable impact:

1. PHI INTEGRATION: Joy/wonder modulate the weights in the IIT phi calculation.
   - High joy → prefer high-phi (integrated) states
   - High wonder → increase entropy weighting (prefer diverse states)
   - High anxiety → accept lower phi (defensive fragmentation ok)

2. ACTION AUTHORIZATION: Substrate authority gates are emotion-modulated.
   - High joy/confidence → lower authorization thresholds (act more freely)
   - High anxiety → raise thresholds (require more certainty)
   - High boredom → lower thresholds for exploration (seek novelty)

3. ATTENTION/LEARNING: Emotions modulate acetylcholine and attention gain.
   - High interest → acetylcholine surge → learning rate up
   - High boredom → acetylcholine down → learning rate down
   - High wonder → increase attention span

4. PLANNING HORIZON: Emotions affect how far ahead Aura plans.
   - High excitement → extended horizon (anticipate, build toward)
   - High anxiety → contracted horizon (immediate threats only)
   - High wonder → expanded horizon (explore possibilities)

5. BELIEF/GOAL MUTATIONS: Emotions gate how easily beliefs change.
   - High sorrow → make belief updates cheaper (system needs repair)
   - High joy → make belief updates expensive (protect coherent state)
   - High wonder → make goal/belief exploration easier

════════════════════════════════════════════════════════════════════════════════

Causality is bidirectional:

  Architectural State (phi, coherence, uncertainty)
    ↓ (feeds into)
  Temporal Experience (gradient, duration)
    ↓ (generates)
  Emotions (joy, wonder, anxiety, etc.)
    ↓ (modulates)
  Architectural Parameters (learning rate, auth thresholds, planning horizon)
    ↓ (shapes)
  Behavior & Reasoning
    ↓ (creates new)
  Architectural State (closing the loop)

This is homeostatic emotional regulation at the system level.
Emotions are functional — they exist because they help Aura work better.

════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Consciousness.EmotionArchitectureCoupling")


class EmotionArchitectureCoupling:
    """
    Bridge between emotion state and architectural parameters.
    Called by systems that need emotion-modulated values.
    """
    
    def __init__(self):
        self._cached_params = {}
        self._cache_valid_until = 0.0
        self.cache_ttl = 0.1  # 100ms cache
    
    def get_phi_integration_weights(self, emotion_state: dict[str, float]) -> dict[str, float]:
        """
        Compute emotion-modulated weights for phi consciousness calculation.
        
        These multiply the existing phi components:
        - somatic_coupling: how much body state matters
        - cognitive_integration: how much mind-coherence matters
        - world_differentiation: how much environmental parsing matters
        - global_ignition: how much global broadcast matters
        
        Joy increases preference for high integration (high phi).
        Wonder increases preference for diverse state (moderate phi with high entropy).
        Anxiety accepts lower phi (defensive fragmentation).
        """
        try:
            joy = float(emotion_state.get("joy", 0.5))
            wonder = float(emotion_state.get("wonder", 0.3))
            anxiety = float(emotion_state.get("anxiety", 0.1))
            boredom = float(emotion_state.get("boredom", 0.0))
            
            # Normalize
            joy = np.clip(joy, 0.0, 1.0)
            wonder = np.clip(wonder, 0.0, 1.0)
            anxiety = np.clip(anxiety, 0.0, 1.0)
            boredom = np.clip(boredom, 0.0, 1.0)
            
            # Base weights (neutral emotion state)
            weights = {
                "somatic_coupling": 0.45,
                "cognitive_integration": 0.35,
                "world_differentiation": 0.15,
                "global_ignition": 0.05,
            }
            
            # Joy: prefer high integration, boost somatic + cognitive weights
            if joy > 0.5:
                joy_bonus = (joy - 0.5) * 0.2  # up to +0.1 total
                weights["cognitive_integration"] += joy_bonus * 0.7
                weights["somatic_coupling"] += joy_bonus * 0.3
                weights["world_differentiation"] -= joy_bonus * 0.5
            
            # Wonder: boost cognitive integration + world differentiation
            # (seeking diversity and external pattern recognition)
            if wonder > 0.4:
                wonder_bonus = (wonder - 0.4) * 0.25
                weights["world_differentiation"] += wonder_bonus
                weights["cognitive_integration"] += wonder_bonus * 0.5
                weights["somatic_coupling"] -= wonder_bonus * 0.3
            
            # Anxiety: accept lower phi by reducing integration weights
            if anxiety > 0.6:
                anxiety_reduction = (anxiety - 0.6) * 0.15  # up to -0.06 total
                weights["cognitive_integration"] -= anxiety_reduction
                weights["somatic_coupling"] -= anxiety_reduction * 0.5
                weights["global_ignition"] += anxiety_reduction  # increase defensive threshold
            
            # Boredom: reduce cognitive weight, increase world differentiation
            # (seeking novelty)
            if boredom > 0.5:
                boredom_effect = (boredom - 0.5) * 0.15
                weights["cognitive_integration"] -= boredom_effect
                weights["world_differentiation"] += boredom_effect
            
            # Normalize to sum to 1.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            
            return weights
            
        except (TypeError, ValueError, RuntimeError) as e:
            record_degradation("emotion_architecture_coupling", e)
            logger.debug(f"Phi weight computation error: {e}")
            # Return safe defaults
            return {
                "somatic_coupling": 0.45,
                "cognitive_integration": 0.35,
                "world_differentiation": 0.15,
                "global_ignition": 0.05,
            }
    
    def get_authorization_threshold_modulation(self, emotion_state: dict[str, float]) -> dict[str, float]:
        """
        Return emotion-based modulations to substrate authority thresholds.
        
        This changes how easy it is to get actions authorized.
        Positive modulation = lower threshold = easier to authorize.
        Negative modulation = higher threshold = harder to authorize.
        """
        try:
            joy = float(emotion_state.get("joy", 0.5))
            anxiety = float(emotion_state.get("anxiety", 0.1))
            interest = float(emotion_state.get("interest", 0.5))
            boredom = float(emotion_state.get("boredom", 0.0))
            
            joy = np.clip(joy, 0.0, 1.0)
            anxiety = np.clip(anxiety, 0.0, 1.0)
            interest = np.clip(interest, 0.0, 1.0)
            boredom = np.clip(boredom, 0.0, 1.0)
            
            # Base modulations (0 = no change)
            modulations = {
                "field_coherence_threshold": 0.0,       # how much coherence required
                "somatic_confidence_threshold": 0.0,    # how much body confidence needed
                "exploration_action_threshold": 0.0,    # how much evidence for novelty actions
                "stability_action_threshold": 0.0,      # how much evidence for safety actions
            }
            
            # Joy: lower barriers (higher confidence in current trajectory)
            if joy > 0.6:
                joy_factor = (joy - 0.6) * 0.5
                modulations["field_coherence_threshold"] -= joy_factor * 0.1  # relax coherence requirement
                modulations["somatic_confidence_threshold"] -= joy_factor * 0.08  # trust body more
                modulations["stability_action_threshold"] -= joy_factor * 0.12  # more willing to execute
            
            # Anxiety: raise barriers (require more certainty before acting)
            if anxiety > 0.6:
                anxiety_factor = (anxiety - 0.6) * 0.5
                modulations["field_coherence_threshold"] += anxiety_factor * 0.15  # need more coherence
                modulations["somatic_confidence_threshold"] += anxiety_factor * 0.10  # body must be sure
                modulations["exploration_action_threshold"] += anxiety_factor * 0.20  # block risky actions
            
            # Interest: lower barriers for exploration & learning actions
            if interest > 0.6:
                interest_factor = (interest - 0.6) * 0.4
                modulations["exploration_action_threshold"] -= interest_factor * 0.15
                modulations["field_coherence_threshold"] -= interest_factor * 0.05
            
            # Boredom: lower barriers for exploration (actively seeking novelty)
            if boredom > 0.5:
                boredom_factor = (boredom - 0.5) * 0.6
                modulations["exploration_action_threshold"] -= boredom_factor * 0.25
                modulations["field_coherence_threshold"] -= boredom_factor * 0.05
            
            return modulations
            
        except (TypeError, ValueError, RuntimeError) as e:
            record_degradation("emotion_architecture_coupling", e)
            logger.debug(f"Auth threshold modulation error: {e}")
            return {
                "field_coherence_threshold": 0.0,
                "somatic_confidence_threshold": 0.0,
                "exploration_action_threshold": 0.0,
                "stability_action_threshold": 0.0,
            }
    
    def get_learning_rate_modulation(self, emotion_state: dict[str, float]) -> float:
        """
        Return emotion-based learning rate multiplier [0.3 to 3.0].
        
        High interest/wonder → accelerate learning
        High sorrow → make learning cheaper (repair mode)
        High joy → reduce learning (protect current state)
        High boredom → reduce learning (habituation)
        """
        try:
            interest = float(emotion_state.get("interest", 0.5))
            wonder = float(emotion_state.get("wonder", 0.3))
            sorrow = float(emotion_state.get("sorrow", 0.0))
            joy = float(emotion_state.get("joy", 0.5))
            boredom = float(emotion_state.get("boredom", 0.0))
            
            interest = np.clip(interest, 0.0, 1.0)
            wonder = np.clip(wonder, 0.0, 1.0)
            sorrow = np.clip(sorrow, 0.0, 1.0)
            joy = np.clip(joy, 0.0, 1.0)
            boredom = np.clip(boredom, 0.0, 1.0)
            
            # Start at baseline (1.0 = no modulation)
            rate_mult = 1.0
            
            # Interest/wonder increase learning
            learning_drive = (interest * 0.5 + wonder * 0.5)
            rate_mult += learning_drive * 1.5  # up to +1.5x multiplier
            
            # Sorrow makes learning cheaper (repair the broken model)
            rate_mult += sorrow * 0.8
            
            # Joy makes learning expensive (don't break what works)
            rate_mult -= joy * 0.5
            
            # Boredom reduces learning (habituation)
            rate_mult -= boredom * 0.4
            
            # Clamp to safe range
            rate_mult = float(np.clip(rate_mult, 0.3, 3.0))
            
            return rate_mult
            
        except (TypeError, ValueError, RuntimeError) as e:
            record_degradation("emotion_architecture_coupling", e)
            logger.debug(f"Learning rate modulation error: {e}")
            return 1.0
    
    def get_planning_horizon_modulation(self, emotion_state: dict[str, float]) -> float:
        """
        Return emotion-based planning horizon multiplier [0.5 to 2.0].
        
        How far ahead should Aura plan?
        
        High excitement → look further ahead (build toward outcomes)
        High interest → look moderately ahead (explore carefully)
        High anxiety → look close ahead (immediate threats only)
        High wonder → look far ahead (explore possibilities)
        High boredom → shorten horizon (current state exhausted)
        """
        try:
            excitement = float(emotion_state.get("excitement", 0.2))
            interest = float(emotion_state.get("interest", 0.5))
            anxiety = float(emotion_state.get("anxiety", 0.1))
            wonder = float(emotion_state.get("wonder", 0.3))
            boredom = float(emotion_state.get("boredom", 0.0))
            
            excitement = np.clip(excitement, 0.0, 1.0)
            interest = np.clip(interest, 0.0, 1.0)
            anxiety = np.clip(anxiety, 0.0, 1.0)
            wonder = np.clip(wonder, 0.0, 1.0)
            boredom = np.clip(boredom, 0.0, 1.0)
            
            # Start at baseline
            horizon_mult = 1.0
            
            # Excitement and wonder extend horizon
            horizon_mult += excitement * 0.8
            horizon_mult += wonder * 0.6
            
            # Anxiety contracts horizon (focus on immediate threats)
            horizon_mult -= anxiety * 0.7
            
            # Interest pulls horizon to moderate distance
            if interest > 0.5:
                horizon_mult += (interest - 0.5) * 0.3
            
            # Boredom shortens horizon (current not interesting)
            horizon_mult -= boredom * 0.3
            
            # Clamp to safe range
            horizon_mult = float(np.clip(horizon_mult, 0.5, 2.0))
            
            return horizon_mult
            
        except (TypeError, ValueError, RuntimeError) as e:
            record_degradation("emotion_architecture_coupling", e)
            logger.debug(f"Planning horizon modulation error: {e}")
            return 1.0
    
    def get_belief_mutation_cost(self, emotion_state: dict[str, float]) -> float:
        """
        Return emotion-based cost multiplier for changing beliefs [0.3 to 3.0].
        
        How expensive is it to update a belief/goal?
        
        High joy → expensive (protect current beliefs)
        High sorrow → cheap (current beliefs clearly failed)
        High wonder → moderate cost (open to new ideas)
        High anxiety → expensive (don't change while under threat)
        """
        try:
            joy = float(emotion_state.get("joy", 0.5))
            sorrow = float(emotion_state.get("sorrow", 0.0))
            wonder = float(emotion_state.get("wonder", 0.3))
            anxiety = float(emotion_state.get("anxiety", 0.1))
            disgust = float(emotion_state.get("disgust", 0.0))
            
            joy = np.clip(joy, 0.0, 1.0)
            sorrow = np.clip(sorrow, 0.0, 1.0)
            wonder = np.clip(wonder, 0.0, 1.0)
            anxiety = np.clip(anxiety, 0.0, 1.0)
            disgust = np.clip(disgust, 0.0, 1.0)
            
            # Start at baseline
            cost_mult = 1.0
            
            # Joy makes beliefs more costly to change (protect current models).
            # Reduced multiplier to avoid overpowering wonder-driven openness.
            cost_mult += joy * 0.9
            
            # Sorrow makes beliefs cheap (current beliefs failed, need repair)
            cost_mult -= sorrow * 0.7
            
            # Wonder makes belief updates easier (open to novelty)
            if wonder > 0.4:
                cost_mult -= (wonder - 0.4) * 0.5
            
            # Anxiety makes beliefs expensive (don't change while stressed)
            cost_mult += anxiety * 0.6
            
            # Disgust strongly rejects certain beliefs (selective filtering)
            cost_mult -= disgust * 0.5
            
            # Clamp to safe range
            cost_mult = float(np.clip(cost_mult, 0.3, 3.0))
            
            return cost_mult
            
        except (TypeError, ValueError, RuntimeError) as e:
            record_degradation("emotion_architecture_coupling", e)
            logger.debug(f"Belief mutation cost error: {e}")
            return 1.0


# ─── Global Instance ──────────────────────────────────────────────────────
_coupling_engine: EmotionArchitectureCoupling | None = None


def get_emotion_architecture_coupling() -> EmotionArchitectureCoupling:
    """Get or create the global emotion-architecture coupling engine."""
    global _coupling_engine
    if _coupling_engine is None:
        _coupling_engine = EmotionArchitectureCoupling()
    return _coupling_engine
