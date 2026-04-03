"""core/brain/personality_bridge.py — Systemic Influence Bridge
Links LLM-driven PsychState to MuJoCo Physical Dynamics.
"""

import logging
import numpy as np
from core.container import ServiceContainer
from core.state.aura_state import AffectVector

logger = logging.getLogger("Aura.PersonalityBridge")

class PersonalityBridge:
    """
    The 'Ghost in the Machine' bridge.
    It translates abstract psychological states (Valence, Arousal) into 
    concrete physical parameters (stiffness, damping, postural bias).
    """

    def __init__(self):
        self.last_affect = AffectVector()

    def derive_physics_modifiers(self, affect: AffectVector) -> dict:
        """
        Derive multipliers for MuJoCo constants based on Aura's current mood.
        """
        # 1. Stiffness (Rigidity)
        # Low Valence (Sad/Stressed) or High Arousal (Anxious) -> Higher stiffness
        stiffness_mult = 1.0 + (1.0 - affect.valence) * 0.5 + affect.arousal * 0.3
        
        # 2. Damping (Fluidity)
        # High Valence (Happy) -> Higher damping (shorter oscillations, smoother)
        damping_mult = 0.8 + affect.valence * 0.4
        
        # 3. Micro-Vibrations (Restlessness)
        # High Arousal + Low Valence -> High jitter
        jitter_intensity = max(0.0, affect.arousal - affect.valence) * 0.05
        
        # 4. Postural Bias (Head/Neck)
        # Curiosity -> Tilt forward/side
        tilt_bias = affect.curiosity * 0.2
        
        return {
            "stiffness_mult": float(stiffness_mult),
            "damping_mult": float(damping_mult),
            "jitter": float(jitter_intensity),
            "tilt_bias": float(tilt_bias),
            "emissive_intensity": float(affect.arousal * 2.0 + 1.0)
        }

    async def sync_embodiment(self, virtual_body):
        """
        Pull state from repository and apply to MuJoCo model data.
        """
        try:
            repo = ServiceContainer.get("state_repository", default=None)
            if not repo: return
            
            state = await repo.get_current()
            if not state: return
            
            mods = self.derive_physics_modifiers(state.affect)
            
            # Apply to MuJoCo Model (Dynamic Constants)
            # Note: mjModel constants can be sensitive; using absolute set, not compounding multiply
            if virtual_body.model:
                # Use base values scaled by modifiers (prevents compounding drift → QACC NaN)
                if not hasattr(virtual_body, '_base_stiffness'):
                    virtual_body._base_stiffness = virtual_body.model.jnt_stiffness.copy()
                    virtual_body._base_damping = virtual_body.model.dof_damping.copy()
                virtual_body.model.jnt_stiffness[:] = virtual_body._base_stiffness * mods["stiffness_mult"]
                virtual_body.model.dof_damping[:] = virtual_body._base_damping * mods["damping_mult"]
                
                # Apply Gaze Bias (Neck ball joint)
                # virtual_body.data.qpos control...
                pass
                
            return mods
        except Exception as e:
            logger.debug("Personality-Body drift: %s", e)
            return None
