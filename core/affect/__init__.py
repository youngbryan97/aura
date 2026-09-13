"""
core/affect/__init__.py

FIX: AffectState was defined here AND in core/affect/damasio_v2.py.
This caused ambiguity about which AffectState was being instantiated
by which component. Canonical definition lives here. damasio_v2 imports
from here.

A PAD ``AffectEngine`` also lived here, documented as the "lightweight fallback
when DamasioV2 is unavailable". It was unreachable: its only constructor in the
whole tree was core/affect/emotion_engine.py, and that module had zero
importers. Aura therefore shipped two affect engines of which one could never
run, while two tests spent coverage on it.

It was removed rather than wired up, because wiring it would break CP126's
rule that "no engine must never look like a calm engine" — a silent fallback to
a different affect model emits confident numbers from a model nothing else
reads, and no caller could tell which engine answered. The canonical engine is
core/affect/damasio_v2.py::AffectEngineV2, registered as the `affect_engine`
service; tests/test_one_canonical_affect_engine.py pins that there is one.

The decay baselines below are retained: they are the PAD reference points, and
AffectState is still the canonical state type used across the codebase.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("Aura.Affect")


@dataclass
class AffectState:
    """
    Canonical affective state representation.
    
    PAD model: Pleasure-Arousal-Dominance.
    
    Single definition for the entire codebase.
    Import from core.affect, not from core.affect.damasio_v2.
    """
    valence:          float = 0.0    # -1.0 (negative) to 1.0 (positive)
    arousal:          float = 0.3    # 0.0 (calm) to 1.0 (agitated)
    engagement:       float = 0.5    # 0.0 (bored) to 1.0 (hyper-focused)
    dominant_emotion: str   = "Neutral"
    last_update:      float = field(default_factory=lambda: time.time())


# Decay baselines
BASELINE_VALENCE    = 0.1
BASELINE_AROUSAL    = 0.3
BASELINE_ENGAGEMENT = 0.5
DECAY_RATE          = 0.02


# Phenomenal Substrate Integration
# Lazy import to avoid circular dependencies
def get_phenomenal_integrator():
    """Lazy accessor for the phenomenal integrator singleton."""
    from core.affect.phenomenal_integration import PhenomenalIntegrator
    return PhenomenalIntegrator._sync_instance()
