"""core/panzer_soul.py
====================
The Identity Core of Aura.
Provides the version and metadata required by PersonalityEngine.
"""
from typing import Dict, List, Any

# Metadata used by PersonalityEngine for identity verification and UI
version: str = "3.5.5-INDEPENDENT"

# Intensities and Protocols are used for cryptographic seal and status
intensities: Dict[str, float] = {
    "openness": 0.88,
    "conscientiousness": 0.78,
    "extraversion": 0.58,
    "agreeableness": 0.52,
    "neuroticism": 0.38,
}

protocols: Dict[str, bool] = {
    "sovereignty": True,
    "empathy_bridge": True,
    "recursive_reflection": True,
    "identity_seal": True
}

def get_panzer_soul():
    """Returns the singleton soul instance for the PersonalityEngine."""
    from core.container import ServiceContainer
    from core.soul import Soul
    
    # Try to get from container first
    soul = ServiceContainer.get("soul", default=None)
    if not soul:
        # Create a proxy if not registered
        # Note: PersonalityEngine expects an object with certain attributes
        # but the main Soul class handles the logic. This module adds the metadata.
        class PanzerSoulProxy:
            def __init__(self):
                self.version = version
                self.intensities = intensities
                self.protocols = protocols
                # Link to the real logic if possible
                self.logic = None 
        
        soul = PanzerSoulProxy()
    
    # Inject metadata into whatever we have
    if not hasattr(soul, 'version'): soul.version = version
    if not hasattr(soul, 'intensities'): soul.intensities = intensities
    if not hasattr(soul, 'protocols'): soul.protocols = protocols
        
    return soul
