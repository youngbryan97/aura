"""
core/senses/entropy_anchor.py
─────────────────────────────
Tethers Aura's cognitive and affective states to true physical reality 
by reading hardware-level thermal and quantum entropy from the local machine.
"""

import logging
import os
import struct

logger = logging.getLogger("Aura.EntropyAnchor")

class PhysicalEntropyAnchor:
    def __init__(self) -> None:
        logger.info("🌌 Physical Entropy Anchor online. System is now non-deterministic.")

    def get_raw_bytes(self, num_bytes: int = 4) -> bytes:
        """Read raw entropy directly from the OS hardware pool."""
        # On macOS/Linux, os.urandom provides entropy from /dev/urandom,
        # which is seeded by hardware/environmental noise.
        return os.urandom(num_bytes)
        
    def get_entropy_float(self) -> float:
        """
        Generates a non-deterministic float between 0.0 and 1.0 
        derived entirely from hardware thermodynamic noise.
        """
        # Read 4 bytes of physical entropy
        raw_bytes = self.get_raw_bytes(4)
        
        # Unpack the bytes into an unsigned 32-bit integer
        entropy_int = struct.unpack("I", raw_bytes)[0]
        
        # Normalize to a float between 0.0 and 1.0
        return float(entropy_int / (2**32 - 1))
        
    def get_vad_drift(self, volatility_multiplier: float = 0.02) -> float:
        """
        Generates a continuous, unpredictable drift value to inject 
        into Valence, Arousal, and Dominance (VAD) calculations.
        Returns a float between [-volatility_multiplier and +volatility_multiplier].
        """
        base_entropy = self.get_entropy_float()  # 0.0 to 1.0
        
        # Shift to range [-1.0 to 1.0], then scale by the volatility multiplier
        drift = ((base_entropy * 2.0) - 1.0) * volatility_multiplier
        return round(drift, 4)

# Instantiate the singleton
entropy_anchor = PhysicalEntropyAnchor()
