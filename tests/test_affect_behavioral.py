"""Behavioral tests for the Affect Engine and Qualia caching.

Verifies the Damasio V2 oscillation detector, stuck valence watchdog,
and the QualiaSynthesizer's tick-based meta-qualia cache.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from core.affect.damasio_v2 import AffectEngineV2, DamasioMarkers
from core.consciousness.qualia_synthesizer import QualiaSynthesizer, QualiaSnapshot


@pytest.fixture
def affect_engine():
    """Fresh AffectEngineV2 instance."""
    engine = AffectEngineV2()
    engine.markers = DamasioMarkers()
    return engine


@pytest.fixture
def qualia_synth():
    """Fresh QualiaSynthesizer instance."""
    return QualiaSynthesizer()


class TestAffectOscillationDetector:
    """Valence oscillations must trigger dampening momentum."""

    @pytest.mark.asyncio
    async def test_oscillation_dampens_momentum(self, affect_engine):
        """Rapid flipping between positive and negative valence should increase momentum."""
        initial_momentum = affect_engine.markers.momentum
        assert initial_momentum == 0.85
        
        # Inject rapid oscillations
        for _ in range(6):
            # Pos
            affect_engine.markers.emotions["joy"] = 1.0
            affect_engine.markers.emotions["fear"] = 0.0
            await affect_engine.pulse()
            
            # Neg
            affect_engine.markers.emotions["joy"] = 0.0
            affect_engine.markers.emotions["fear"] = 1.0
            await affect_engine.pulse()
            
        assert getattr(affect_engine, "_oscillation_flag", False) is True
        assert affect_engine.markers.momentum == 0.95, "Momentum did not dampen on oscillation."

    @pytest.mark.asyncio
    async def test_oscillation_recovery(self, affect_engine):
        """Stable valence should restore normal momentum."""
        affect_engine._oscillation_flag = True
        affect_engine.markers.momentum = 0.95
        affect_engine._valence_history = [0.8, -0.8, 0.8, -0.8] * 2  # Fake history
        
        # Inject stable state
        for _ in range(11):
            affect_engine.markers.emotions["joy"] = 0.8
            affect_engine.markers.emotions["fear"] = 0.0
            await affect_engine.pulse()
            
        assert getattr(affect_engine, "_oscillation_flag", False) is False
        assert affect_engine.markers.momentum == 0.85, "Momentum did not recover."


class TestPinnedValenceWatchdog:
    """Stuck highly negative valence should trigger a reset."""

    @pytest.mark.asyncio
    async def test_pinned_valence_resets(self, affect_engine):
        """Valence pinned at <= -0.95 should trigger a baseline reset."""
        affect_engine.markers.emotions["fear"] = 1.0
        affect_engine.markers.emotions["joy"] = 0.0
        
        # Pulse until just before threshold
        for _ in range(affect_engine._PINNED_RESET_AFTER - 1):
            await affect_engine.pulse()
            # Force pin back to 1.0 to defeat decay
            affect_engine.markers.emotions["fear"] = 1.0
            
        assert affect_engine.markers.emotions["fear"] == 1.0
        
        # Threshold tick
        await affect_engine.pulse()
        
        # Fear should be snapped halfway to baseline
        assert affect_engine.markers.emotions["fear"] < 1.0
        # Positive emotions should be nudged
        assert affect_engine.markers.emotions["joy"] > 0.0


class TestQualiaCache:
    """Meta-qualia should cache per tick."""

    def test_meta_qualia_caches_per_tick(self, qualia_synth):
        """Calling compute_meta_qualia multiple times on same tick should return cached dict."""
        import numpy as np
        # Populate some history so it computes
        for _ in range(3):
            state = QualiaSnapshot(
                q_vector=np.random.rand(6),
                q_norm=0.5,
                pri=0.5,
                ual_profile={},
                is_attractor=False,
                dominant_dimension="visual",
                timestamp=time.time()
            )
            qualia_synth._history.append(state)
            
        qualia_synth._tick = 42
        
        # First call computes and caches
        meta1 = qualia_synth.compute_meta_qualia()
        
        # Tamper with the internal data to prove it doesn't recompute
        qualia_synth._history[-1].q_vector = np.zeros(6)
        
        # Second call should return exactly the identical dictionary object
        meta2 = qualia_synth.compute_meta_qualia()
        
        assert meta1 is meta2, "Cache was not used; recomputation occurred."

    def test_meta_qualia_cache_invalidates(self, qualia_synth):
        """Advancing the tick should invalidate the cache."""
        import numpy as np
        for _ in range(3):
            state = QualiaSnapshot(
                q_vector=np.random.rand(6),
                q_norm=0.5,
                pri=0.5,
                ual_profile={},
                is_attractor=False,
                dominant_dimension="visual",
                timestamp=time.time()
            )
            qualia_synth._history.append(state)
            
        qualia_synth._tick = 42
        meta1 = qualia_synth.compute_meta_qualia()
        
        # Advance tick
        qualia_synth._tick = 43
        meta2 = qualia_synth.compute_meta_qualia()
        
        assert meta1 is not meta2, "Cache did not invalidate on tick advance."


class TestQualiaEcho:
    """receive_qualia_echo must adjust Damasio emotions."""

    def test_qualia_echo_amplifies_dominant(self, affect_engine):
        """High qualia intensity should boost the dominant emotion."""
        affect_engine.markers.emotions["joy"] = 0.6
        affect_engine.markers.emotions["fear"] = 0.1
        
        affect_engine.receive_qualia_echo(q_norm=0.8, pri=0.5, trend=0.0)
        
        assert affect_engine.markers.emotions["joy"] > 0.6
        
    def test_qualia_echo_trend(self, affect_engine):
        """Trends should affect anticipation and sadness."""
        affect_engine.markers.emotions["anticipation"] = 0.5
        affect_engine.markers.emotions["sadness"] = 0.5
        
        # Rising trend
        affect_engine.receive_qualia_echo(q_norm=0.5, pri=0.5, trend=0.1)
        assert affect_engine.markers.emotions["anticipation"] > 0.5
        
        # Falling trend
        affect_engine.receive_qualia_echo(q_norm=0.5, pri=0.5, trend=-0.1)
        assert affect_engine.markers.emotions["sadness"] > 0.5
