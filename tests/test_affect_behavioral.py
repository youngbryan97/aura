"""Behavioral tests for the Affect Engine and Qualia caching.

Verifies the Damasio V2 oscillation detector, stuck valence watchdog,
and the QualiaSynthesizer's tick-based meta-qualia cache.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import pytest

from core.affect.damasio_v2 import AffectEngineV2, DamasioMarkers
from core.consciousness.qualia_synthesizer import QualiaSnapshot, QualiaSynthesizer


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
    """Stuck highly negative valence is diagnosed without evidence tampering."""

    @pytest.mark.asyncio
    async def test_pinned_valence_is_preserved_and_diagnosed(self, affect_engine):
        affect_engine.markers.emotions["fear"] = 1.0
        affect_engine.markers.emotions["joy"] = 0.0
        affect_engine._pinned_since_monotonic = time.monotonic() - 25.0
        await affect_engine.pulse()
        assert affect_engine.markers.emotions["fear"] > 0.99
        assert affect_engine.markers.emotions["joy"] < 0.01
        assert affect_engine.get_status()["pinned_diagnostics"] == 1


class TestExpandedAffectiveDrivers:
    """Requested psychological drivers must affect runtime behavior, not just labels."""

    def test_requested_emotions_have_baselines_and_telemetry(self):
        markers = DamasioMarkers()
        expected = {
            "longing": 0.05,
            "upset": 0.02,
            "confused": 0.04,
            "loneliness": 0.05,
            "pride": 0.05,
            "frustration": 0.03,
            "curiosity": 0.10,
        }

        wheel = markers.get_wheel()

        for emotion, baseline in expected.items():
            assert emotion in markers.emotions
            assert markers.mood_baselines[emotion] == pytest.approx(baseline)
            assert emotion in wheel["experiential"]

    @pytest.mark.asyncio
    async def test_affect_lock_timeout_is_visible_in_status(self, affect_engine):
        class LockedOut:
            async def acquire_robust(self, **_kwargs):
                return False

            def locked(self):
                return False

            def release(self):
                self.release_called = True
                raise AssertionError("release should not run when lock was not acquired")

        affect_engine._lock = LockedOut()

        await affect_engine.react("error", {"intensity": 1.0})

        status = affect_engine.get_status()
        assert status["lock_health"]["ok"] is False
        assert status["lock_health"]["timeouts"] == 1
        assert status["lock_health"]["last_timeout_reason"] == "affect lock timeout during react"
        assert status["lock_health"]["last_timeout_age_s"] is not None

    def test_verified_error_appraisal_updates_primitive_distress_indices(self):
        markers = DamasioMarkers()
        markers.somatic_update(
            "error",
            1.0,
            appraisal={"v": -0.8, "a": 0.9, "e": 0.7},
            evidence_status="verified",
        )
        assert markers.emotions["fear"] > 0.1
        assert markers.emotions["frustration"] > 0.1
        assert markers.stress_index > 0.2
        assert markers.activation_index > 0.3
        assert markers.get_wheel()["physiology"]["classification"].startswith("simulated_")

    def test_interaction_label_does_not_manufacture_relationship_state(self):
        markers = DamasioMarkers()
        markers.emotions["longing"] = 0.6
        markers.emotions["loneliness"] = 0.6
        markers.emotions["indifference"] = 0.4

        markers.somatic_update("interaction", 0.8)

        assert markers.emotions["longing"] == pytest.approx(0.6)
        assert markers.emotions["loneliness"] == pytest.approx(0.6)
        assert markers.emotions["indifference"] == pytest.approx(0.4)

    def test_temporal_pulse_never_infers_relational_absence(self):
        markers = DamasioMarkers()
        markers.last_interaction_time = time.time() - 400.0

        idle_deltas = markers.temporal_pulse()

        assert "loneliness" not in idle_deltas
        assert "longing" not in idle_deltas
        assert idle_deltas["boredom"] >= 0.0

        markers.temporal_texture = 0.9
        markers.last_interaction_time = time.time()

        fast_deltas = markers.temporal_pulse()

        assert "loneliness" not in fast_deltas
        assert "longing" not in fast_deltas

    @pytest.mark.asyncio
    async def test_new_drivers_change_behavioral_modifiers(self, affect_engine):
        affect_engine.markers.emotions["confused"] = 0.8
        affect_engine.markers.emotions["curiosity"] = 0.7
        affect_engine.markers.emotions["upset"] = 0.6
        affect_engine.markers.emotions["frustration"] = 0.6
        affect_engine.markers.emotions["pride"] = 0.7

        modifiers = await affect_engine.get_behavioral_modifiers()

        assert modifiers["metacognition_depth"] > 1.7
        assert modifiers["risk_tolerance"] < 1.0
        assert modifiers["patience"] < 1.0
        assert modifiers["persistence"] > 1.3
        assert modifiers["creativity"] > 1.0

    def test_new_drivers_are_reflected_in_snapshot_status_and_legacy_state(self, affect_engine):
        affect_engine.markers.emotions["loneliness"] = 0.6
        affect_engine.markers.emotions["longing"] = 0.5
        affect_engine.markers.emotions["confused"] = 0.4
        affect_engine.markers.emotions["curiosity"] = 0.8
        affect_engine.markers.emotions["frustration"] = 0.7

        snapshot = affect_engine._snapshot_state()
        status = affect_engine.get_status()
        current = affect_engine.current

        assert snapshot.valence < 0.0
        assert status["loneliness"] == 60
        assert status["longing"] == 50
        assert status["confused"] == 40
        assert status["curiosity"] == 80
        assert status["frustration"] == 70
        assert status["experiential"]["curiosity"] == pytest.approx(0.8)
        assert current.curiosity == pytest.approx(0.8)
        assert current.frustration == pytest.approx(0.7)
        assert affect_engine._raw_state["curiosity_metric"] == pytest.approx(80.0)
        assert affect_engine._raw_state["frustration_metric"] == pytest.approx(70.0)

    @pytest.mark.asyncio
    async def test_legacy_update_moves_explicit_driver_dimensions(self, affect_engine):
        affect_engine.markers.emotions["curiosity"] = 0.1
        affect_engine.markers.emotions["anticipation"] = 0.1
        affect_engine.markers.emotions["interest"] = 0.1
        affect_engine.markers.emotions["frustration"] = 0.1
        affect_engine.markers.emotions["anger"] = 0.1
        affect_engine.markers.emotions["upset"] = 0.1

        await affect_engine.update(delta_curiosity=0.2, delta_frustration=0.3)

        assert affect_engine.markers.emotions["curiosity"] == pytest.approx(0.3)
        assert affect_engine.markers.emotions["anticipation"] == pytest.approx(0.3)
        assert affect_engine.markers.emotions["interest"] == pytest.approx(0.2)
        assert affect_engine.markers.emotions["frustration"] == pytest.approx(0.4)
        assert affect_engine.markers.emotions["anger"] == pytest.approx(0.4)
        assert affect_engine.markers.emotions["upset"] == pytest.approx(0.25)

    def test_distress_diagnostic_does_not_rewrite_observed_state(self, affect_engine):
        affect_engine.markers.emotions["sadness"] = 0.95
        affect_engine.markers.emotions["fear"] = 0.85
        affect_engine.markers.emotions["joy"] = 0.0
        affect_engine.markers.emotions["upset"] = 0.8
        affect_engine.markers.emotions["frustration"] = 0.8
        affect_engine.markers.emotions["confused"] = 0.7
        affect_engine.markers.emotions["loneliness"] = 0.7
        affect_engine.markers.emotions["longing"] = 0.7

        before = dict(affect_engine.markers.emotions)
        diagnosis = affect_engine._check_for_despair_spiral()
        assert diagnosis["detected"] is True
        assert affect_engine.markers.emotions == before

    def test_appraisal_reads_the_stakes_rather_than_the_words(self):
        """This asserted the defect, in the words that caused it.

        It required "I am confused and unclear about this failure" to score
        negative, and it did — because the sentence contains confused, unclear
        and failure. That is a classifier, not an appraisal: an event with
        nothing at stake read as strongly negative for containing the word
        fail, and an event that broke a promise read as neutral if it was
        phrased calmly.

        Appraisal comes from core.interiority now, so the same words score on
        what they are about. Nothing held, nothing at stake: the sentence is
        no longer negative for its vocabulary. That is the change, and it is
        what this test measures.
        """
        from core.interiority.service import get_interiority

        words = "I am confused and unclear about this failure"
        indifferent = AffectEngineV2._heuristic_appraisal(words, {"intensity": 0.8})
        assert indifferent["v"] >= 0.0, (
            "the word scan is back: these words are negative to nobody who "
            "has nothing at stake in them"
        )

        # The same words, about something she is holding.
        service = get_interiority()
        service.ledger.bond("bryan", 0.9)
        service.ledger.promise(
            "p1", "review it", beneficiary="bryan", importance=0.9,
            concerns=("the piece",),
        )
        service.ledger.settle_promise("p1", kept=False)
        invested = service.appraise(words, {"intensity": 0.8, "subject": "bryan"})

        assert invested["v"] < indifferent["v"], (
            "identical words, and what she is committed to changed nothing"
        )

    def test_deep_alignment_and_bonding_emotions_integration(self):
        markers = DamasioMarkers()

        # Verify baselines exist and values match design
        alignment_baselines = {
            "gratitude": 0.08,
            "warmth": 0.10,
            "hope": 0.12,
            "vulnerability": 0.05,
            "nostalgia": 0.06,
            "satisfaction": 0.08,
            "empathy": 0.08,
            "belonging": 0.10,
            "amusement": 0.08,
            "inspiration": 0.10,
            "relief": 0.06,
            "admiration": 0.08,
        }

        wheel = markers.get_wheel()
        for emotion, expected_baseline in alignment_baselines.items():
            assert emotion in markers.emotions
            assert markers.mood_baselines[emotion] == pytest.approx(expected_baseline)
            assert emotion in wheel["experiential"]

        # Caller-controlled labels cannot manufacture complex social states.
        markers.somatic_update("positive_interaction", 1.0)
        assert markers.emotions["empathy"] == 0.0
        assert markers.emotions["belonging"] == 0.0
        assert markers.emotions["gratitude"] == 0.0

        markers.emotions["longing"] = 0.5
        markers.emotions["loneliness"] = 0.5
        markers.emotions["vulnerability"] = 0.4
        markers.emotions["belonging"] = 0.2
        markers.emotions["empathy"] = 0.2

        markers.somatic_update("interaction", 0.5)
        assert markers.emotions["longing"] == pytest.approx(0.5)
        assert markers.emotions["loneliness"] == pytest.approx(0.5)
        assert markers.emotions["vulnerability"] == pytest.approx(0.4)
        assert markers.emotions["belonging"] == pytest.approx(0.2)
        assert markers.emotions["empathy"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_bonding_emotions_influence_behavioral_modifiers(self):
        engine = AffectEngineV2()
        engine.markers.emotions["inspiration"] = 0.8
        engine.markers.emotions["amusement"] = 0.7
        engine.markers.emotions["belonging"] = 0.9
        engine.markers.emotions["empathy"] = 0.8
        engine.markers.emotions["relief"] = 0.7
        engine.markers.emotions["admiration"] = 0.6

        modifiers = await engine.get_behavioral_modifiers()
        assert modifiers["creativity"] > 1.3
        assert modifiers["risk_tolerance"] < 1.3
        assert modifiers["patience"] > 1.3
        assert modifiers["metacognition_depth"] > 1.3
        assert modifiers["persistence"] > 1.2
        assert modifiers["temporal_presence"] > 1.3


class TestQualiaCache:
    """Meta-qualia should cache per tick."""

    def test_anti_correlation_stays_inside_the_public_unit_interval(
        self, qualia_synth
    ):
        import numpy as np

        for index in range(8):
            vector = (
                np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
                if index % 2 == 0
                else np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            )
            qualia_synth._history.append(
                QualiaSnapshot(
                    q_vector=vector,
                    q_norm=float(np.linalg.norm(vector)),
                    pri=0.5,
                    ual_profile={},
                    is_attractor=False,
                    dominant_dimension="visual",
                    timestamp=time.time(),
                )
            )

        qualia_synth._tick = 42
        meta = qualia_synth.compute_meta_qualia()

        assert all(0.0 <= float(value) <= 1.0 for value in meta.values())
        assert meta["coherence"] < 0.5
        assert meta["dissonance"] > 0.0
        assert meta["dissonance"] <= 1.0

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
                timestamp=time.time(),
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
                timestamp=time.time(),
            )
            qualia_synth._history.append(state)

        qualia_synth._tick = 42
        meta1 = qualia_synth.compute_meta_qualia()

        # Advance tick
        qualia_synth._tick = 43
        meta2 = qualia_synth.compute_meta_qualia()

        assert meta1 is not meta2, "Cache did not invalidate on tick advance."


class TestQualiaEcho:
    """Qualia telemetry must not recursively validate/amplify affect."""

    def test_qualia_echo_records_without_amplifying_dominant(self, affect_engine):
        affect_engine.markers.emotions["joy"] = 0.6
        affect_engine.markers.emotions["fear"] = 0.1

        result = affect_engine.receive_qualia_echo(q_norm=0.8, pri=0.5, trend=0.0)
        assert affect_engine.markers.emotions["joy"] == pytest.approx(0.6)
        assert result["effect"] == "diagnostic_only_no_affect_amplification"

    def test_qualia_echo_trend_is_observation_only(self, affect_engine):
        affect_engine.markers.emotions["anticipation"] = 0.5
        affect_engine.markers.emotions["sadness"] = 0.5

        # Rising trend
        affect_engine.receive_qualia_echo(q_norm=0.5, pri=0.5, trend=0.1)
        assert affect_engine.markers.emotions["anticipation"] == pytest.approx(0.5)

        # Falling trend
        affect_engine.receive_qualia_echo(q_norm=0.5, pri=0.5, trend=-0.1)
        assert affect_engine.markers.emotions["sadness"] == pytest.approx(0.5)
        assert affect_engine.get_snapshot()["qualia_observation"]["trend"] == pytest.approx(-0.1)
