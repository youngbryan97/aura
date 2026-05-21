import threading
import numpy as np
import pytest

from core.pneuma.precision_engine import PrecisionEngine, PrecisionConfig


def test_precision_engine_accepts_feedback():
    engine = PrecisionEngine()
    initial_v = engine.fhn.state.v

    # High surprise should depolarize/excite (increase v)
    engine.accept_inference_feedback(surprise=2.0, coherence=0.0)
    assert engine.fhn.state.v > initial_v

    # Positive coherence should stabilize (pull v back towards negative resting potential or reduce tension)
    v_excited = engine.fhn.state.v
    engine.accept_inference_feedback(surprise=0.0, coherence=1.0)
    assert engine.fhn.state.v < v_excited


def test_precision_engine_voltage_boundaries():
    engine = PrecisionEngine()

    # Extreme surprise shouldn't blow up state.v past 3.0
    for _ in range(50):
        engine.accept_inference_feedback(surprise=10.0, coherence=0.0)
    assert engine.fhn.state.v == 3.0

    # Extreme coherence shouldn't push state.v below -3.0
    for _ in range(50):
        engine.accept_inference_feedback(surprise=0.0, coherence=10.0)
    assert engine.fhn.state.v == -3.0


def test_feedback_impacts_outputs():
    engine = PrecisionEngine()

    # Reset/verify state
    engine.fhn.state.v = 0.0
    weights_neutral = engine.get_head_weights()
    temp_neutral = engine.get_temperature()

    # Apply positive feedback (surprise = 0.0, coherence = 1.0) -> v goes negative
    engine.accept_inference_feedback(surprise=0.0, coherence=2.0)
    # state.v decreases -> arousal decreases -> temperature increases (mapping is 0.95 - 0.40 * arousal)
    temp_post_positive = engine.get_temperature()
    assert temp_post_positive > temp_neutral

    # Apply exciting surprise feedback -> v goes positive -> arousal increases -> temperature decreases
    engine.accept_inference_feedback(surprise=3.0, coherence=0.0)
    temp_post_surprise = engine.get_temperature()
    assert temp_post_surprise < temp_post_positive


def test_precision_engine_thread_safety():
    engine = PrecisionEngine()

    # Create 50 threads running concurrent feedback calls
    def run_feedback():
        for _ in range(100):
            engine.accept_inference_feedback(surprise=1.5, coherence=-0.5)
            engine.get_head_weights()
            engine.get_temperature()
            engine.get_state_dict()

    threads = [threading.Thread(target=run_feedback) for _ in range(50)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # State dict must be intact and valid
    state = engine.get_state_dict()
    assert -3.0 <= state["fhn_v"] <= 3.0
    assert not np.isnan(state["head_weights_mean"])
