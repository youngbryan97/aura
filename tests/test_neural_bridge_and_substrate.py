from __future__ import annotations

import concurrent.futures
import inspect

from core.runtime.errors import get_degradation_tracker


def test_neural_bridge_uses_telemetry_not_thought_decode():
    import core.senses.neural_bridge as nb

    source = inspect.getsource(nb.NeuralBridge._run_inference_loop)
    payload_source = inspect.getsource(nb.NeuralBridge._build_event_payload)
    assert "Thought Decoded" not in source
    assert "SIMULATED_NEURAL_TELEMETRY" in payload_source
    bridge = nb.NeuralBridge(lightweight_mode=True)
    bridge._calibrate()
    sample = bridge._generate_synthetic_eeg(6)
    idx, confidence, probabilities = bridge.model.predict(sample)
    assert 0 <= idx < len(nb.TELEMETRY_PATTERNS)
    assert confidence > 0
    assert len(probabilities) == len(nb.TELEMETRY_PATTERNS)
    status = bridge.get_status()
    assert status["simulated_not_thought_decoding"]
    assert status["last_thought"] is None


def test_neural_bridge_publish_failure_disables_broadcast_after_threshold():
    import core.senses.neural_bridge as nb

    tracker = get_degradation_tracker()
    tracker.reset()
    bridge = nb.NeuralBridge(lightweight_mode=True)
    bridge._max_broadcast_failures = 1
    bridge._event_bus = object()
    future: concurrent.futures.Future[None] = concurrent.futures.Future()
    future.set_exception(RuntimeError("publish path unavailable"))

    bridge._handle_broadcast_result(future)

    assert bridge._event_bus is None
    assert any(
        "disabled neural telemetry event broadcast" in record.action
        for record in tracker.recent(subsystem="neural_bridge")
    )
    tracker.reset()


def test_neural_bridge_stops_after_repeated_inference_failures():
    import core.senses.neural_bridge as nb

    tracker = get_degradation_tracker()
    tracker.reset()
    bridge = nb.NeuralBridge(lightweight_mode=True)
    bridge._poll_interval_range = (0.0, 0.0)
    bridge._max_consecutive_failures = 1
    bridge._is_running = True

    def fail_generation(_class_label):
        bridge._last_pattern = "SENSOR_FAILURE"
        raise RuntimeError("sensor math unavailable")

    bridge._generate_synthetic_eeg = fail_generation

    bridge._run_inference_loop()

    assert bridge._is_running is False
    assert bridge._stop_event.is_set()
    assert any(
        "stopped neural telemetry after repeated recoverable inference failures" in record.action
        for record in tracker.recent(subsystem="neural_bridge")
    )
    tracker.reset()


def test_substrate_default_dim_uses_env(monkeypatch):
    import core.consciousness.liquid_substrate as substrate

    monkeypatch.setenv("AURA_SUBSTRATE_DIM", "512")
    assert substrate.SubstrateConfig().neuron_count == 512
    monkeypatch.setenv("AURA_SUBSTRATE_DIM", "999999")
    assert substrate.SubstrateConfig().neuron_count == 4096
    monkeypatch.setenv("AURA_SUBSTRATE_DIM", "bad")
    assert substrate.SubstrateConfig().neuron_count == 512
