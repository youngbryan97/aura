from __future__ import annotations

import inspect


def test_neural_bridge_uses_telemetry_not_thought_decode():
    import core.senses.neural_bridge as nb

    source = inspect.getsource(nb.NeuralBridge._run_inference_loop)
    assert "Thought Decoded" not in source
    assert "SIMULATED_NEURAL_TELEMETRY" in source
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


def test_substrate_default_dim_uses_env(monkeypatch):
    import core.consciousness.liquid_substrate as substrate

    monkeypatch.setenv("AURA_SUBSTRATE_DIM", "512")
    assert substrate.SubstrateConfig().neuron_count == 512
    monkeypatch.setenv("AURA_SUBSTRATE_DIM", "999999")
    assert substrate.SubstrateConfig().neuron_count == 4096
    monkeypatch.setenv("AURA_SUBSTRATE_DIM", "bad")
    assert substrate.SubstrateConfig().neuron_count == 512
