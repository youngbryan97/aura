import numpy as np

from core.pneuma.neural_ode_flow import BeliefFlowNetwork, NeuralODEFlow
from core.pneuma.pneuma import PNEUMA


def test_belief_flow_hebbian_update_limits_touched_rows():
    net = BeliefFlowNetwork(dim=6, hidden=4, hebbian_top_k=2)
    before = net.W2.copy()

    b_pre = np.array([0.2, -0.1, 0.3, 0.0, 0.1, -0.2], dtype=np.float32)
    b_post = np.array([0.0, 0.8, 0.0, 0.0, -0.4, 0.0], dtype=np.float32)
    net.hebbian_update(b_pre, b_post)

    changed_rows = np.where(np.any(np.abs(net.W2 - before) > 1e-9, axis=1))[0].tolist()
    assert changed_rows
    assert set(changed_rows) <= {1, 4}


def test_neural_ode_flow_throttles_hebbian_updates_between_ticks(monkeypatch):
    flow = NeuralODEFlow(dim=4)
    calls = []

    monkeypatch.setattr(flow.flow_net, "hebbian_update", lambda *_args, **_kwargs: calls.append("hebbian"))

    flow.step(0.1)
    flow.step(0.1)
    assert len(calls) == 1

    flow.step(0.1)
    assert len(calls) == 2


def test_pneuma_context_block_uses_short_cache(monkeypatch):
    pneuma = PNEUMA()
    calls = {"precision": 0}

    def _precision_state():
        calls["precision"] += 1
        return {"fhn_v": 0.1, "fhn_w": 0.2, "arousal": 0.3, "temperature": 0.7}

    monkeypatch.setattr(pneuma.precision, "get_state_dict", _precision_state)
    monkeypatch.setattr(pneuma.ode_flow, "get_state_dict", lambda: {"belief_norm": 0.4, "belief_confidence": 0.5})
    monkeypatch.setattr(pneuma.ig_tracker, "get_state_dict", lambda: {"stability": 0.6, "is_drifting": False})
    monkeypatch.setattr(pneuma.topo_memory, "get_state_dict", lambda: {"attractor_count": 2, "topological_complexity": 0.7})

    first = pneuma.get_context_block()
    second = pneuma.get_context_block()

    assert first == second
    assert calls["precision"] == 1
