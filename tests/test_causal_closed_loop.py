import pytest
import numpy as np
from core.consciousness.closed_loop import OutputReceptor, SelfPredictiveCore
from core.container import ServiceContainer

class MockSubstrate:
    def __init__(self):
        self.x = np.zeros(64, dtype=np.float32)
    def inject_stimulus(self, delta, weight):
        self.x += delta * weight
        return None

def test_output_receptor_action_parsing_and_simulation():
    # Setup mock substrate in ServiceContainer
    sub = MockSubstrate()
    ServiceContainer.register("conscious_substrate", sub)
    
    receptor = OutputReceptor(neuron_count=64)
    
    # 1. Verbal text with action call
    text = "Aura suggests we: reroute_vessel(Vessel_Alpha, 270.0, 22.0)"
    res = receptor.receive_output(text)
    
    assert res is not None
    delta, magnitude = res
    assert magnitude > 0.0
    # Positive valence/arousal should have been set
    assert delta[0] > 0.0  # Valence
    assert delta[1] > 0.0  # Arousal
    assert delta[3] < 0.0  # Frustration should be reduced

def test_self_predictive_core_physical_free_energy():
    core = SelfPredictiveCore(neuron_count=64)
    core.predict(np.zeros(64, dtype=np.float32))
    
    # Expected vs actual physical sensors matches
    expectations = {
        "port_east_load": 800.0,
        "port_west_load": 400.0,
        "vessel_alpha_speed": 15.0,
        "warehouse_load": 200.0,
        "system_cpu_usage": 35.0
    }
    
    # Actual state
    actual_x = np.zeros(64, dtype=np.float32)
    
    cycle = core.observe_and_update(actual_x, simulated_expectations=expectations)
    assert cycle is not None
    # Since simulated matches actual physical values (we synced the real sensors), physical FE should be low/zero
    assert cycle.free_energy < 0.2
