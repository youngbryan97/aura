"""tests/test_transition_model.py
================================
Unit and mathematical convergence tests for core/world_model/transition_model.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.container import ServiceContainer
from core.world_model.transition_model import get_transition_model, TransitionModel


class MockSubstrate:
    """Mock LiquidSubstrate for state vector extraction."""
    def __init__(self) -> None:
        self.sync_lock = threading_lock = pytest.importorskip("threading").Lock()
        self.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 50, dtype=np.float32)
        self.idx_valence = 0
        self.idx_arousal = 1
        self.idx_frustration = 2
        self.idx_curiosity = 3
        self.idx_focus = 4


class MockFreeEnergy:
    """Mock FreeEnergyEngine for state vector extraction."""
    def __init__(self) -> None:
        self.smoothed_fe = 0.65
        self.surprise = 0.0

    def accept_surprise_signal(self, surprise: float) -> None:
        self.surprise = surprise


class MockFHN:
    """Mock FitzHugh-Nagumo state."""
    def __init__(self) -> None:
        self.arousal = 0.72


class MockPrecisionEngine:
    """Mock PrecisionEngine for state vector extraction."""
    def __init__(self) -> None:
        self.fhn = MockFHN()


def test_transition_model_singleton() -> None:
    """Verifies TransitionModel acts as a proper singleton."""
    model1 = get_transition_model()
    model2 = get_transition_model()
    assert model1 is model2
    assert isinstance(model1, TransitionModel)


def test_state_vector_extraction() -> None:
    """Verifies that TransitionModel extracts substrate state dimensions correctly."""
    # Register mock services in ServiceContainer
    sub = MockSubstrate()
    fe = MockFreeEnergy()
    prec = MockPrecisionEngine()
    
    ServiceContainer.register_instance("liquid_substrate", sub)
    ServiceContainer.register_instance("free_energy_engine", fe)
    ServiceContainer.register_instance("precision_engine", prec)
    
    model = TransitionModel()
    state = model.extract_state_vector()
    
    assert state.shape == (7,)
    assert state[0] == 0.1  # valence
    assert state[1] == 0.2  # arousal
    assert state[2] == 0.3  # frustration
    assert state[3] == 0.4  # curiosity
    assert state[4] == 0.5  # focus
    assert state[5] == 0.65  # free energy
    assert state[6] == 0.72  # FHN arousal


def test_predict_and_clamping() -> None:
    """Verifies that predicted states remain within physical boundaries [0.0, 1.0]."""
    model = TransitionModel()
    current_state = np.array([0.9, 0.9, 0.1, 0.1, 0.5, 0.5, 0.5], dtype=np.float32)
    action_vec = np.zeros(8, dtype=np.float32)
    action_vec[0] = 1.0  # file_read
    
    # Force extreme positive weight matrix to test upper clamping
    model.W = np.ones((7, 15), dtype=np.float32) * 5.0
    predicted = model.predict_next_state(current_state, action_vec)
    
    assert np.all(predicted <= 1.0)
    assert np.all(predicted >= 0.0)
    
    # Force extreme negative weight matrix to test lower clamping
    model.W = np.ones((7, 15), dtype=np.float32) * -5.0
    predicted_neg = model.predict_next_state(current_state, action_vec)
    assert np.all(predicted_neg >= 0.0)
    assert np.all(predicted_neg <= 1.0)


def test_lms_delta_learning_convergence() -> None:
    """Mathematical test demonstrating that the online Delta Rule weight updates reduce prediction errors."""
    sub = MockSubstrate()
    fe = MockFreeEnergy()
    prec = MockPrecisionEngine()
    
    ServiceContainer.register_instance("liquid_substrate", sub)
    ServiceContainer.register_instance("free_energy_engine", fe)
    ServiceContainer.register_instance("precision_engine", prec)
    
    # Create isolated model with high learning rate to accelerate convergence
    model = TransitionModel(learning_rate=0.2)
    
    errors = []
    
    # Simulate repeated transitions where executing 'file_write' predictably drops arousal and frustration
    for step in range(10):
        # Update mock states dynamically based on transition
        sub.x[sub.idx_arousal] = max(0.0, 0.8 - 0.05 * step)
        sub.x[sub.idx_frustration] = max(0.0, 0.7 - 0.06 * step)
        
        err = model.process_step("file_write")
        if step > 0:  # Skip first step because it has no past prediction to update from
            errors.append(err)
            
    # Check that error is generally declining or converges to a low value
    assert len(errors) == 9
    assert errors[-1] < errors[0]  # Empirical proof of online predictive convergence!
