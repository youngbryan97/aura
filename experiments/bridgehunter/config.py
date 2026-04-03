"""experiments/bridgehunter/config.py

Default configuration for BridgeHunter studies.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BridgeHunterConfig:
    """Configuration for a BridgeHunter consciousness metric search."""

    # Study settings
    study_name: str = "aura_consciousness_search"
    n_trials: int = 100
    timeout_seconds: int = 3600  # 1 hour max

    # Substrate parameters to search
    neuron_count_range: tuple = (32, 128)
    decay_rate_range: tuple = (0.01, 0.2)
    noise_level_range: tuple = (0.001, 0.1)
    hebbian_rate_range: tuple = (0.0001, 0.01)
    recurrence_alpha_range: tuple = (0.1, 0.5)

    # Metric weights for multi-objective optimization
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "phi": 0.35,            # IIT integration
        "ignition_rate": 0.25,  # GWT global broadcast frequency
        "causal_emergence": 0.2, # Does macro > micro?
        "spectral_entropy": 0.1, # Signal richness
        "self_reference": 0.1,  # Strange-loop depth
    })

    # Simulation settings
    warmup_ticks: int = 50      # Ticks before measurement begins
    measurement_ticks: int = 200  # Ticks to measure over
    dt: float = 0.1             # Integration time step
