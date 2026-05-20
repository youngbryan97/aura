import pytest
import numpy as np
from core.adaptation.adaptive_immunity import (
    AdaptiveImmuneSystem,
    AdaptiveImmuneConfig,
    ImmuneCell,
    CellKind,
    Antigen,
    _mutate_behavioral_rule,
    _evaluate_causal_fitness
)

def test_rule_mutation():
    rng = np.random.default_rng(42)
    rule = _mutate_behavioral_rule(None, rng)
    assert "conditions" in rule
    assert "actions" in rule
    
    mutated = _mutate_behavioral_rule(rule, rng)
    assert "conditions" in mutated
    assert "actions" in mutated

def test_causal_fitness_evaluation():
    rule = {
        "conditions": [
            {"sensor": "port_east_load", "operator": ">", "value": 500.0}
        ],
        "actions": [
            {
                "actuator": "reallocate_flow",
                "params": {
                    "source_id": "Port_East",
                    "target_id": "Port_West",
                    "amount": 100.0
                }
            }
        ]
    }
    fitness = _evaluate_causal_fitness(rule)
    # Valid execution should result in positive fitness (latency reduction or baseline value)
    assert fitness >= 0.0

def test_coevolution_lab_evolves_rules(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=12, max_population=24)
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=42)
    
    # Verify seeded population has rules for B cells
    b_cells = [cell for cell in immune._cells if cell.kind == CellKind.B]
    assert len(b_cells) > 0
    for cell in b_cells:
        assert cell.behavioral_rule is not None
        assert "conditions" in cell.behavioral_rule
        
    # Run dream consolidation to trigger evolution lab
    result = immune.dream_consolidate()
    assert result["population"] > 0
    
    # Assert that survivors still carry valid mutated rules
    for cell in immune._cells:
        if cell.kind in {CellKind.B, CellKind.MEMORY}:
            assert cell.behavioral_rule is not None
