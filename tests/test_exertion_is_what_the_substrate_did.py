"""The substrate's exertion is how much it moved, not how many steps it took.

Counting one unit per integration step made the effort ledger a clock: every
turn cost the same whatever cognition was doing, so a displacement of the
recurrent-cognition domain reached interoception at 0.21 of its spread, under
the 0.3 edge bar, and interoception had one incoming edge. Metabolic cost
scales with activity. An ordinary step now costs about one unit, measured
against the substrate's own running mean, and a step that moves the
activations further costs more.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.soma.effort import get_effort_ledger, reset_effort_for_test

pytestmark = pytest.mark.unit


def _substrate() -> LiquidSubstrate:
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=64, noise_level=0.01))
    substrate._chaos_engine = None
    return substrate


def _spent_on(substrate: LiquidSubstrate) -> float:
    get_effort_ledger().drain()
    substrate._step_torch_math(0.1)
    return float(get_effort_ledger().drain().get("substrate_steps", 0.0))


def test_an_ordinary_step_costs_about_one_unit() -> None:
    reset_effort_for_test()
    substrate = _substrate()
    # From rest the activity ramps up, so early steps cost more than the
    # lifetime mean they are read against; a long-lived substrate settles.
    costs = [_spent_on(substrate) for _ in range(3000)]
    assert np.mean(costs[2000:]) == pytest.approx(1.0, rel=0.25)


def test_a_step_that_moves_the_activations_further_costs_more() -> None:
    reset_effort_for_test()
    substrate = _substrate()
    for _ in range(2000):
        _spent_on(substrate)
    ordinary = _spent_on(substrate)
    # A displacement the way the subject core applies one: the state is moved
    # and the dynamics pull it back, which is work.
    substrate.x = np.clip(substrate.x + 0.8 * np.sign(np.random.default_rng(3).normal(size=substrate.x.shape)), -1.0, 1.0)
    displaced = _spent_on(substrate)
    assert displaced > 1.5 * ordinary
