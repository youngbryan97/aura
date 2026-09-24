"""Her confidence in her model of herself is scored against predicting no change.

Confidence was one less a composite error that weighs valence on the scale of
-1 to 1. Her valence moves about 0.02 a turn, so on seed 7 confidence sat at 1.0
at the median while her valence error was 1.03 times what predicting no change
would have missed by.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.consciousness.self_prediction import SelfPredictionLoop

pytestmark = pytest.mark.unit


def _loop() -> SelfPredictionLoop:
    return SelfPredictionLoop(orchestrator=None)


def test_skill_is_one_less_her_error_over_what_no_change_missed() -> None:
    loop = _loop()
    loop._persistence_error_ema = {"valence": 0.02, "drive": 0.0, "focus": 0.5}
    loop._valence_error_ema, loop._drive_error_ema, loop._focus_error_ema = 0.02, 0.0, 0.25
    # Valence no better than no change (0), focus twice as good (0.5); the
    # drive never moved and is left out: (0.3 * 0 + 0.3 * 0.5) / 0.6.
    assert loop._skill_confidence() == pytest.approx(0.25)


def test_worse_than_no_change_is_no_confidence_and_nothing_moving_keeps_the_old_reading() -> None:
    loop = _loop()
    loop._persistence_error_ema = {"valence": 0.01, "drive": 0.0, "focus": 0.0}
    loop._valence_error_ema = 0.03
    assert loop._skill_confidence() == 0.0
    still = _loop()
    still._smoothed_error = 0.2
    assert still._skill_confidence() == pytest.approx(0.8)


def test_a_self_that_drifts_is_not_predicted_with_certainty() -> None:
    """A random walk of her size, with her own predictor: the old confidence said near-certain."""
    loop = _loop()
    rng = np.random.default_rng(7)
    valence = 0.4

    async def run() -> None:
        nonlocal valence
        for _ in range(120):
            valence = float(np.clip(valence + rng.normal(0.0, 0.02), -1.0, 1.0))
            await loop.tick(valence, "curiosity", "affect_joy")

    asyncio.run(run())
    old = 1.0 - loop._smoothed_error
    new = loop.get_current_prediction().confidence
    assert old > 0.99
    assert new < 0.6, "a model no better than no change is not confident"
