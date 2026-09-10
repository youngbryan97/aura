"""A model of oneself built only from oneself cannot be wrong for a reason.

The self-prediction loop was a pure autoregression: the next valence was the
recency-weighted average of the last sixty, the next drive the most frequent
recent drive, the next focus the most frequent recent winner. Its error was
therefore a function of its own history and of nothing else, so nothing that
happened to her could predict how surprised she would be. Measured across the
ten cognitive domains of the subject-core battery, the self-state was the one
domain whose change no other domain helped predict at all.

These tests hold the repair: the situation enters the prediction, and the old
predictor is nested inside the new one so it cannot be made worse.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.consciousness.self_prediction import SelfPredictionLoop


def _loop() -> SelfPredictionLoop:
    return SelfPredictionLoop(orchestrator=None)


def test_the_old_predictor_is_the_first_feature_of_the_new_one() -> None:
    assert SelfPredictionLoop._SITUATION[0] == "recent_valence"


def test_before_enough_moments_it_predicts_exactly_what_it_used_to() -> None:
    loop = _loop()
    for value in (0.2, 0.3, 0.25):
        asyncio.run(loop.tick(value, "curiosity", "affect_joy"))
    history = list(loop._valence_history)
    weights = [i + 1 for i in range(len(history))]
    expected = sum(v * w for v, w in zip(history, weights, strict=True)) / sum(weights)
    assert loop.get_current_prediction() is not None
    assert loop.get_current_prediction().predicted_affect_valence == pytest.approx(
        round(expected, 3)
    )


def test_it_learns_that_the_situation_moves_the_feeling() -> None:
    """Valence that follows body pressure, and history that says nothing."""
    loop = _loop()
    rng = np.random.default_rng(3)
    valence = 0.0
    for _ in range(200):
        pressure = float(rng.random())
        # The next feeling is a function of the pressure now, and the sequence
        # of feelings alone carries no information about it.
        asyncio.run(
            loop.tick(
                valence,
                "curiosity",
                "affect_joy",
                situation={"body_pressure": pressure},
            )
        )
        valence = -0.8 * pressure + 0.1

    weights = loop._valence_weights()
    assert weights is not None, "two hundred moments and still no model"
    index = SelfPredictionLoop._SITUATION.index("body_pressure")
    assert weights[index] < -0.3, f"the body did not enter the self-model: {weights}"


def test_what_is_competing_predicts_what_she_will_attend_to() -> None:
    loop = _loop()
    for _ in range(5):
        asyncio.run(loop.tick(0.0, "curiosity", "affect_joy"))
    asyncio.run(
        loop.tick(0.0, "curiosity", "affect_joy", situation={"strongest_bid": "world_model"})
    )
    assert loop.get_current_prediction().predicted_focus_source == "world_model"


def test_the_pressing_drive_predicts_the_next_drive() -> None:
    loop = _loop()
    for _ in range(5):
        asyncio.run(loop.tick(0.0, "curiosity", "affect_joy"))
    asyncio.run(
        loop.tick(
            0.0,
            "curiosity",
            "affect_joy",
            situation={"dominant_drive": "social", "drive_urgency": 0.6},
        )
    )
    assert loop.get_current_prediction().predicted_dominant_drive == "social"


def test_the_heartbeat_hands_the_situation_over() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "consciousness" / "heartbeat.py"
    ).read_text()
    assert "situation={" in source, "the loop is predicting from itself again"
    for key in ("body_pressure", "world_surprise", "novelty", "strongest_bid"):
        assert key in source, f"{key} no longer reaches the self-model"
