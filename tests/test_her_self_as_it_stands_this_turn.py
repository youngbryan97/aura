"""Her self as it stands this turn reaches K and the units that meet her self.

S held her self through averages and rates over hundreds of turns, and the
conjunctive codes read the self through the same slow fields, so the units that
meet feeling at attention and intention at recurrent cognition carried her drift
(23 October). The self-predictor now keeps its last comparison, K's S reads it,
and the codes read the self as that moment.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import core.consciousness.conjunctive_codes as codes
from core.consciousness.self_prediction import SelfPredictionLoop
from core.subject.state import SELF_NOW_PARTS, schema
from core.subject.state_readers import _self_now


def _loop() -> SelfPredictionLoop:
    return SelfPredictionLoop(orchestrator=None)


def test_the_self_predictor_keeps_its_last_comparison():
    loop = _loop()

    async def two_turns() -> None:
        await loop.tick(0.2, "curiosity", "user")
        await loop.tick(0.9, "social", "memory")

    asyncio.run(two_turns())
    moment = loop.get_snapshot()["moment"]
    assert set(moment) == set(SELF_NOW_PARTS)
    predicted = moment["valence_signed"]
    assert abs(abs(predicted) - moment["valence"]) < 1e-3
    assert moment["drive"] in (0.0, 1.0) and moment["focus"] in (0.0, 1.0)


def test_k_reads_her_self_now_column_by_column():
    features = schema("S").features
    assert [f"self_now_{part}" for part in SELF_NOW_PARTS] == [f for f in features if f.startswith("self_now_")]
    moment = {"valence_signed": -0.4, "valence": 0.4, "drive": 1.0, "focus": 0.0, "composite": 0.52}
    assert _self_now({"moment": moment}) == [moment[part] for part in SELF_NOW_PARTS]
    assert _self_now({}) == [0.0] * len(SELF_NOW_PARTS)


def test_the_units_read_the_self_as_that_moment(monkeypatch):
    moment = {"valence_signed": -0.4, "valence": 0.4, "drive": 1.0, "focus": 0.0, "composite": 0.52}
    snapshot = {"moment": moment, "confirmation": 0.25, "current_prediction": {"confidence": 0.7},
                "smoothed_error": 9.0, "confirmation_rate": 9.0}
    monkeypatch.setattr(codes, "_self_prediction", lambda: snapshot)
    state = SimpleNamespace(identity=SimpleNamespace(stability=9.0, bonding_level=9.0))
    self_reading = codes.readings(state)["S"]
    assert self_reading == [-0.4, 0.4, 1.0, 0.0, 0.52, 0.25, 0.7]
