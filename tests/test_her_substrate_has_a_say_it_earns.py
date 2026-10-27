"""Her substrate scales decisions she already makes, by what it has learned pays.

Recurrent cognition reached the rest of her at 0.94% of its reverse gain on the
seed-7 validation at 23e596071. Each gate in core/consciousness/substrate_gates.py
is a multiplier exp(w . f + xi) on a decision that exists: f her substrate's own
readings in units of their spread, xi one free unit's deviation from its recent
mean, and w the running covariance of the turn's worth with that wandering.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.consciousness import substrate_gates
from core.consciousness.substrate_gates import READINGS, SubstrateGates


class _Substrate:
    def __init__(self, size: int = 16) -> None:
        self.x = np.zeros(size)
        for i, name in enumerate(READINGS):
            setattr(self, name, i)
        self._state_revision = 0

    def set(self, values: dict[int, float]) -> None:
        for index, value in values.items():
            self.x[index] = value
        self._state_revision += 1


def test_with_no_substrate_a_gate_changes_nothing(monkeypatch):
    monkeypatch.setattr(SubstrateGates, "_substrate", staticmethod(lambda: None))
    gates = SubstrateGates()
    assert gates.multiplier("recall") == 1.0
    assert gates.learn(1.0) == {}


def test_switched_off_a_gate_changes_nothing(monkeypatch):
    monkeypatch.setenv(substrate_gates.DISABLE_ENV, "1")
    substrate = _Substrate()
    substrate.set({len(READINGS): 0.9})
    assert SubstrateGates().multiplier("recall", substrate=substrate) == 1.0


def test_before_it_has_learned_a_gate_wanders_by_its_own_unit():
    gates = SubstrateGates()
    substrate = _Substrate()
    unit = len(READINGS)  # the first unit no reading is taken from
    substrate.set({unit: 0.0})
    assert gates.multiplier("recall", substrate=substrate) == pytest.approx(1.0)
    substrate.set({unit: 0.4})
    # The unit's running mean moved a half of the way (the second of its
    # values), so it now stands 0.2 above it.
    assert gates.multiplier("recall", substrate=substrate) == pytest.approx(math.exp(0.2))
    assert gates.status()["recall"]["unit"] == unit


def test_each_gate_wanders_by_a_different_unit():
    gates = SubstrateGates()
    substrate = _Substrate()
    substrate.set({})
    for name in ("recall", "curiosity", "social"):
        gates.multiplier(name, substrate=substrate)
    units = [gates.status()[name]["unit"] for name in ("recall", "curiosity", "social")]
    assert len(set(units)) == 3
    assert not set(units) & set(range(len(READINGS)))


def test_a_wander_that_came_with_better_turns_is_kept():
    """Valence high when the unit ran high, and those turns paid: w on valence grows."""
    rng = np.random.default_rng(3)
    gates = SubstrateGates()
    substrate = _Substrate()
    unit = len(READINGS)
    for _ in range(400):
        valence = float(rng.normal())
        wander = float(rng.normal())
        substrate.set({0: valence, unit: wander})
        gates.multiplier("recall", substrate=substrate)
        gates.learn(1.0 if valence * wander > 0 else -1.0)
    weights = gates.status()["recall"]["weights"]
    assert weights["idx_valence"] > 0.3
    assert abs(weights["idx_focus"]) < 0.1


def test_a_gate_nobody_consulted_is_not_taught():
    gates = SubstrateGates()
    substrate = _Substrate()
    substrate.set({len(READINGS): 0.5})
    gates.multiplier("recall", substrate=substrate)
    assert gates.learn(1.0) == {"recall": 1}
    assert gates.learn(1.0) == {}


def _fixed(monkeypatch, values: dict[str, float]) -> None:
    def multiplier(self, name, *, substrate=None):
        return values.get(name, 1.0)

    monkeypatch.setattr(SubstrateGates, "multiplier", multiplier)


def test_recall_asks_its_gate(monkeypatch):
    from core.phases import memory_retrieval

    captured = {}

    def resonance(now, felt, centre):
        captured.setdefault("calls", 0)
        captured["calls"] += 1
        return 0.5

    monkeypatch.setattr("core.memory.felt_at_encoding.felt_now", lambda affect: {"valence": 0.2})
    monkeypatch.setattr("core.memory.felt_at_encoding.resonance", resonance)
    candidates = [(0.2, "a memory")]
    _fixed(monkeypatch, {"recall": 1.0})
    plain = memory_retrieval._cued_by_what_she_feels(candidates, {"a memory": {}}, object(), 0.3)[0][0]
    _fixed(monkeypatch, {"recall": 2.0})
    lifted = memory_retrieval._cued_by_what_she_feels(candidates, {"a memory": {}}, object(), 0.3)[0][0]
    assert lifted > plain


def test_a_drive_grows_as_fast_as_its_gate_says(monkeypatch):
    import asyncio

    from core.phases.motivation_update import MotivationUpdatePhase
    from core.state.aura_state import AuraState

    def spent(values):
        _fixed(monkeypatch, values)
        state = AuraState.default()
        before = {name: budget["level"] for name, budget in state.motivation.budgets.items()}
        state.motivation.last_tick -= 60.0
        after = asyncio.run(MotivationUpdatePhase(None).execute(state, objective="a turn"))
        return before["social"] - after.motivation.budgets["social"]["level"]

    assert spent({"drive:social": 2.0}) > spent({"drive:social": 1.0})


def test_the_turns_worth_teaches_the_gates(monkeypatch):
    from core.affect import what_it_was_worth

    taught = {}
    monkeypatch.setattr(SubstrateGates, "learn", lambda self, dose: taught.setdefault("dose", dose) and {})
    reading = what_it_was_worth.Worth(measured=True, worth=1.0, size=0.5)
    what_it_was_worth.teach_connections(reading)
    assert taught["dose"] == pytest.approx(0.5)
