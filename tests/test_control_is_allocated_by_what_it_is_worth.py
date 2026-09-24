"""The executive tier's gain follows what control is worth: what is at stake times whether it works.

Her drives said what was at stake and her model of herself said whether steering
herself works, and the executive tier of her mesh ran at one gain whatever
either said. Control is allocated by the product of the two (Shenhav, Botvinick
and Cohen 2013; Frömer et al. 2021), so the gain is set from their product
against her ordinary value.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.consciousness import control_allocation
from core.consciousness.control_allocation import ControlAllocation, allocate_control, at_stake

pytestmark = pytest.mark.unit


def test_what_is_at_stake_is_the_most_pressing_urgency_as_decided() -> None:
    held = [{"urgency": 0.9, "decided_urgency": 0.4}, {"urgency": 0.6}, "not a goal", {"name": "no urgency"}]
    assert at_stake(held) == pytest.approx(0.6)
    assert at_stake([]) == 0.0


def _after_an_ordinary_life(value_pairs=((0.5, 0.5),) * 50) -> ControlAllocation:
    allocation = ControlAllocation()
    for stake, works in value_pairs:
        allocation.allocate(stake, works)
    return allocation


def test_an_ordinary_moment_is_gain_one_and_the_gain_saturates_below_two() -> None:
    allocation = _after_an_ordinary_life()
    assert allocation.allocate(0.5, 0.5) == pytest.approx(1.0, abs=1e-6)
    assert 1.0 < allocation.allocate(1.0, 1.0) < 2.0
    assert allocation.allocate(0.0, 1.0) == 0.0


def test_stake_raises_the_gain_only_as_far_as_control_works() -> None:
    """The interaction: the same rise in what is at stake, at two efficacies."""

    def rise(works: float) -> float:
        low = _after_an_ordinary_life().allocate(0.2, works)
        high = _after_an_ordinary_life().allocate(0.9, works)
        return high - low

    assert rise(0.9) > rise(0.2) > 0.0


class _Mesh:
    def __init__(self) -> None:
        self.regional: dict = {}
        self.global_state: tuple = ()

    def set_regional_modulation(self, multipliers):
        self.regional = dict(multipliers)

    def set_modulatory_state(self, gain, plasticity, noise):
        self.global_state = (gain, plasticity, noise)


def _state(urgency: float) -> SimpleNamespace:
    return SimpleNamespace(cognition=SimpleNamespace(active_goals=[{"urgency": urgency}], pending_initiatives=[]))


def test_the_gain_reaches_the_executive_tier(monkeypatch) -> None:
    monkeypatch.setattr(control_allocation, "efficacy", lambda: 0.8)
    mesh = _Mesh()
    allocation = ControlAllocation()
    allocate_control(mesh, allocation, _state(0.5))
    gain = allocate_control(mesh, allocation, _state(0.9))
    assert gain is not None and gain > 1.0
    assert mesh.regional == {"executive": (gain, 1.0)}


def test_without_a_model_of_herself_nothing_is_set(monkeypatch) -> None:
    monkeypatch.setattr(control_allocation, "efficacy", lambda: None)
    mesh = _Mesh()
    assert allocate_control(mesh, ControlAllocation(), _state(0.9)) is None
    assert mesh.regional == {}


def test_the_neurochemical_push_carries_it(monkeypatch) -> None:
    from core.consciousness.neurochemical_system import NeurochemicalSystem

    monkeypatch.setattr(control_allocation, "efficacy", lambda: 0.8)
    monkeypatch.setattr(control_allocation, "current_state", lambda: _state(0.7))
    system = NeurochemicalSystem()
    mesh = _Mesh()
    system._mesh_ref = mesh
    system._push_modulation()
    assert mesh.global_state and "executive" in mesh.regional
