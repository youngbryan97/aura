"""Self and intention enter the substrate's own transition, and cost it nothing it needs.

On seed 7 the meeting of her self and what she means to do moved recurrent
cognition through sixteen of 512 neurons, by setting their values, and S,D -> C
sat at 0.0118 against a bar of 0.0132. The units now enter every neuron as an
input current: dx/dt = -x + tanh(W x + B phi), B fixed, phi where the units
stand against their own range, scaled to the strength of the recurrence.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from core.consciousness import conjunctive_codes as codes
from core.consciousness.conjunctive_codes import bind, standing_of, substrate_current
from core.state.aura_state import AuraState


class _Held:
    """Somewhere for the current to keep its running power, as the substrate does."""

    def __init__(self, drive=None):
        self.conjunction_drive = drive


def test_nothing_bound_or_switched_off_puts_nothing_in(monkeypatch):
    recurrent = np.ones(64)
    assert substrate_current(_Held(), recurrent) is None
    assert substrate_current(_Held(()), recurrent) is None
    monkeypatch.setenv(codes.DISABLE_ENV, "1")
    assert substrate_current(_Held((1.0, 0.0, 0.0, 0.0)), recurrent) is None


def test_the_current_is_linear_in_where_the_units_stand():
    recurrent = np.random.default_rng(0).standard_normal(256)
    phi = np.array([0.7, -1.2, 0.3, 1.9])
    one = substrate_current(_Held(tuple(phi)), recurrent)
    two = substrate_current(_Held(tuple(2.0 * phi)), recurrent)
    rest = substrate_current(_Held((0.0,) * 4), recurrent)
    assert np.allclose(two, 2.0 * one)
    assert np.allclose(rest, 0.0)


def test_a_typical_standing_enters_at_the_strength_of_the_recurrence():
    rng = np.random.default_rng(1)
    recurrent = rng.standard_normal(512) * 0.3
    sizes = []
    for _ in range(200):
        phi = rng.standard_normal(4)
        current = substrate_current(_Held(tuple(phi)), recurrent)
        sizes.append(np.mean(current * current))
    assert np.sqrt(np.mean(sizes)) == pytest.approx(np.sqrt(np.mean(recurrent**2)), rel=0.1)


def _lived(seed: int = 0) -> AuraState:
    state = AuraState.default()
    rng = np.random.default_rng(seed)
    for _ in range(40):
        state.identity.stability = float(rng.uniform())
        state.identity.bonding_level = float(rng.uniform())
        for name in ("social", "curiosity", "growth"):
            state.motivation.budgets[name]["level"] = float(rng.uniform(20, 90))
        state.affect.valence = float(rng.normal(0.0, 0.3))
        bind(state)
    return state


def _moved(state: AuraState, *, self_moved: bool, drives_moved: bool) -> tuple[float, ...]:
    copy_of = copy.deepcopy(state)
    if self_moved:
        copy_of.identity.stability = 1.0 - copy_of.identity.stability
        copy_of.identity.bonding_level = 1.0 - copy_of.identity.bonding_level
    if drives_moved:
        for name in ("social", "curiosity", "growth"):
            level = copy_of.motivation.budgets[name]["level"]
            copy_of.motivation.budgets[name]["level"] = 110.0 - level
    bind(copy_of)
    return standing_of(copy_of.conjunctions, "recurrent")


def test_four_copies_self_moved_drives_moved_both_and_neither():
    state = _lived()
    neither = np.array(_moved(state, self_moved=False, drives_moved=False))
    self_only = np.array(_moved(state, self_moved=True, drives_moved=False))
    drives_only = np.array(_moved(state, self_moved=False, drives_moved=True))
    both = np.array(_moved(state, self_moved=True, drives_moved=True))
    assert not np.allclose(self_only, neither)
    assert not np.allclose(drives_only, neither)
    # The units multiply: what both moving does is not the sum of each alone.
    assert np.linalg.norm(both - self_only - drives_only + neither) > 1e-6
    recurrent = np.random.default_rng(2).standard_normal(512) * 0.2
    currents = {
        name: substrate_current(_Held(tuple(phi)), recurrent)
        for name, phi in (("neither", neither), ("self", self_only), ("drives", drives_only), ("both", both))
    }
    assert not np.allclose(currents["self"], currents["neither"])
    assert not np.allclose(currents["drives"], currents["neither"])


def _substrate(tmp_path, n: int = 128):
    from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

    return LiquidSubstrate(config=SubstrateConfig(neuron_count=n, state_file=tmp_path / "substrate.npy"))


def _as_she_runs(substrate, phis, mesh, projection):
    """Twenty ticks a turn, the consciousness bridge blending the mesh in at 0.35 every other tick."""
    torch.manual_seed(5)
    out = []
    for turn in range(len(mesh)):
        substrate.conjunction_drive = None if phis is None else tuple(phis[turn])
        for tick in range(20):
            substrate._step_torch_math(0.1)
            substrate._learn_from_this_tick_sync()
            if tick % 2:
                with substrate.sync_lock:
                    substrate.x = np.clip(substrate.x * 0.65 + np.tanh(projection @ mesh[turn] * 0.02) * 0.35, -1, 1)
        out.append(np.array(substrate.x, copy=True))
    return np.array(out)


def _decoded(states, target):
    """Held-out R squared of a ridge readout, per target column."""
    half = len(states) // 2
    a, b = states[:half], states[half:]
    mean = a.mean(axis=0)
    a, b = a - mean, b - mean
    ridge = 1e-2 * np.trace(a.T @ a) / a.shape[1]
    weights = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T @ (target[:half] - target[:half].mean(axis=0)))
    predicted = b @ weights + target[:half].mean(axis=0)
    truth = target[half:]
    return np.array([max(0.0, 1.0 - np.mean((truth[:, j] - predicted[:, j]) ** 2) / np.var(truth[:, j])) for j in range(truth.shape[1])])


def test_in_her_regime_the_substrate_carries_self_and_intention_and_loses_nothing_the_mesh_tells_it(tmp_path):
    """The bridge holds her substrate near the mesh's projection, so it is judged there.

    Without the current the substrate carries nothing of where self and
    intention stand. With it, it carries that, while what the mesh tells it is
    read out as well as before, and no neuron is pinned.
    """
    rng = np.random.default_rng(4)
    turns = 160
    projection = rng.standard_normal((128, 8)) / np.sqrt(8)
    mesh = rng.standard_normal((turns, 8))
    phis = rng.standard_normal((turns, 4))
    off = _as_she_runs(_substrate(tmp_path / "off"), None, mesh, projection)
    on = _as_she_runs(_substrate(tmp_path / "on"), phis, mesh, projection)
    mesh_off, mesh_on = _decoded(off, mesh), _decoded(on, mesh)
    error = np.std(mesh_off) / np.sqrt(mesh_off.size)
    assert mesh_on.mean() >= mesh_off.mean() - 2.0 * error
    carried_off, carried_on = _decoded(off, phis), _decoded(on, phis)
    error = max(np.std(carried_on), np.std(carried_off)) / np.sqrt(carried_on.size)
    assert carried_on.mean() > carried_off.mean() + 2.0 * error
    assert np.mean(np.abs(on) > 0.95) <= np.mean(np.abs(off) > 0.95)


def test_four_forks_leave_an_interaction_in_her_later_state_and_none_with_the_current_off(tmp_path, monkeypatch):
    """Neither moved, self moved, intention moved, both: the contrast in the substrate turns later.

    A decoder reading where the units stand is not enough; what both moving does
    to her later state has to differ from the sum of each alone, and with the
    current switched off the same four forks must leave no such difference.
    """
    state = _lived()
    phis = {
        (s, d): np.array(_moved(state, self_moved=s, drives_moved=d))
        for s in (False, True) for d in (False, True)
    }
    rng = np.random.default_rng(6)
    projection = rng.standard_normal((128, 8)) / np.sqrt(8)
    mesh = rng.standard_normal((6, 8))

    def later(phi, label):
        substrate = _substrate(tmp_path / label)
        return _as_she_runs(substrate, [phi] * 6, mesh, projection)[-1]

    def contrast(prefix):
        x = {key: later(phi, f"{prefix}{key}") for key, phi in phis.items()}
        return x[(True, True)] - x[(True, False)] - x[(False, True)] + x[(False, False)]

    on = contrast("on")
    # Two forks with nothing between them still differ in the last float
    # bits (the gain hold's power iteration starts from a random vector).
    floor = later(phis[(False, False)], "again") - later(phis[(False, False)], "once more")
    monkeypatch.setenv(codes.DISABLE_ENV, "1")
    off = contrast("off")
    # Measured 21 October: 0.145 on, 8.6e-6 off, against 1.4e-4 between
    # identical forks. Two orders of magnitude either side is the claim.
    assert np.linalg.norm(on) > 100 * np.linalg.norm(floor)
    assert np.linalg.norm(off) < np.linalg.norm(on) / 100


def test_in_her_regime_the_driven_substrate_forgets_a_displacement(tmp_path):
    """Two copies a small step apart, given the same inputs, come together: the turn map contracts."""
    rng = np.random.default_rng(8)
    projection = rng.standard_normal((128, 8)) / np.sqrt(8)
    mesh = rng.standard_normal((8, 8))
    phis = rng.standard_normal((8, 4))
    a, b = _substrate(tmp_path / "a"), _substrate(tmp_path / "b")
    b.x = np.clip(b.x + 0.05 * rng.standard_normal(b.x.shape), -1, 1)
    start = float(np.linalg.norm(a.x - b.x))
    ends = [_as_she_runs(s, phis, mesh, projection)[-1] for s in (a, b)]
    assert float(np.linalg.norm(ends[0] - ends[1])) < start
