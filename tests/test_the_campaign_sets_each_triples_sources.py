"""The campaign's v6 stage forks, displaces and reads the targets as the preregistration says.

A stand-in organism whose attention moves by the product of feeling and self
gives A,S -> G an interaction the stage finds; the same product written only
into a conjunction register of G is not found, because registers are out of
the target. Forks of the stand-in are deterministic, as hers are.
"""

from __future__ import annotations

import asyncio
import copy

import numpy as np

import core.subject.perturbation as perturbation
from core.subject.interventional_synergy import collect, summarise, target_columns
from core.subject.paired_displacement import displacement_metric
from core.subject.state import domain_slices, feature_names


class _Frame:
    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector

    def vector(self) -> np.ndarray:
        return self._vector


class _Organism:
    """Displacements accumulate in `state`; a turn writes the target's response into its frame."""

    def __init__(self, *, product_in: str, seed: int = 0) -> None:
        self.names = list(feature_names())
        self.slices = domain_slices()
        self.state: dict = {"displaced": {}, "moment": 0}
        self.ontogeny = None
        self.organs = None
        rng = np.random.default_rng(seed)
        self.base = rng.normal(size=len(self.names))
        g = self.slices["G"]
        own = [i for i in range(g.start, g.stop) if ".conj_" not in self.names[i]]
        registers = [i for i in range(g.start, g.stop) if ".conj_" in self.names[i]]
        self.direction = np.zeros(len(self.names))
        chosen = own[:3] if product_in == "own" else registers[:3]
        assert chosen, "G has no columns of this kind"
        self.direction[chosen] = 1.0
        self.first = np.zeros(len(self.names))
        self.first[own[3:6]] = 1.0

    def snapshot(self):
        return copy.deepcopy(self.state)

    def restore(self, snapshot) -> None:
        self.state = copy.deepcopy(snapshot)

    def freeze_host(self) -> None:
        pass

    def thaw_host(self) -> None:
        pass

    async def turn_once(self, condition, perturb_at=None, perturb=None, sustain=None):
        if perturb is not None and perturb_at is not None:
            await perturb(self)
        self.state["moment"] += 1
        moved = self.state["displaced"]
        a, s = moved.get("A", 0.0), moved.get("S", 0.0)
        shift = 0.1 * self.state["moment"] * np.ones(len(self.names))
        vector = self.base + shift + a * self.first + s * self.first[::-1] + 40.0 * a * s * self.direction
        return [_Frame(vector)]


def _displace(state, domain, delta, *, ontogeny=None) -> bool:
    state["displaced"][domain] = state["displaced"].get(domain, 0.0) + delta
    return True


async def _no_organs(*args, **kwargs) -> bool:
    return False


def _run(organism: _Organism) -> dict:
    names = organism.names
    targets = {y: target_columns(names, organism.slices[y]) for y in ("G", "W", "D", "C")}
    rng = np.random.default_rng(1)
    baseline = organism.base + rng.normal(size=(200, len(names)))
    metrics = {y: displacement_metric(baseline[:, cols]) for y, cols in targets.items()}
    held = asyncio.run(collect(
        organism, ["work", "rest"], metrics=metrics, targets=targets, trials=6, turns=1, delta=0.15, seed=3,
    ))
    return {key: summarise(samples, seed=3, draws=2000) for key, samples in held.items()}


def test_a_product_in_the_targets_own_columns_is_found(monkeypatch):
    monkeypatch.setattr(perturbation, "perturb", lambda state, domain, delta, ontogeny=None: _displace(state, domain, delta))
    monkeypatch.setattr(perturbation, "perturb_organs", _no_organs)
    read = _run(_Organism(product_in="own"))
    assert read["A,S->G"]["sham"] == 0.0, "forks from one snapshot are identical"
    assert read["A,S->G"]["passes"], read["A,S->G"]
    assert not read["P,M->W"]["passes"]


def test_a_product_written_only_into_a_register_is_not(monkeypatch):
    monkeypatch.setattr(perturbation, "perturb", lambda state, domain, delta, ontogeny=None: _displace(state, domain, delta))
    monkeypatch.setattr(perturbation, "perturb_organs", _no_organs)
    read = _run(_Organism(product_in="register"))
    assert not read["A,S->G"]["passes"]
