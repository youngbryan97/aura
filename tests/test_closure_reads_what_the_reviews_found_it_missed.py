"""What six reviews of the closure gate found it could not see, 25 October 2026.

An environmental proxy read as a leak because E was never conditioned on; a
broker kept as a value in a dict of fixed size was invisible to the reader;
and a carrier the gate could not decide was reported as not found.
"""
from __future__ import annotations

import numpy as np

from core.subject.closure import _numbers, _resolution, _scored
from core.subject.v25_exclusion import select_carriers


def _environment_world(rows: int = 900, seed: int = 0):
    """K's next state is driven by the environment; P only echoes it. No P -> K arrow."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, rows + 1)
    k = np.zeros((rows + 1, 2))
    for t in range(rows):
        k[t + 1] = 0.8 * k[t] + np.array([1.0, 0.5]) * e[t + 1] + rng.normal(0, 0.1, 2)
    p = (e + rng.normal(0, 0.05, rows + 1))[:, None]
    # The input driving t -> t+1 is the one on row t+1, as the recording keeps it.
    return k[:-1], e[1:, None], k[1:], p[1:]


def test_an_echo_of_the_environment_is_a_leak_only_when_the_environment_is_left_out():
    now, env, nxt, echo = _environment_world()
    without = _scored(now, np.zeros((now.shape[0], 0)), nxt, echo, seed=1, draws=8)
    given = _scored(now, env, nxt, echo, seed=1, draws=8)
    assert without["leak"] > 0.5
    assert given["leak"] < 0.05 and not any(fold["open"] for fold in given["folds"])


def test_a_broker_kept_in_a_dict_is_read():
    class Organ:
        def __init__(self, level: float) -> None:
            self.state = {"level": level, "gain": 1.0}

    first: dict[str, float] = {}
    second: dict[str, float] = {}
    _numbers(Organ(0.5), "service.broker", first)
    _numbers(Organ(-3.7), "service.broker", second)
    assert first["service.broker.state[level]"] == 0.5
    assert second["service.broker.state[level]"] == -3.7


def test_a_wide_container_enters_as_a_sketch_of_its_values():
    class Organ:
        def __init__(self) -> None:
            self.weights = [float(i) for i in range(50)]

    out: dict[str, float] = {}
    _numbers(Organ(), "service.organ", out)
    assert out["service.organ.weights#"] == 50.0
    assert out["service.organ.weights[].max"] == 49.0


def test_a_closed_world_closes_once_a_margin_is_named():
    from core.subject.closure import ClosureReport

    rng = np.random.default_rng(4)
    rows = 1200
    k = np.zeros((rows + 1, 3))
    for t in range(rows):
        k[t + 1] = 0.7 * k[t] + rng.normal(0, 0.3, 3)
    noise = rng.normal(0, 1, (rows, 2))
    now, nxt = k[:-1], k[1:]
    empty = np.zeros((rows, 0))
    scored = _scored(now, empty, nxt, noise, seed=4, draws=8)
    found = _resolution(now, empty, nxt, noise, seed=4, draws=8)
    report = ClosureReport(
        1.0, 1.0, 1.0, scored["leak"], scored["shuffled_leak"], 0.0, noise.shape[1],
        floor_high=scored["floor_high"], leak_lower=scored["leak_lower"], leak_upper=scored["leak_upper"],
        smallest_leak_resolved=found["smallest_leak_resolved"], margin=0.5,
        leaking_folds=sum(1 for fold in scored["folds"] if fold["open"]),
    )
    assert found["detected"]
    assert report.verdict == "CLOSED_WITHIN_MARGIN"


def test_a_carrier_held_back_only_by_an_undecided_closure_is_unresolved_not_absent():
    nodes = ["A", "B"]
    edges = [("A", "B"), ("B", "A")]
    spectra = {frozenset(nodes): {2.0: 0.01}}
    undecided = select_carriers(nodes, edges, spectra, {frozenset(nodes): (False, 0.02, "UNRESOLVED")})
    assert undecided.status == "UNRESOLVED"
    assert undecided.as_dict()["undecided_by_closure"] == ["AB"]
    shut = select_carriers(nodes, edges, spectra, {frozenset(nodes): (False, 0.2, "OPEN")})
    assert shut.status == "NOT_FOUND"
    found = select_carriers(nodes, edges, spectra, {frozenset(nodes): (True, 0.0, "CLOSED_WITHIN_MARGIN")})
    assert found.status == "FOUND"


def test_the_bridge_reports_an_undecided_carrier_as_unresolved():
    from core.subject.bridge import carrier_term

    term = carrier_term({
        "authority": {"authoritative": True},
        "exclusion": {"status": "UNRESOLVED", "undecided_by_closure": ["ACDGIMNPSW"], "frontier": []},
    })
    assert term.status == "UNRESOLVED"
    assert any("closure of ACDGIMNPSW is unresolved" in b for b in term.blockers)
