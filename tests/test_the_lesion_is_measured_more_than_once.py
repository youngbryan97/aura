"""A deficit between two single readings has no scale to be read against.

The lesion ran one intact arm, one cut arm and one rescued arm, and both
criteria were decided from those three numbers. Whether the cut cost anything
was three comparisons of one reading against one reading, and whether the
rescue brought anything home was a ratio of two such differences. Neither has
an error bar, so neither could say whether it had found an effect or a run.

The same budget now buys several shorter cycles. That returns a spread, and
with a spread the run can say what it could have detected — a lesion reporting
"no deficit" from an experiment too small to find one has reported the size of
its own budget.

Two more things come out of cycling. Each cycle measures a plain life twice
before cutting anything, so a deficit smaller than ordinary drift is drift. And
which side of the partition runs first alternates, because running one side
always first puts the other side's whole life later in the cycle and confounds
time with which half of the partition a domain is in.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

from core.subject.state import DOMAINS
from tools.run_subject_core import _lesion, _lesion_power, _watched_across_the_cut

pytestmark = pytest.mark.unit

MEASURES = ("phi_do", "spread", "synergy")


@dataclasses.dataclass
class _Frame:
    values: dict[str, Any]
    misses: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _Condition:
    name: str


@dataclasses.dataclass
class _Phi:
    best_cut: tuple[tuple[str, ...], tuple[str, ...]]


class _Args:
    def __init__(self, *, rounds: int = 6, cycles: int = 3) -> None:
        self.lesion_rounds = rounds
        self.lesion_cycles = cycles
        self.trials = 6
        self.seed = 7


class _Runtime:
    """A life that records which arm it is in, and answers on a schedule.

    ``held`` is whatever the clamp is holding at the moment, so an arm can be
    told apart from the one beside it without the test reaching into the
    runner.
    """

    def __init__(self) -> None:
        self.held: tuple[str, ...] = ()
        self.order: list[tuple[str, ...]] = []
        self.turns = 0
        self.restores = 0

    async def turn_once(self, condition: Any) -> list[_Frame]:
        self.turns += 1
        return [_Frame(values={d: float(self.turns) for d in DOMAINS})]

    def snapshot(self) -> str:
        return f"snapshot-{self.turns}"

    def restore(self, token: str) -> None:
        self.restores += 1


def _harness(*, cut_costs: float, rescue_returns: float, cycles: int = 3, drift: float = 0.0):
    """Everything ``_lesion`` is handed, with the arms answering by design."""
    runtime = _Runtime()
    conditions = [_Condition(f"c{i}") for i in range(4)]

    class _Clamp:
        def __init__(self, _runtime: Any, domains: Any) -> None:
            self.domains = tuple(domains)

        def __enter__(self):
            runtime.held = self.domains
            runtime.order.append(self.domains)
            return self

        def __exit__(self, *exc):
            runtime.held = ()
            return False

    state = {"arm": 0}

    def build_recording(frames: Any, notes: dict[str, Any] | None = None) -> Any:
        label = (notes or {}).get("arm", "")

        class _Recording:
            def by_turn(self_inner) -> Any:
                return {"arm": label, "held": runtime.held, "rows": len(frames)}

        return _Recording()

    def phi_do(recording: Any, at: Any = None) -> Any:
        held = recording["held"]
        base = 1.0
        # Composed cut recordings are built while nothing is held, so the cut
        # arm is recognised by the runner having just clamped both sides.
        value = base
        if recording["arm"] == "cut":
            value = base - cut_costs
        elif recording["arm"] == "rescued":
            value = base - cut_costs + rescue_returns
        elif recording["arm"] == "drift":
            value = base + drift
        elif held:
            value = base - cut_costs
        return type("P", (), {"phi": value})()

    def synergy_suite(recording: Any, seed: int = 0) -> list[Any]:
        return [type("S", (), {"normalised": phi_do(recording).phi})()]

    def perturbational_complexity(results: Any, source: str = "") -> Any:
        return type("C", (), {"spread": float(results["value"])})()

    async def run_interventions(_runtime: Any, conds: Any, **kw: Any) -> Any:
        state["arm"] += 1
        held = runtime.held
        value = 1.0 - cut_costs if held else 1.0
        return {"value": value}

    return dict(
        runtime=runtime,
        conditions=conditions,
        phi=_Phi(best_cut=(("P", "I", "A"), tuple(d for d in DOMAINS if d not in "PIA"))),
        args=_Args(cycles=cycles),
        build_recording=build_recording,
        clamped=_Clamp,
        phi_do=phi_do,
        perturbational_complexity=perturbational_complexity,
        synergy_suite=synergy_suite,
        run_interventions=run_interventions,
        scale=None,
    )


def _run(**kw: Any) -> dict[str, Any]:
    return _run_with_runtime(**kw)[0]


def _run_with_runtime(**kw: Any) -> tuple[dict[str, Any], _Runtime]:
    parts = _harness(**kw)
    return asyncio.run(_lesion(**parts)), parts["runtime"]


def test_the_cut_is_made_and_released_more_than_once() -> None:
    out = _run(cut_costs=0.5, rescue_returns=0.4)
    assert out["cycles"] == 3
    assert len(out["per_cycle"]) == 3


def test_the_side_that_runs_first_alternates() -> None:
    """Otherwise time is confounded with which half of the partition it is."""
    out = _run(cut_costs=0.5, rescue_returns=0.4)
    first = [cycle["left_ran_first"] for cycle in out["per_cycle"]]
    assert first == [True, False, True]


def test_each_cycle_drives_a_different_rotation_of_the_workload() -> None:
    out = _run(cut_costs=0.5, rescue_returns=0.4)
    rotations = [tuple(cycle["conditions"]) for cycle in out["per_cycle"]]
    assert len(set(rotations)) == 3
    assert set(rotations[0]) == set(rotations[1])


def test_a_deficit_inside_the_drift_between_two_uncut_arms_is_drift() -> None:
    out = _run(cut_costs=0.02, rescue_returns=0.02, drift=0.5)
    assert out["deficit_over_drift"]["phi_do"] is False
    assert "phi_do" not in out["judged_on"]


def test_a_deficit_larger_than_the_drift_is_judged() -> None:
    out = _run(cut_costs=0.5, rescue_returns=0.4, drift=0.01)
    assert out["deficit_over_drift"]["phi_do"] is True
    assert out["deficit"] is True
    assert out["rescued_ok"] is True


def test_a_rescue_that_returns_too_little_fails_however_many_cycles() -> None:
    out = _run(cut_costs=0.5, rescue_returns=0.05)
    assert out["deficit"] is True
    assert out["rescued_ok"] is False


def test_how_many_cycles_each_measure_held_in_is_reported() -> None:
    """A criterion carried by one cycle out of three is a different fact from
    one that held every time, and the verdict cannot show the difference."""
    out = _run(cut_costs=0.5, rescue_returns=0.4)
    held = out["held_across_cycles"]["phi_do"]
    assert held["of"] == 3
    assert held["fell_when_cut"] == 3
    assert held["returned_when_released"] == 3


def test_the_rescue_continues_the_lesioned_individual() -> None:
    """A restore before the rescue arm would measure a fresh baseline.

    Two restores a cycle and no more: one between the two clamped sides, one
    before the node clamp. A third would be the rescue starting somewhere the
    lesion never happened, which is a different individual.
    """
    out, runtime = _run_with_runtime(cut_costs=0.5, rescue_returns=0.4)
    assert runtime.restores == 2 * out["cycles"]


def test_every_cycle_clamps_both_sides_of_the_partition() -> None:
    """Each side runs once with the other held, and the two are composed. One
    clamp a cycle would be the node lesion the equation does not state."""
    out, runtime = _run_with_runtime(cut_costs=0.5, rescue_returns=0.4)
    left, right = tuple(out["severed"]["left"]), tuple(out["severed"]["right"])
    for index in range(out["cycles"]):
        held = runtime.order[index * 3 : index * 3 + 2]
        assert set(held) == {left, right}


def test_one_cycle_says_it_cannot_speak_to_power() -> None:
    """A power of zero from an unmeasurable spread reads as a refusal the
    experiment never made."""
    out = _run(cut_costs=0.5, rescue_returns=0.4, cycles=1)
    assert out["cycles"] == 1
    assert out["power"]["measured"] is False


def test_power_names_the_smallest_deficit_the_cycles_could_have_found() -> None:
    every = [
        {"deltas": {"phi_do": 0.10}, "recovery": {"phi_do": 0.08}},
        {"deltas": {"phi_do": 0.12}, "recovery": {"phi_do": 0.09}},
        {"deltas": {"phi_do": 0.11}, "recovery": {"phi_do": 0.07}},
    ]
    out = _lesion_power(every, ("phi_do",))
    assert out["measured"] is True
    row = out["lesion"]["phi_do"]
    assert row["observed"] == pytest.approx(0.11, abs=1e-6)
    assert row["detectable"] < row["observed"]
    assert row["powered"] is True
    assert row["cycles_needed"] == 1


def test_a_deficit_under_the_detectable_size_is_reported_unpowered() -> None:
    """The finding that matters: a null result from too small an experiment."""
    every = [
        {"deltas": {"phi_do": 0.30}, "recovery": {"phi_do": 0.0}},
        {"deltas": {"phi_do": -0.28}, "recovery": {"phi_do": 0.0}},
        {"deltas": {"phi_do": 0.01}, "recovery": {"phi_do": 0.0}},
    ]
    row = _lesion_power(every, ("phi_do",))["lesion"]["phi_do"]
    assert row["powered"] is False
    assert row["cycles_needed"] > 3


def test_the_spread_is_read_from_both_sides_of_the_cut() -> None:
    """run_023 read A, G and S against a P|C cut, none of which the cut could
    reach less, and spread did not move in any arm."""
    rest = ("I", "A", "G", "S", "M", "W", "D", "N")
    watched = _watched_across_the_cut(("P", "C"), rest)
    assert len(watched) == 3
    assert set(watched) & {"P", "C"}
    assert set(watched) & set(rest)
    assert watched[0] in {"P", "C"}, "the isolated side goes first"


def test_the_sources_do_not_depend_on_which_side_is_named_first() -> None:
    rest = ("I", "A", "G", "S", "M", "W", "D", "N")
    assert _watched_across_the_cut(("P", "C"), rest) == _watched_across_the_cut(rest, ("P", "C"))


def test_a_cut_with_a_one_domain_side_still_reads_three_sources() -> None:
    rest = ("P", "I", "A", "G", "S", "M", "W", "D", "N")
    watched = _watched_across_the_cut(("C",), rest)
    assert len(watched) == 3 and watched[0] == "C"
