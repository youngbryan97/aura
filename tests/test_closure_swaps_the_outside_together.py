"""A joint cause can be invisible to every one-unit interchange."""
from dataclasses import dataclass, field

from tools.run_interventional_closure import anchor_schedule, hybrid_snapshot, joint_snapshot


@dataclass
class Snapshot:
    services: dict
    state: tuple = (73, "held")
    phases: dict = field(default_factory=dict)
    organs: dict = field(default_factory=dict)
    module_state: dict = field(default_factory=dict)


def test_joint_interchange_finds_an_and_cause_the_individual_swaps_miss():
    receiver = Snapshot({"a": {"level": 0}, "b": {"level": 0}, "core": {"level": 7}})
    donor = Snapshot({"a": {"level": 1}, "b": {"level": 1}, "core": {"level": 99}})
    units = [("a", None), ("b", None)]

    def future(s):
        return s.services["a"]["level"] * s.services["b"]["level"]

    assert all(future(hybrid_snapshot(receiver, donor, *unit)) == future(receiver) for unit in units)
    swapped = joint_snapshot(receiver, donor, units)
    assert future(swapped) == 1
    assert swapped.services["core"] == receiver.services["core"]
    assert swapped.state == receiver.state
    assert receiver.services["a"]["level"] == 0


def test_joint_swap_keeps_receiver_instants_and_module_identity():
    identity = object()
    receiver = Snapshot(
        {"a": {"level": 0, "last_update": 1_790_000_000.}},
        module_state={"x": (identity, {"outside": 0, "held": 7})},
    )
    donor = Snapshot(
        {"a": {"level": 1, "last_update": 1_790_000_100.}},
        module_state={"x": (object(), {"outside": 1, "held": 99})},
    )
    swapped = joint_snapshot(receiver, donor, [("a", None), ("module:x", "outside")])
    assert swapped.services["a"] == {"level": 1, "last_update": 1_790_000_000.}
    assert swapped.module_state["x"][0] is identity
    assert swapped.module_state["x"][1] == {"outside": 1, "held": 7}


def test_an_identical_donor_does_not_count_as_an_intervention():
    snapshot = Snapshot({"a": {"level": 0}})
    assert joint_snapshot(snapshot, snapshot, [("a", None)]) is None


def test_nonuniform_donors_do_not_all_hide_a_period_thirteen_cause():
    positions = anchor_schedule("13,26,39,53,66,79,92,105", 8, 13)
    distances = [abs(positions[(a + offset) % 8] - positions[a])
                 for a in range(8) for offset in (1, 4)]
    assert any(distance % 13 != 0 for distance in distances)
    assert all(any(distance % period for distance in distances) for period in range(2, 106))
    assert anchor_schedule("", 8, 13) == list(range(13, 105, 13))


def test_the_collector_uses_the_explicit_turn_positions():
    import asyncio
    from types import SimpleNamespace

    import numpy as np

    from core.subject.v25_runtime import collect_anchor_bank

    class Runtime:
        turn = 0
        async def turn_once(self, condition):
            self.turn += 1
            value = self.turn
            return [SimpleNamespace(vector=lambda: np.array([value]))]
        def snapshot(self):
            return self.turn

    bank = asyncio.run(collect_anchor_bank(
        Runtime(), [SimpleNamespace(name="one"), SimpleNamespace(name="two")],
        rounds=5, history_turns=1, take_at=frozenset({3, 6, 10}),
    ))
    assert [anchor.snapshot for anchor in bank] == [3, 6, 10]
