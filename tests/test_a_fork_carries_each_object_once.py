"""A snapshot carries each object once, and a value that did not move is not copied again.

On 22 September one snapshot of the organism held 230 MB. The services hold
each other as fields: the authority keeps the somatic gate, the gate keeps
interoception, and three of them keep the neurochemical system. Every owner
copied what it held, so the neurochemical system went into each snapshot five
times. The organs are services too, and each was copied under both names. A
sweep worker keeps 128 snapshots, and six workers restarted the machine.

Measured on a ten-round organism (`probe_snapshot_memory.py`): 235 MB a
snapshot before, 45 MB with each object carried once, 13 MB with unchanged
fields shared between snapshots.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from core.subject import copies
from core.subject import snapshot as snap
from core.subject.copies import identical
from core.subject.snapshot import _organ_state, _restore_organ, _service_state

pytestmark = pytest.mark.unit


class _Chemistry:
    def __init__(self) -> None:
        self.levels = np.linspace(0.0, 1.0, 64)
        self.history: list[float] = [0.1, 0.2]


class _Gate:
    def __init__(self, chemistry: _Chemistry) -> None:
        self.chemistry = chemistry
        self.threshold = 0.5


class _Authority:
    def __init__(self, gate: _Gate, chemistry: _Chemistry) -> None:
        self.gate = gate
        self.chemistry = chemistry
        self.decisions = deque([1, 2], maxlen=8)


@pytest.fixture
def container(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    chemistry = _Chemistry()
    gate = _Gate(chemistry)
    authority = _Authority(gate, chemistry)
    built = {
        "neurochemical_system": chemistry,
        "somatic_marker_gate": gate,
        "substrate_authority": authority,
        # One object under a second name, as the container registers it.
        "neurochemistry": chemistry,
    }
    monkeypatch.setattr(snap, "_built_services", lambda: dict(built))
    monkeypatch.setattr(copies, "_LAST_COPY", {})
    return built


def test_a_service_held_by_another_is_not_copied_under_its_owner(container) -> None:
    carried = frozenset(id(obj) for obj in container.values())
    saved = _service_state(skip=carried)
    assert "chemistry" not in saved["substrate_authority"]
    assert "gate" not in saved["substrate_authority"]
    assert "chemistry" not in saved["somatic_marker_gate"]
    # Its own state is still carried, under its own name.
    assert saved["neurochemical_system"]["history"] == [0.1, 0.2]


def test_one_object_under_two_names_is_captured_once(container) -> None:
    saved = _service_state(skip=frozenset(id(obj) for obj in container.values()))
    assert "neurochemical_system" in saved
    assert "neurochemistry" not in saved


def test_a_service_that_is_an_organ_is_left_to_the_organs(container) -> None:
    chemistry = container["neurochemical_system"]
    saved = _service_state(organs=frozenset({id(chemistry)}))
    assert "neurochemical_system" not in saved
    assert "neurochemistry" not in saved


def test_without_the_skip_every_owner_copies_what_it_holds(container) -> None:
    """The control: the old capture, and the duplication the skip removes."""
    saved = _service_state()
    assert "chemistry" in saved["substrate_authority"]
    assert saved["substrate_authority"]["chemistry"] is not saved["neurochemical_system"]


def test_restoring_the_owner_leaves_the_shared_service_to_its_own_entry(container) -> None:
    carried = frozenset(id(obj) for obj in container.values())
    saved = _service_state(skip=carried)
    chemistry = container["neurochemical_system"]
    chemistry.history.append(0.9)
    chemistry.levels[0] = 7.0
    for name, fields in saved.items():
        _restore_organ(container[name], fields)
    assert chemistry.history == [0.1, 0.2]
    assert chemistry.levels[0] == 0.0
    # The owner still points at the live object, not at a copy of it.
    assert container["substrate_authority"].chemistry is chemistry


def test_a_field_that_did_not_move_is_the_same_copy_in_both_snapshots(container) -> None:
    chemistry = container["neurochemical_system"]
    first = _organ_state(chemistry)
    second = _organ_state(chemistry)
    assert second["levels"] is first["levels"]
    chemistry.levels[3] = 0.25
    third = _organ_state(chemistry)
    assert third["levels"] is not second["levels"]
    assert third["levels"][3] == 0.25
    assert second["levels"][3] != 0.25


def test_an_arm_that_mutates_after_a_restore_cannot_reach_a_shared_copy(container) -> None:
    chemistry = container["neurochemical_system"]
    first = _organ_state(chemistry)
    second = _organ_state(chemistry)
    _restore_organ(chemistry, second)
    chemistry.history.append(5.0)
    chemistry.levels[:] = 9.0
    assert first["history"] == [0.1, 0.2] and second["history"] == [0.1, 0.2]
    assert float(first["levels"].max()) == 1.0


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (0.0, -0.0),
        (np.zeros(3), np.zeros(3, dtype=np.float32)),
        (np.array([0.0]), np.array([-0.0])),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
        (deque([1], maxlen=2), deque([1], maxlen=3)),
        ([1, [2]], [1, [3]]),
        (1, True),
    ],
)
def test_any_difference_a_restore_could_see_counts(left, right) -> None:
    assert not identical(left, right)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        np.array([np.nan, 1.0]),
        {"a": [1, {"b": np.arange(4)}]},
        deque([1, 2], maxlen=5),
        _Gate(_Chemistry()),
    ],
)
def test_a_copy_is_identical_to_its_original(value) -> None:
    import copy

    assert identical(value, copy.deepcopy(value))


def test_a_cycle_does_not_recurse_forever() -> None:
    import copy

    loop: list = [1]
    loop.append(loop)
    assert identical(loop, copy.deepcopy(loop))
