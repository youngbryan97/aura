"""Exact/read-basis observations reach the real baseline callback or fail closed."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.declared_observations import ObservationError, ObservationPlan
from core.subject.state import Organs


def declaration(mode="exact"):
    entry = {"unit": "organ:substrate", "path": ["weights"], "shape": [2, 3], "mode": mode}
    if mode == "read_basis":
        entry["readers"] = [[1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0.]]
    return {"version": 1, "max_bytes": 1_000_000, "arrays": [entry]}


def test_exact_mode_preserves_all_coordinates_and_has_a_fixed_named_schema():
    plan = ObservationPlan(declaration())
    unit = SimpleNamespace(weights=np.arange(6.).reshape(2, 3))
    first = plan.read_units({"organ:substrate": unit})
    unit.weights += 10.
    second = plan.read_units({"organ:substrate": unit})
    assert list(first.values()) == list(range(6))
    assert list(second.values()) == list(range(10, 16))
    assert first.keys() == second.keys()
    assert all(name.startswith("held.") for name in first)
    assert plan.description()["rows"] == 2


def test_read_basis_preserves_the_declared_reads():
    spec = declaration("read_basis")
    plan = ObservationPlan(spec)
    unit = SimpleNamespace(weights=np.arange(6.).reshape(2, 3))
    read = np.array(list(plan.read_units({"organ:substrate": unit}).values()))
    axes = plan.arrays[0].axes
    readers = np.asarray(spec["arrays"][0]["readers"])
    np.testing.assert_allclose(readers @ axes.T @ read, readers @ unit.weights.ravel(), atol=1e-12)
    assert plan.width == 2


@pytest.mark.parametrize("bad", [None, np.zeros(6), np.full((2, 3), np.nan), np.zeros((2, 3), dtype=complex)])
def test_a_missing_or_changed_array_does_not_become_zero_evidence(bad):
    plan = ObservationPlan(declaration())
    with pytest.raises(ObservationError):
        plan.read_units({"organ:substrate": SimpleNamespace(weights=bad)})
    assert plan.rows == 0


def test_an_outside_service_cannot_be_relabelled_held_by_an_observation_plan():
    spec = declaration()
    spec["arrays"][0]["unit"] = "outside_service"
    with pytest.raises(ObservationError, match="outside the declared carrier"):
        ObservationPlan(spec)


def test_a_held_root_cannot_smuggle_another_units_state_through_a_reference():
    spec = declaration()
    spec["arrays"][0]["path"] = ["other", "weights"]
    other = SimpleNamespace(weights=np.zeros((2, 3)))
    plan = ObservationPlan(spec)
    with pytest.raises(ObservationError, match="crossed"):
        plan.read_units({"organ:substrate": SimpleNamespace(other=other), "other_unit": other})


def test_budget_is_checked_before_reading_and_while_rows_accumulate():
    spec = declaration()
    spec["max_bytes"] = 4 * 8 * 6 * 512
    plan = ObservationPlan(spec)
    plan.check_budget(512)
    with pytest.raises(ObservationError, match="budget"):
        plan.check_budget(513)
    plan.rows = 512
    with pytest.raises(ObservationError, match="budget"):
        plan.read_units({})


def test_changing_the_declared_reads_changes_the_plan_identity():
    spec = declaration("read_basis")
    changed = deepcopy(spec)
    changed["arrays"][0]["readers"][0][1] = 1.
    assert ObservationPlan(spec).digest != ObservationPlan(changed).digest


def test_the_baseline_callback_keeps_declared_coordinates_at_turn_close(monkeypatch):
    import core.subject.closure as closure
    from tools.run_subject_core import _record

    substrate = SimpleNamespace(weights=np.zeros((2, 3)))
    organs = Organs(substrate=substrate)
    monkeypatch.setattr(closure, "read_periphery", lambda *_: {"outside": 3.})

    class Runtime:
        kernel = SimpleNamespace(_phases=[])
        def __init__(self):
            self.organs = organs
        async def turn_once(self, condition, on_frame):
            substrate.weights += condition
            reading = SimpleNamespace(tag=closure.TURN_CLOSES, misses={})
            on_frame(reading)
            return [reading]

    plan = ObservationPlan(declaration())
    frames, (matrix, names) = asyncio.run(_record(Runtime(), [1., 2.], 2, observations=plan))
    own = [i for i, name in enumerate(names) if name.startswith("held.declared_")]
    assert len(frames) == 4 and plan.rows == 4 and len(own) == 6
    np.testing.assert_array_equal(matrix[:, own], np.repeat([[1.], [3.], [4.], [6.]], 6, axis=1))


def test_a_recording_over_budget_never_advances_a_turn():
    from tools.run_subject_core import _record

    spec = declaration()
    spec["max_bytes"] = 1
    runtime = SimpleNamespace()
    with pytest.raises(ObservationError, match="budget"):
        asyncio.run(_record(runtime, [1.], 2, observations=ObservationPlan(spec)))


def test_an_integer_array_that_float64_cannot_preserve_is_refused():
    plan = ObservationPlan(declaration())
    value = np.full((2, 3), 2**53 + 1, dtype=np.int64)
    with pytest.raises(ObservationError, match="numeric precision"):
        plan.read_units({"organ:substrate": SimpleNamespace(weights=value)})
