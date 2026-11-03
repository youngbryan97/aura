"""Why her domains cut apart read as integrated: the clamp holds her portrait, and each domain can live in its own order.

On 3 November her portrait's column moved while S was held, because the portrait recomputes from the choice history
D writes; and every domain alone had lived the same order of conditions. These pin the repair of the first and the
reading of the second.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.subject.clamp import PINNED_READERS, _pin_readers
from tools.read_disconnected_diagnostic import read

pytestmark = pytest.mark.unit


class Portrait:
    """Reads a history it does not own, as her portrait reads the choices D makes."""

    def __init__(self, history):
        self.history = history

    def columns(self):
        return {"values_foretell_choice": float(len(self.history))}


def _runtime(history):
    return SimpleNamespace(organs=SimpleNamespace(portrait=Portrait(history)))


def test_a_held_self_keeps_its_portrait_while_the_choices_it_reads_go_on():
    history = [1, 2]
    runtime = _runtime(history)
    pinned = _pin_readers(runtime, ["S"])
    assert [method for _, method in pinned] == ["columns"]
    history.extend([3, 4, 5])
    assert runtime.organs.portrait.columns() == {"values_foretell_choice": 2.0}
    for organ, method in pinned:
        vars(organ).pop(method, None)
    assert runtime.organs.portrait.columns() == {"values_foretell_choice": 5.0}


def test_a_domain_that_does_not_read_the_portrait_leaves_it_live():
    history = [1]
    runtime = _runtime(history)
    assert _pin_readers(runtime, ["D"]) == []
    history.append(2)
    assert runtime.organs.portrait.columns() == {"values_foretell_choice": 2.0}


def test_the_pinned_value_cannot_be_changed_through_what_it_returns():
    runtime = _runtime([1])
    _pin_readers(runtime, ["S"])
    runtime.organs.portrait.columns()["values_foretell_choice"] = 99.0
    assert runtime.organs.portrait.columns() == {"values_foretell_choice": 1.0}
    assert PINNED_READERS == {"portrait": "columns"}


def _report(shared, own, leaks=None, own_leaks=None):
    life = lambda phi, se: {"phi_do": phi, "standard_error": se}  # noqa: E731
    return {
        "rounds": 80,
        "intact": life(0.017, 0.013),
        "drift": life(0.012, 0.005),
        "disconnected": life(*shared),
        "disconnected_shuffled": life(*own),
        "held_columns_that_moved": leaks or {d: {} for d in "PIAGCSMWDN"},
        "held_columns_that_moved_shuffled": own_leaks or {d: {} for d in "PIAGCSMWDN"},
    }


def test_the_shared_order_explaining_it_reads_as_predicted():
    reading = read(_report(shared=(0.011, 0.0036), own=(0.0005, 0.003)))
    assert all(reading["rules"].values()) and reading["as_predicted"]


def test_integration_without_a_shared_order_is_not_explained_by_it():
    reading = read(_report(shared=(0.011, 0.0036), own=(0.010, 0.0035)))
    assert not reading["rules"]["2_own_order_reads_none"]
    assert not reading["rules"]["3_shared_order_adds_integration"]
    assert not reading["as_predicted"]


def test_a_column_that_still_moves_fails_the_fix():
    leaks = {d: {} for d in "PIAGCSMWDN"}
    leaks["D"] = {"S.values_foretell_choice": 0.5}
    reading = read(_report(shared=(0.011, 0.0036), own=(0.0005, 0.003), leaks=leaks))
    assert not reading["rules"]["0_the_fix_holds"] and not reading["as_predicted"]


def test_a_report_without_the_own_order_is_refused():
    report = _report(shared=(0.011, 0.0036), own=(0.0, 0.003))
    report["disconnected_shuffled"] = None
    with pytest.raises(ValueError):
        read(report)
