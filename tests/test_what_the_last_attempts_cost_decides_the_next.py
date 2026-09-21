"""Two readings that change what she may reach for, and are changed by it.

"In the End" holds both halves of one arithmetic: the effort was spent, and it
did not convert. "How to Love" holds a different one: a capability absent
because nobody showed her, which is not the same absence as one she has seen
done and still cannot manage.

Neither is a readout here. The capability engine decides how expensive a tool
may be before it is offered, and which tools are offered at all, and both
readings move that decision:

* while effort and return are not moving together, the cost ceiling comes down
  by one, so the next attempts are cheaper — and those attempts are the pairs
  the reading is taken from;
* a capability tried, never managed and never seen done is not offered, so the
  turn is not spent on a retry with nothing new behind it. A request by name
  goes through anyway, and a success on that route is the example arriving,
  which takes the capability back out of the list.

Both loops close: the reading changes the action, and the action's outcome
comes back into the reading.
"""

from __future__ import annotations

import pytest

from core.self.converted import ConversionLedger, get_conversion_ledger
from core.self.converted import reset_for_test as reset_conversion
from core.self.never_taught import TeachingLedger, get_teaching_ledger
from core.self.never_taught import reset_for_test as reset_teaching


@pytest.fixture(autouse=True)
def _clean():
    reset_conversion()
    reset_teaching()
    yield
    reset_conversion()
    reset_teaching()


# ── what the effort turned into ──────────────────────────────────────


def test_nothing_finished_is_not_a_reading() -> None:
    reading = ConversionLedger().read()
    assert reading.measured is False
    assert reading.converting is False


def test_effort_and_return_moving_together_reads_as_converting() -> None:
    ledger = ConversionLedger()
    for index in range(10):
        ledger.note(effort=float(index + 1), returned=float(index + 1))
    reading = ledger.read()
    assert reading.measured is True
    assert reading.converting is True
    assert reading.conversion > 0.9


def test_paying_more_and_getting_less_reads_as_not_converting() -> None:
    ledger = ConversionLedger()
    for index in range(10):
        ledger.note(effort=float(index + 1), returned=float(10 - index))
    reading = ledger.read()
    assert reading.measured is True
    assert reading.converting is False
    assert reading.conversion < 0.0


def test_the_effort_stands_whatever_it_returned() -> None:
    """The line does not say the trying was not real."""
    ledger = ConversionLedger()
    for _ in range(10):
        ledger.note(effort=3.0, returned=0.0)
    assert ledger.read().stood == pytest.approx(30.0)


def test_one_series_that_never_varies_says_so() -> None:
    ledger = ConversionLedger()
    for _ in range(10):
        ledger.note(effort=2.0, returned=1.0)
    reading = ledger.read()
    assert reading.measured is True
    assert reading.converting is False
    assert "same" in reading.why


# ── which kind of missing ────────────────────────────────────────────


def test_tried_and_never_managed_and_never_shown_is_its_own_kind() -> None:
    ledger = TeachingLedger()
    for _ in range(4):
        ledger.attempted("how to love", managed=False)
    reading = ledger.read()
    assert reading.never_taught == ("how to love",)
    assert reading.failing == ()
    assert reading.untried == ()


def test_an_example_moves_it_out_of_never_taught() -> None:
    ledger = TeachingLedger()
    for _ in range(4):
        ledger.attempted("how to love", managed=False)
    assert ledger.read().never_taught == ("how to love",)
    ledger.shown("how to love")
    after = ledger.read()
    assert after.never_taught == ()
    assert after.failing == ("how to love",)


def test_credit_she_does_not_take_is_counted() -> None:
    ledger = TeachingLedger()
    ledger.attempted("write the note", managed=True, credited=False)
    ledger.attempted("write the note", managed=True, credited=True)
    assert ledger.read().credit_not_taken == pytest.approx(0.5)


# ── the loops, at the gate ───────────────────────────────────────────


def _engine():
    from core.capability_engine import CapabilityEngine

    return CapabilityEngine.__new__(CapabilityEngine)


def test_not_converting_takes_a_step_off_the_ceiling() -> None:
    engine = _engine()
    assert engine._conversion_penalty() == 0, "no reading yet is no penalty"

    ledger = get_conversion_ledger()
    for index in range(10):
        ledger.note(effort=float(index + 1), returned=float(10 - index))
    assert engine._conversion_penalty() == 1

    reset_conversion()
    ledger = get_conversion_ledger()
    for index in range(10):
        ledger.note(effort=float(index + 1), returned=float(index + 1))
    assert engine._conversion_penalty() == 0


def test_a_blind_retry_is_not_offered_and_a_named_request_still_is() -> None:
    engine = _engine()
    teaching = get_teaching_ledger()
    for _ in range(3):
        teaching.attempted("build_app", managed=False)

    assert engine._never_taught("build_app") is True
    assert engine._never_taught("code_repl") is False

    # The one route through is the request by name, and a success on it is the
    # example arriving.
    teaching.shown("build_app")
    assert engine._never_taught("build_app") is False


def test_the_gate_asks_before_it_offers() -> None:
    """The withholding is in the offer path, not only in the ledger."""
    from pathlib import Path

    engine = Path("core/capability_engine.py").read_text(encoding="utf-8")
    assert "if not requested and self._never_taught(skill_name):" in engine
    assert "allowed_max_cost - self._conversion_penalty()" in engine
    assert "self._note_what_the_attempt_cost(skill_name, ctx, result)" in engine
