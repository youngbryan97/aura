"""A successful compounding cycle, recorded as a failed weight update.

The real weight-update branch asked the compounding scheduler for a cycle and
called it a success when the status was "promoted". That is not a value the
compounding contract has ever produced: a cycle ends as a candidate or as a
qualified adapter, and moving the active model pointer is a separate staged
act on purpose.

So a cycle that trained and qualified an adapter came back as a failed weight
update, which is the whole of weight-level self-improvement reporting that it
did not happen. And because the rollback only runs when a weight action
SUCCEEDED, that could not run either — one wrong string and two mechanisms
that could not fire.

The unit tests could not see it. Their learner returns a boolean, so the run
never reaches the scheduler branch at all: a double that does not track the
contract of the thing it stands in for reports the opposite of the truth.
"""

from __future__ import annotations

import pytest

from core.learning.recursive_self_improvement import _the_weight_update_worked
from core.learning.weight_compounding import WORKED, CycleReceipt


def test_promoted_is_not_a_status_this_contract_produces():
    """The literal the caller was checking for."""
    assert "promoted" not in WORKED


@pytest.mark.parametrize("status", sorted(WORKED))
def test_a_cycle_that_produced_something_reads_as_worked(status):
    assert CycleReceipt(generation_id="g1", status=status).worked()
    assert _the_weight_update_worked(CycleReceipt(generation_id="g1", status=status))


@pytest.mark.parametrize("status", ["refused", "blocked", "failed", "deferred"])
def test_a_cycle_that_produced_nothing_does_not(status):
    assert not CycleReceipt(generation_id="g1", status=status).worked()
    assert not _the_weight_update_worked(CycleReceipt(generation_id="g1", status=status))


def test_the_receipt_as_the_scheduler_actually_hands_it_over():
    """run_cycle_now hands back the dict form, not the object."""
    for status in sorted(WORKED):
        said = CycleReceipt(generation_id="g1", status=status).to_dict()
        assert _the_weight_update_worked(said), status
    for status in ("refused", "blocked", "failed"):
        said = CycleReceipt(generation_id="g1", status=status).to_dict()
        assert not _the_weight_update_worked(said), status


def test_the_status_the_caller_used_to_look_for_is_still_refused():
    """A receipt claiming a status this contract does not define is not a pass."""
    assert not _the_weight_update_worked({"status": "promoted"})
    assert not _the_weight_update_worked({"status": ""})
    assert not _the_weight_update_worked({})


def test_a_learner_that_answers_with_a_boolean_still_works():
    """The injected doubles the gauntlet uses must keep working."""
    assert _the_weight_update_worked(True)
    assert not _the_weight_update_worked(False)


def test_every_status_the_contract_names_is_decided_one_way_or_the_other():
    """A status nobody classified would silently read as a failure."""
    named = {"candidate", "qualified_adapter", "refused", "blocked", "failed"}
    assert WORKED <= named
    for status in named:
        decided = CycleReceipt(generation_id="g1", status=status).worked()
        assert decided is (status in WORKED)
